#!/usr/bin/env python3
"""Final analysis: FP8 vs FP16 KV accuracy, with a same-config noise floor and a
context-length control. Emits Markdown to /home/sg/acc/summary.md."""
import json, math, os

ACC = os.environ.get("ACC_DIR", "/home/sg/acc")
RUNS = ["fp16", "fp16b", "fp8_288k", "fp8_480k"]
D = {}
for r in RUNS:
    p = "%s/acc_%s.json" % (ACC, r)
    if os.path.exists(p):
        D[r] = json.load(open(p))
    else:
        print("  (missing run %s)" % r)

L = []
def w(s=""):
    L.append(s); print(s)


def kl_pair(p, q):
    ids = set(p) | set(q)
    pp = {i: math.exp(p.get(i, -30.0)) for i in ids}
    qq = {i: math.exp(q.get(i, -30.0)) for i in ids}
    zp = sum(pp.values()) or 1.0
    zq = sum(qq.values()) or 1.0
    k1 = k2 = 0.0
    for i in ids:
        a, b = pp[i] / zp, qq[i] / zq
        if a > 0 and b > 0:
            k1 += a * math.log(a / b); k2 += b * math.log(b / a)
        elif a > 0:
            k1 += a * 30.0
        elif b > 0:
            k2 += b * 30.0
    return (k1 + k2) / 2.0


def shift(x, y):
    """Compare two runs' stored top-k arrays for one corpus."""
    out = {}
    for name in ("en", "zh", "code"):
        a, b = D[x]["t1"][name]["topk"], D[y]["t1"][name]["topk"]
        n = min(len(a), len(b)); agree = 0; kls = []; mass = []; nll = []
        for i in range(n):
            pa = {e[1]: e[0] for e in a[i]}
            pb = {e[1]: e[0] for e in b[i]}
            if not pa or not pb:
                continue
            ta, tb = max(pa, key=pa.get), max(pb, key=pb.get)
            agree += int(ta == tb)
            kls.append(kl_pair(pa, pb))
            inter = set(pa) & set(pb)
            sa = sum(math.exp(v) for v in pa.values()) or 1.0
            mass.append(sum(math.exp(pa[i2]) for i2 in inter) / sa)
            nll.append(abs(pa[ta] - pb.get(ta, -30.0)))
        m = len(kls) or 1
        out[name] = {"n": m, "top1_agree": agree / m, "sym_kl": sum(kls) / m,
                     "overlap": sum(mass) / m, "dlogp_top1": sum(nll) / m}
    return out


w("# FP8 E4M3 与 FP16 KV cache 的精度对比\n")
w("模型：GLM-5.3-Flash-NVFP4（W4A16, Marlin）　主机：8× V100-SXM2-32GB，TP8，MTP 3/4")
w("采样：全部 temperature 0，同一批固定语料与固定提示，因此两次运行可直接对比。\n")

w("## 0. 运行说明与噪声下限\n")
w("| 运行 | 说明 | kv dtype | context |")
w("|---|---|---|---|")
w("| fp16 | 冷启动基准 | fp16 | %s |" % D["fp16"]["max_ctx"])
if "fp16b" in D:
    w("| fp16b | 同配置重复，用于度量噪声 | fp16 | %s |" % D["fp16b"]["max_ctx"])
if "fp8_288k" in D:
    w("| fp8_288k | FP8，与 fp16 同 context（隔离 context 变量） | fp8_e4m3 | %s |" % D["fp8_288k"]["max_ctx"])
if "fp8_480k" in D:
    w("| fp8_480k | FP8 生产配置 | fp8_e4m3 | %s |" % D["fp8_480k"]["max_ctx"])
w("")

# ---------------- T1
w("## 1. 短上下文困惑度（T1）\n")
cols = [r for r in RUNS if r in D]
w("| 语料 | tokens | " + " | ".join(cols) + " | FP8−FP16 |")
w("|---|---|" + "---|" * len(cols) + "---|")
for name, label in (("en", "英文维基（电力/变压器/光纤）"), ("zh", "中文维基 + 中文技术文档"), ("code", "SGLang Python 源码")):
    row = ["%.4f" % D[r]["t1"][name]["full"]["ppl"] for r in cols]
    ref = D["fp8_288k"] if "fp8_288k" in D else D[cols[-1]]
    d = 100 * (ref["t1"][name]["full"]["ppl"] / D["fp16"]["t1"][name]["full"]["ppl"] - 1)
    w("| %s | %d | %s | %+.3f%% |" % (label, D["fp16"]["t1"][name]["tokens"], " | ".join(row), d))
w("")

# ---------------- noise floor
w("## 2. 运行间噪声下限（同配置重复）\n")
if "fp16b" in D:
    ident = all(D["fp16"]["t1"][n]["topk"] == D["fp16b"]["t1"][n]["topk"] for n in ("en", "zh", "code"))
    w("`fp16` 与 `fp16b` 的 T1 困惑度与 top-20 logprob 数组：**%s**。" %
      ("逐位完全相同" if ident else "存在差异"))
    w("")
    w("| 语料 | fp16 PPL | fp16b PPL | 差异 |")
    w("|---|---|---|---|")
    for n in ("en", "zh", "code"):
        x, y = D["fp16"]["t1"][n]["full"]["ppl"], D["fp16b"]["t1"][n]["full"]["ppl"]
        w("| %s | %.4f | %.4f | %+.4f%% |" % (n, x, y, 100 * (y / x - 1)))
    w("")
    w("T2/T4/T6 两次运行有差异，原因是第二次运行时**前缀缓存已热**，chunk 切分不同；"
      "T1 与 T3 在冷/热两种状态下都完全一致，因此以它们作为主判据。\n")

# ---------------- distribution shift
w("## 3. 下一 token 分布差异（T1 语料，每个位置 top-20）\n")
pairs = [("fp16", "fp16b", "噪声下限 fp16 vs fp16b"),
         ("fp16", "fp8_288k", "对照 fp16 vs fp8（同 context）")]
if "fp8_288k" in D and "fp8_480k" in D:
    pairs.append(("fp8_288k", "fp8_480k", "对照 fp8@288k vs fp8@480k"))
w("| 对比 | 语料 | 位置数 | top-1 一致率 | 对称 KL | top-20 交集概率质量 | top-1 logprob 平均绝对差 |")
w("|---|---|---|---|---|---|---|")
for x, y, label in pairs:
    if x not in D or y not in D:
        continue
    s = shift(x, y)
    for n in ("en", "zh", "code"):
        v = s[n]
        w("| %s | %s | %d | %.4f | %.5f | %.4f | %.4f |" % (label, n, v["n"], v["top1_agree"],
          v["sym_kl"], v["overlap"], v["dlogp_top1"]))
w("")

# ---------------- T2
w("## 4. 长上下文困惑度曲线（T2）\n")
w("同一篇长文档，统计指定位置之前 2048 个 token 的困惑度。\n")
ats = sorted({c["at"] for r in D.values() for c in r["t2"]["cuts"]})
w("| 位置 (tokens) | " + " | ".join(cols) + " |")
w("|---|" + "---|" * len(cols))
for at in ats:
    cells = []
    for r in cols:
        c = {x["at"]: x for x in D[r]["t2"]["cuts"]}.get(at)
        cells.append("%.4f" % c["ppl"] if c else "—")
    w("| %d | %s |" % (at, " | ".join(cells)))
w("")

# ---------------- T3
w("## 5. 选择题（T3）\n")
w("60 题 × 4 选项，用「Answer: X」最后一个 token 的 logprob 判定答案。\n")
w("| 指标 | " + " | ".join(cols) + " |")
w("|---|" + "---|" * len(cols))
for key, label in (("accuracy", "准确率"), ("mean_logprob_correct", "正确选项平均 logprob"), ("mean_margin", "平均 margin")):
    w("| %s | %s |" % (label, " | ".join(
        ("%.4f" % D[r]["t3"][key]) for r in cols)))
if "fp16" in D and "fp8_288k" in D:
    same = sum(1 for a, b in zip(D["fp16"]["t3"]["rows"], D["fp8_288k"]["t3"]["rows"]) if a["pred"] == b["pred"])
    w("\n逐题预测一致（fp16 vs fp8_288k）：%d/60。各自答错的题：" % same)
    for r in cols:
        wrong = [x["i"] for x in D[r]["t3"]["rows"] if x["pred"] != x["correct"]]
        w("- %s: %s" % (r, wrong or "无"))
w("")

# ---------------- T4/T5
w("## 6. 长上下文检索（T4 针 / T5 多事实）\n")
w("| 运行 | 规模 | 针命中 | 针精确匹配 | 多事实召回 |")
w("|---|---|---|---|---|")
for r in cols:
    for key, lab in (("t45", "270k"), ("t45_400k", "400k")):
        v = D[r].get(key)
        if not v:
            continue
        w("| %s | %s | %d/%d | %d/%d | %.4f |" % (r, lab,
          sum(n["hit"] for n in v["needles"]), len(v["needles"]),
          sum(n["exact"] for n in v["needles"]), len(v["needles"]),
          v["facts"]["recall"]))
w("")

# ---------------- T6
w("## 7. 贪心生成一致性（T6）\n")
w("| 对比 | 完全一致的提示数 | 平均公共前缀(字符) |")
w("|---|---|---|")
def cmp6(x, y):
    eq = 0; pref = []
    for a, b in zip(D[x]["t6"], D[y]["t6"]):
        if a["out"] == b["out"]:
            eq += 1; pref.append(len(a["out"])); continue
        n = 0
        for c1, c2 in zip(a["out"], b["out"]):
            if c1 != c2:
                break
            n += 1
        pref.append(n)
    return eq, sum(pref) // len(pref)
for x, y, label in pairs:
    if x in D and y in D:
        eq, pr = cmp6(x, y)
        w("| %s | %d/12 | %d |" % (label, eq, pr))
w("")

open(ACC + "/summary.md", "w").write("\n".join(L) + "\n")
print("\n[written] %s/summary.md" % ACC)
