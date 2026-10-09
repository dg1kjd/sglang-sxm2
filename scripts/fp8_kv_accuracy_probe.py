#!/usr/bin/env python3
"""FP8 E4M3 vs FP16 KV cache accuracy battery for GLM-5.3-Flash on SM70.

Usage: accuracy_ab.py <tag> <max_ctx_tokens>

Writes /home/sg/acc/acc_<tag>.json. Every task is deterministic (temperature 0)
so the two configs are directly comparable.
"""
import json, math, os, random, re, sys, time, urllib.request

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:28088")
GEN = BASE + "/generate"
CHAT = BASE + "/v1/chat/completions"
ACC = os.environ.get("ACC_DIR", "/home/sg/acc")
CORP = ACC + "/corpora"

TAG = sys.argv[1] if len(sys.argv) > 1 else "unset"
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 480000
OUT = {}
RATIO = None          # tokens per char of the long document, measured once
# Single-token encodings of the option letters (verified with the checkpoint tokenizer).
OPT_TOKENS = {" A": 362, " B": 425, " C": 356, " D": 422}

# ---------------------------------------------------------------- plumbing
def call(text, *, max_new=1, start=0, top=None, timeout=5400):
    body = {"text": text,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0},
            "return_logprob": True, "logprob_start_len": start}
    if top is not None:
        body["top_logprobs_num"] = top
    req = urllib.request.Request(GEN, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    d["_wall"] = time.time() - t0
    return d


def count_tokens(text):
    return call(text, start=-1)["meta_info"]["prompt_tokens"]


def nll(lps, lo=1, hi=None):
    vals = [x[0] for x in lps[lo:hi] if x[0] is not None]
    if not vals:
        return None
    n = len(vals)
    mn = -sum(vals) / n
    return {"n": n, "mean_nll": round(mn, 5), "ppl": round(math.exp(min(20.0, mn)), 4)}


def log(msg):
    print("[%s] %s" % (TAG, msg), flush=True)


# ------------------------------------------------------- T1 short perplexity
def t1_short():
    res = {}
    for name in ("en", "zh", "code"):
        p = os.path.join(CORP, name + ".txt")
        text = open(p, encoding="utf8").read()
        d = call(text, start=0, top=20)
        mi = d["meta_info"]
        lps, tl = mi["input_token_logprobs"], mi["input_top_logprobs"]
        half = len(lps) // 2
        pos = [[e[0], e[1]] for e in tl if e]
        step = max(1, len(pos) // 1200)
        res[name] = {"tokens": mi["prompt_tokens"], "wall": round(d["_wall"], 1),
                     "full": nll(lps, 1), "second_half": nll(lps, half),
                     "chars": len(text),
                     "topk": [p[:20] for p in pos[::step]]}
        log("T1 %-5s tokens=%d PPL=%s" % (name, mi["prompt_tokens"], res[name]["full"]["ppl"]))
    return res


# -------------------------------------------------------- T2 long-context NLL
def t2_long():
    doc = open(os.path.join(CORP, "long.txt"), encoding="utf8").read()
    cuts = [c for c in (16000, 64000, 128000, 256000, 400000) if c <= MAX_CTX]
    biggest = max(cuts)
    text = doc[: int(biggest / RATIO)]
    d = call(text, start=0)                            # one prefill -> whole curve
    mi = d["meta_info"]
    lps = mi["input_token_logprobs"]
    total = mi["prompt_tokens"]
    out = {"prompt_tokens": total, "wall": round(d["_wall"], 1), "cuts": []}
    for c in cuts:
        s = nll(lps, max(1, c - 2048), c)
        if s:
            out["cuts"].append({"at": c, **s})
    log("T2 long tokens=%d cuts=%s" % (total, [c["at"] for c in out["cuts"]]))
    return out


# -------------------------------------------------------------- T3 MCQ set
# (question, [4 options], index of the correct option)
MCQ = [
 ("17 × 23 等于多少？", ["391", "381", "401", "411"], 0),
 ("240 的 15% 是多少？", ["36", "24", "32", "48"], 0),
 ("一列火车 3 小时行驶 240 公里，平均速度是多少 km/h？", ["80", "60", "70", "90"], 0),
 ("97 之后的下一个质数是？", ["101", "99", "103", "107"], 0),
 ("解方程 3x + 7 = 22，x = ?", ["5", "3", "7", "15"], 0),
 ("169 的平方根是？", ["13", "17", "11", "19"], 0),
 ("一个矩形边长 12 和 5，对角线长度为？", ["13", "17", "12", "11"], 0),
 ("六边形内角和是多少度？", ["720", "540", "900", "360"], 0),
 ("2 的 10 次方等于？", ["1024", "512", "2048", "1000"], 0),
 ("把 0.375 化成最简分数是？", ["3/8", "3/4", "1/3", "2/5"], 0),
 ("金的化学元素符号是？", ["Au", "Ag", "Gd", "Go"], 0),
 ("哪颗行星被称为红色星球？", ["火星", "金星", "木星", "水星"], 0),
 ("细胞中进行有氧呼吸产生能量的主要结构是？", ["线粒体", "细胞核", "核糖体", "高尔基体"], 0),
 ("真空中的光速约为？", ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s", "1.5×10^8 m/s"], 0),
 ("植物进行光合作用主要吸收什么气体？", ["二氧化碳", "氧气", "氮气", "氢气"], 0),
 ("碳的原子序数是？", ["6", "12", "8", "14"], 0),
 ("地壳中含量最多的元素是？", ["氧", "硅", "铝", "铁"], 0),
 ("地球大气中含量最多的气体是？", ["氮气", "氧气", "二氧化碳", "氩气"], 0),
 ("使行星围绕太阳运行的力是？", ["万有引力", "电磁力", "强核力", "摩擦力"], 0),
 ("25°C 时中性水溶液的 pH 值约为？", ["7", "5", "9", "1"], 0),
 ("澳大利亚的首都是？", ["堪培拉", "悉尼", "墨尔本", "珀斯"], 0),
 ("世界上面积最大的海洋是？", ["太平洋", "大西洋", "印度洋", "北冰洋"], 0),
 ("亚马逊雨林主要位于哪个国家？", ["巴西", "秘鲁", "哥伦比亚", "委内瑞拉"], 0),
 ("海拔最高的山峰是？", ["珠穆朗玛峰", "乔戈里峰", "干城章嘉峰", "洛子峰"], 0),
 ("长城位于哪个国家？", ["中国", "印度", "蒙古", "韩国"], 0),
 ("吉萨金字塔由哪个古代文明建造？", ["古埃及", "古巴比伦", "古希腊", "古罗马"], 0),
 ("第二次世界大战在哪一年结束？", ["1945", "1944", "1946", "1939"], 0),
 ("美国面积最大的州是？", ["阿拉斯加", "德克萨斯", "加利福尼亚", "蒙大拿"], 0),
 ("加拿大的首都是？", ["渥太华", "多伦多", "温哥华", "蒙特利尔"], 0),
 ("分隔欧洲与非洲的海是？", ["地中海", "红海", "黑海", "波罗的海"], 0),
 ("“欲穷千里目”的下一句是？", ["更上一层楼", "白日依山尽", "黄河入海流", "一览众山小"], 0),
 ("中国古代四大发明中用于航海定向的是？", ["指南针", "造纸术", "火药", "印刷术"], 0),
 ("长江发源于哪个高原？", ["青藏高原", "云贵高原", "黄土高原", "内蒙古高原"], 0),
 ("“三人行，必有我师焉”出自哪部典籍？", ["《论语》", "《孟子》", "《庄子》", "《道德经》"], 0),
 ("中国现存最长的古代城墙是？", ["长城", "西安城墙", "南京城墙", "开封城墙"], 0),
 ("成语“守株待兔”比喻什么？", ["死守经验不知变通", "勤奋努力终有收获", "做事果断迅速", "善于抓住机会"], 0),
 ("二十四节气中北半球白昼最长的是？", ["夏至", "冬至", "春分", "立夏"], 0),
 ("《红楼梦》的作者是？", ["曹雪芹", "吴承恩", "施耐庵", "罗贯中"], 0),
 ("中国最长的河流是？", ["长江", "黄河", "珠江", "黑龙江"], 0),
 ("“落霞与孤鹜齐飞”出自哪篇文章？", ["《滕王阁序》", "《岳阳楼记》", "《醉翁亭记》", "《赤壁赋》"], 0),
 ("5 台机器 5 分钟生产 5 个零件，100 台机器生产 100 个零件需要多久？",
  ["5 分钟", "100 分钟", "20 分钟", "50 分钟"], 0),
 ("球拍和球共 1.10 元，球拍比球贵 1.00 元，球多少钱？",
  ["0.05 元", "0.10 元", "0.15 元", "0.01 元"], 0),
 ("若 A > B 且 B > C，则下列必然成立的是？", ["A > C", "A < C", "A = C", "无法确定"], 0),
 ("数列 2, 6, 12, 20, 30, ? 的下一项是？", ["42", "40", "36", "44"], 0),
 ("数列 1, 1, 2, 3, 5, 8, ? 的下一项是？", ["13", "11", "12", "15"], 0),
 ("医生给你 3 粒药，每 30 分钟吃一粒，吃完需要多长时间？",
  ["60 分钟", "90 分钟", "30 分钟", "120 分钟"], 0),
 ("下列哪一项与其他三项不是同一类？", ["立方体", "正方形", "三角形", "圆形"], 0),
 ("若所有 Bloops 都是 Razzies，所有 Razzies 都是 Lazzies，则所有 Bloops 都是 Lazzies 吗？",
  ["是", "否", "无法确定", "只有在部分情况下成立"], 0),
 ("“有些 S 是 P，所有 P 都是 Q”，可以必然推出？", ["有些 S 是 Q", "所有 S 都是 Q", "所有 Q 都是 S", "没有 S 是 Q"], 0),
 ("100 人参加单败淘汰赛，共需进行多少场比赛？", ["99", "100", "50", "199"], 0),
 ("在 Python 中 len([1,2,3]) 返回什么？", ["3", "2", "4", "报错"], 0),
 ("哪种数据结构的存取顺序是先进先出（FIFO）？", ["队列", "栈", "堆", "树"], 0),
 ("在含 n 个元素的有序数组上做二分查找，时间复杂度是？",
  ["O(log n)", "O(n)", "O(n log n)", "O(1)"], 0),
 ("在 Python 中 7 // 2 的结果是？", ["3", "3.5", "4", "2"], 0),
 ("SQL 中 SELECT 语句的作用是？", ["查询数据", "插入数据", "删除表", "修改结构"], 0),
 ("HTTP 状态码 404 表示？", ["未找到", "服务器错误", "未授权", "请求超时"], 0),
 ("在 Python 中 type(3.0) 返回什么？", ["<class 'float'>", "<class 'int'>", "<class 'double'>", "<class 'number'>"], 0),
 ("十进制 5 的二进制表示是？", ["101", "110", "111", "100"], 0),
 ("下列哪一项不是 Python 内置数据类型？", ["array", "list", "tuple", "dict"], 0),
 ("git 中创建并切换到新分支的命令是？", ["git checkout -b", "git branch", "git merge", "git clone"], 0),
]


def t3_mcq():
    rows = []
    for i, (q, opts, ci) in enumerate(MCQ):
        base = q + "\n\n" + "\n".join("%s. %s" % (L, o) for L, o in zip("ABCD", opts)) + "\n\nAnswer:"
        lps, ids = [], []
        for L in "ABCD":
            d = call(base + " " + L, start=0)
            last = d["meta_info"]["input_token_logprobs"][-1]
            lps.append(last[0]); ids.append(last[1])
        if ids != [OPT_TOKENS[" " + L] for L in "ABCD"]:
            log("T3 WARN q%d option token ids %s (expected %s)"
                % (i, ids, [OPT_TOKENS[" " + L] for L in "ABCD"]))
        pred = max(range(4), key=lambda k: lps[k])
        wrong = [lps[k] for k in range(4) if k != ci]
        rows.append({"i": i, "q": q[:40], "lps": [round(x, 4) for x in lps],
                     "correct": ci, "pred": pred,
                     "margin": round(lps[ci] - max(wrong), 4)})
        if (i + 1) % 15 == 0:
            log("T3 %d/%d acc=%.3f" % (i + 1, len(MCQ),
                                       sum(r["pred"] == r["correct"] for r in rows) / len(rows)))
    acc = sum(r["pred"] == r["correct"] for r in rows) / len(rows)
    return {"n": len(rows), "accuracy": round(acc, 4),
            "mean_logprob_correct": round(sum(r["lps"][r["correct"]] for r in rows) / len(rows), 4),
            "mean_margin": round(sum(r["margin"] for r in rows) / len(rows), 4),
            "rows": rows}


# ------------------------------------------------- T4/T5 long-context recall
FILLER_LINES = None


def build_haystack(target_tokens):
    """Insert 4 needle records and 12 fact records into a real long document."""
    doc = open(os.path.join(CORP, "long.txt"), encoding="utf8").read()
    text = doc[: int(target_tokens / RATIO)]
    lines = text.split("\n")
    rnd = random.Random(20261008)
    needles, facts = [], []
    for k in range(8):
        sec = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        needles.append({"pos": round(0.08 + 0.12 * k, 2), "id": "R%d" % (k + 1), "secret": sec})
    mats = [("耐张线夹", 7), ("悬垂线夹", 3), ("扁铁抱箍", 12), ("余缆架", 2), ("分纤箱", 5),
            ("接头盒", 9), ("ONU", 4), ("FTU", 6), ("PE 管", 15), ("PVC 管", 8),
            ("波纹管", 11), ("钢绞线", 21), ("拉线棒", 4), ("绝缘子", 18), ("避雷器", 6),
            ("隔离开关", 2), ("熔断器", 9), ("电缆终端", 3), ("电缆中间接头", 5), ("光缆终端盒", 7),
            ("光纤配线架", 4), ("尾纤", 24), ("法兰盘", 8), ("横担", 6), ("拉线盘", 3),
            ("底盘", 2), ("卡盘", 4), ("接地极", 12), ("接地线", 30), ("U 型挂环", 9),
            ("球头挂环", 9), ("碗头挂板", 9), ("延长环", 5), ("楔形线夹", 7), ("并沟线夹", 14),
            ("电缆支架", 16)]
    for j, (mat, qty) in enumerate(mats):
        facts.append({"id": "F%02d" % (j + 1), "pole": "%03d号杆" % (100 + j * 7), "mat": mat, "qty": qty})
    rnd.shuffle(facts)
    for nd in needles:
        at = int(len(lines) * nd["pos"])
        lines.insert(at, "\n【巡检记录 %s】本工程的验收暗号是 %s。\n" % (nd["id"], nd["secret"]))
    step = len(lines) // (len(facts) + 1)
    for j, f in enumerate(facts):
        at = min(len(lines) - 1, step * (j + 1) + rnd.randint(0, max(0, step // 2)))
        lines.insert(at, "\n【材料记录 %s】杆号 %s 使用 %s，数量 %d。\n" % (f["id"], f["pole"], f["mat"], f["qty"]))
    return "\n".join(lines), needles, facts


def t45_recall(target):
    text, needles, facts = build_haystack(target)
    out = {"prompt_tokens": None, "needles": [], "facts": None}
    for nd in needles:
        q = text + "\n\n问题：巡检记录 %s 里的验收暗号是什么？只输出暗号本身。" % nd["id"]
        d = call(q, max_new=64, start=-1)
        ans = d["text"]
        if out["prompt_tokens"] is None:
            out["prompt_tokens"] = d["meta_info"]["prompt_tokens"]
        out["needles"].append({"id": nd["id"], "pos": nd["pos"], "secret": nd["secret"],
                               "hit": nd["secret"] in ans,
                               "exact": ans.strip().split("\n")[0].strip() == nd["secret"],
                               "answer": ans.strip()[:60],
                               "wall": round(d["_wall"], 1), "cached": d["meta_info"].get("cached_tokens")})
        log("T4 needle %s pos=%.1f hit=%s wall=%.1fs" % (nd["id"], nd["pos"], out["needles"][-1]["hit"], d["_wall"]))
    q = text + ("\n\n问题：把上文中所有【材料记录 F01】到【材料记录 F%02d】的内容整理成 JSON 数组，"
                "每个元素形如 {\"id\":\"F01\",\"pole\":\"101号杆\",\"mat\":\"耐张线夹\",\"qty\":7}。"
                "只输出 JSON。" % len(facts))
    d = call(q, max_new=2600, start=-1)
    ans = d["text"]
    got = []
    a = ans.find("[")
    if a >= 0:
        try:
            got, _ = json.JSONDecoder().raw_decode(ans[a:])
        except Exception:
            got = []
    if not isinstance(got, list):
        got = []
    idx = {}
    for r in got:
        try:
            idx[str(r.get("id", "")).strip().upper().replace(" ", "")] = r
        except Exception:
            pass

    def norm(x):
        return re.sub(r"\s+", "", str(x)).upper()

    ok = 0
    detail = []
    for f in facts:
        r = idx.get(f["id"], {})
        same = (norm(r.get("pole", "")) == norm(f["pole"])
                and norm(r.get("mat", "")) == norm(f["mat"])
                and norm(r.get("qty", "")) == norm(f["qty"]))
        ok += bool(same)
        detail.append({"id": f["id"], "ok": bool(same), "got": {k: r.get(k) for k in ("pole", "mat", "qty")}})
    out["facts"] = {"n": len(facts), "recall": round(ok / len(facts), 4),
                    "parsed": len(got), "wall": round(d["_wall"], 1),
                    "detail": detail, "raw_head": ans[:1200]}
    log("T5 multifact recall=%.3f (%d/%d)" % (ok / len(facts), ok, len(facts)))
    return out


# --------------------------------------------------- T6 greedy completions
GREEDY = [
 "用一句话解释张量并行。",
 "把下面句子翻译成英文：今天施工完成了92号杆的接地装置安装。",
 "列出三种常见的电力金具名称，用逗号分隔。",
 "Write a Python one-liner that reverses a string s.",
 "用 JSON 表示：杆号 043，材料 耐张线夹，数量 2。只输出 JSON。",
 "简述配电网自动化的三个主要功能。",
 "如果一台变压器容量为 400 kVA，功率因数为 0.9，请写出有功功率的计算式。",
 "解释什么是 KV cache，并说明它为什么能加速推理。",
 "把 3.14159 保留两位小数。",
 "给出 1 到 10 的平方数列表，用逗号分隔。",
 "用一句话说明光纤熔接损耗的主要来源。",
 "写一段 50 字左右的施工安全提示。",
]


def t6_greedy():
    rows = []
    for p in GREEDY:
        d = call(p, max_new=256, start=-1)
        rows.append({"prompt": p, "out": d["text"], "tokens": d["meta_info"]["completion_tokens"],
                     "wall": round(d["_wall"], 1)})
    log("T6 greedy %d prompts done" % len(rows))
    return rows


# ------------------------------------------------------------------- main
def main():
    global RATIO
    os.makedirs(ACC, exist_ok=True)
    doc = open(os.path.join(CORP, "long.txt"), encoding="utf8").read()
    RATIO = count_tokens(doc[:40000]) / 40000.0
    print("[%s] long-doc ratio %.4f tokens/char" % (TAG, RATIO), flush=True)
    OUT["tag"] = TAG
    OUT["max_ctx"] = MAX_CTX
    OUT["ratio"] = RATIO
    OUT["t1"] = t1_short()
    OUT["t2"] = t2_long()
    OUT["t3"] = t3_mcq()
    OUT["t45"] = t45_recall(min(MAX_CTX, 270000))
    OUT["t45_400k"] = t45_recall(400000) if MAX_CTX >= 400000 else None
    OUT["t6"] = t6_greedy()
    with open("%s/acc_%s.json" % (ACC, TAG), "w") as f:
        json.dump(OUT, f, ensure_ascii=False)
    log("T1 PPL: " + " ".join("%s=%.4f" % (k, v["full"]["ppl"]) for k, v in OUT["t1"].items()))
    log("T2 cuts: " + " ".join("%d:%.4f" % (c["at"], c["ppl"]) for c in OUT["t2"]["cuts"]))
    log("T3 acc=%.4f meanlogp=%.4f margin=%.4f" % (OUT["t3"]["accuracy"],
        OUT["t3"]["mean_logprob_correct"], OUT["t3"]["mean_margin"]))
    log("T4 needle hit %d/%d" % (sum(n["hit"] for n in OUT["t45"]["needles"]), len(OUT["t45"]["needles"])))
    log("T5 multifact recall=%.3f" % OUT["t45"]["facts"]["recall"])
    if OUT["t45_400k"]:
        log("T4b 400k needle hit %d/%d recall=%.3f"
            % (sum(n["hit"] for n in OUT["t45_400k"]["needles"]), len(OUT["t45_400k"]["needles"]),
               OUT["t45_400k"]["facts"]["recall"]))
    log("done -> %s/acc_%s.json" % (ACC, TAG))


if __name__ == "__main__":
    main()
