# FP8 E4M3 KV 缓存对 GLM-5.3-Flash 精度的影响

模型：**GLM-5.3-Flash-NVFP4**（W4A16，Marlin）　主机：**8× V100-SXM2-32GB**，TP8，MTP 3/4
日期：2026-10-08　关联改动：本仓库 `SM70: FP8 E4M3 KV cache for DSA models`

---

## English summary

Storing the DSA/MLA latent KV in unscaled FP8 E4M3 (the `SGLANG_SM70_DSA_FP8_KV`
path) changes the model's next-token distribution only slightly and does not
degrade measurable task quality on this host:

- Two identical FP16 runs are **bit-identical**, so the comparison has a zero
  noise floor; an FP8 run at 288k and one at 480k are also bit-identical, which
  rules out `--context-length` as a confound. Any FP16/FP8 difference is
  therefore caused by the KV dtype alone.
- Perplexity moves by **at most 0.8% and with no consistent sign** (+0.37% on
  code, −0.82% on English Wikipedia, −0.04% on Chinese text at 3-5k tokens;
  ≤0.4% at 16k-256k tokens).
- The next-token distribution does shift: **top-1 agreement 95.5-97.7%**,
  symmetric KL **0.27-0.68 nats**, top-20 probability-mass overlap 97.6-99.0%.
- On task probes, nothing changed: 60-question MCQ accuracy identical (0.967,
  **the same 60/60 predictions**), 8/8 needle recall and 36/36 multi-fact recall
  at 270k tokens, and identical field accuracy (0.5849) on a real extraction
  task.
- Greedy generation diverges after ~26 characters on every prompt. That is the
  expected consequence of any small logit perturbation in an autoregressive
  loop, not by itself a quality regression.
- Cost: decode **+3.1%**, cold prefill **−5.2%**; the KV pool grows from 290,240
  to 521,024 tokens, which is what makes `--context-length 480000` possible.

Net: on this hardware and workload the FP8 KV cache buys 1.8× the KV capacity
for a small, measurable distribution shift and no detectable task-level loss.
It is not a free lunch for workloads that depend on exact token probabilities.

---

## 1. 背景

上游在 SM70 上没有 DSA 的 FP8 KV 选项：`--kv-cache-dtype fp8_e4m3` 会解析到
Hopper 专用的 `flashmla_kv` 后端，而 V100 上没有 `flashmla_ops`。因此 11 个
DSA/MLA 层一直只能用 FP16，KV 池被限制在 290,240 token。

本仓库的改动让这 512 维隐向量以**无缩放的 E4M3 字节**存储，在 `sparse_mla_sm70`
把行搬进共享内存时用软件转换还原成 fp16（Volta 没有 FP8 算术，也没有硬件
FP8→FP16 转换）。每 token 从 12,716 B 降到 7,084 B，池子扩到 521,024 token。
机制细节见 [SM70-DSA-FP8-KV.md](SM70-DSA-FP8-KV.md)。

**量化必然有损，所以"能跑"不等于"能用"。** 本文回答：损失有多大、在哪些任务上
可见。

## 2. 测试方法

### 2.1 配置

| | FP16 基准 | FP8 待测 |
|---|---|---|
| `--kv-cache-dtype` | `auto`（解析为 float16） | `fp8_e4m3` |
| `--context-length` | 288000 | 288000 **和** 480000 各跑一次 |
| `--mem-fraction-static` | 0.945 | 0.945 |
| KV 池 | 290,240 token | 521,024 token |
| 其余全部参数 | 完全相同 | 完全相同 |

采样一律 `temperature=0`；同一批语料、同一批提示。**每次测量都在服务重启后的冷
前缀缓存下进行**，避免缓存状态影响结果。

### 2.2 三个关键的方法学设计

**（a）噪声下限。** 张量并行的浮点归约顺序可能让相同配置的两次运行结果不同，
如果不量化这个噪声，就无法判断观测到的差异是量化造成的还是抖动。因此把 FP16
配置**完整跑了两遍**。

**（b）context 对照。** FP8 因为池子更大而跑在 480k，FP16 只能跑 288k。为了排除
"差异其实来自 context 设置"，额外用 **FP8 + 288k** 跑了一遍，与 FP16 同 context
直接对比。

**（c）单次长请求取整条曲线。** 长上下文困惑度用**一个**长文档请求（`logprob_start_len=0`
返回全部 prompt logprob）在任意位置切片统计，避免为每个长度重复 prefill。

### 2.3 测试项

| 编号 | 内容 | 说明 |
|---|---|---|
| T1 | 短上下文困惑度 | 英文维基（电力/变压器/光纤，3,148 tok）、中文维基+中文技术文档（4,653 tok）、SGLang Python 源码（3,250 tok） |
| — | 下一 token 分布差异 | 每个位置取 top-20 logprob，跨配置计算 top-1 一致率与对称 KL |
| T2 | 长上下文困惑度 | 同一篇 119 个真实源文件拼成的长文档，在 16k/64k/128k/256k/400k 位置前 2,048 token 上统计 |
| T3 | 选择题 | 60 题 × 4 选项（数学、科学、地理历史、中文、逻辑、编程），用「Answer: X」最后一个 token 的 logprob 判定 |
| T4 | 长上下文找针 | 8 条记录埋在 270k 文档的 0.1–0.9 各深度 |
| T5 | 多事实召回 | 36 条材料记录散布在 270k 文档中，要求整篇抽取为 JSON |
| T6 | 贪心生成一致性 | 12 个提示 × 256 token，逐字比对 |
| T7 | 真实任务 | 用户实际业务：从施工日志抽取 4 天 × 按杆号材料明细，与已核对的人工结果逐字段比对 |

---

## 3. 结果

### 3.1 噪声下限与 context 对照（结论的前提）

| 对比 | top-1 一致率 | 对称 KL | T1 困惑度 | T3 逐题 | T6 逐字 |
|---|---|---|---|---|---|
| fp16 vs fp16b（同配置重复） | **1.0000** | **0.00000** | 逐位相同 | 逐题相同 | 12/12 相同 |
| fp8@288k vs fp8@480k（仅 context 不同） | **1.0000** | **0.00000** | 逐位相同 | 逐题相同 | 12/12 相同 |

- **同配置两次运行逐位相同** → 本环境的运行间噪声为零，观测到的差异不是抖动。
- **FP8 在 288k 与 480k 下逐位相同** → `--context-length` 对数值没有影响，
  因此下面 FP16 与 FP8 的差异**只能归因于 KV 数据类型**。

### 3.2 困惑度

**短上下文（T1）**

| 语料 | tokens | FP16 | FP8 | 变化 |
|---|---|---|---|---|
| 英文维基 | 3,148 | 2.0463 | 2.0296 | **−0.82%** |
| 中文维基 + 技术文档 | 4,653 | 2.8892 | 2.8880 | −0.04% |
| SGLang Python 源码 | 3,250 | 3.3779 | 3.3903 | **+0.37%** |

**长上下文（T2）**

| 位置 (tokens) | FP16 | FP8 @288k | FP8 @480k | FP8−FP16 |
|---|---|---|---|---|
| 16,000 | 2.6312 | 2.6386 | 2.6386 | +0.28% |
| 64,000 | 2.3492 | 2.3577 | 2.3577 | +0.36% |
| 128,000 | 1.7890 | 1.7885 | 1.7885 | −0.03% |
| 256,000 | 1.5251 | 1.5218 | 1.5201 | −0.22% |
| 400,000 | 不适用（超出 FP16 池上限） | — | 1.3954 | — |

**解读**：变化幅度全部 ≤0.8%，且**符号不一致**（英文/中文变好、代码变差；16k/64k
变差、128k/256k 变好）。这说明 FP8 带来的是一个**近似零均值的小扰动**，而不是
系统性的精度下降。困惑度对"平均值"敏感，对这种扰动不敏感。

### 3.3 下一 token 分布差异（更灵敏的指标）

| 对比 | 语料 | 位置数 | top-1 一致率 | 对称 KL | top-20 交集概率质量 | top-1 logprob 平均绝对差 |
|---|---|---|---|---|---|---|
| **噪声下限** | en | 1,574 | 1.0000 | 0.00000 | 1.0000 | 0.0000 |
| fp16 vs fp16b | zh | 1,551 | 1.0000 | 0.00000 | 1.0000 | 0.0000 |
| | code | 1,625 | 1.0000 | 0.00000 | 1.0000 | 0.0000 |
| **fp16 vs fp8** | en | 1,574 | 0.9765 | 0.2695 | 0.9903 | 0.1428 |
| （同 context） | zh | 1,551 | 0.9691 | 0.4400 | 0.9842 | 0.2339 |
| | code | 1,625 | 0.9545 | 0.6827 | 0.9760 | 0.3087 |

**解读**：这是本次测试中**最灵敏**的指标，也是唯一清晰显示出 FP8 影响的地方。

- **2.4%–4.6% 的位置，最高概率 token 发生了变化**；代码最敏感（4.6%），英文最不敏感（2.4%）。
- top-20 的概率质量仍有 **97.6%–99.0% 重叠**，说明分布主体没变，只是尾部/边界位置有移动。
- 代码语料的 KL 最高（0.68），与它 PPL 变差方向一致——**代码对 KV 量化最敏感**。

### 3.4 选择题（T3）

| 指标 | FP16 | FP8 |
|---|---|---|
| 准确率 | 0.9667 | **0.9667** |
| 正确选项平均 logprob | −0.2320 | −0.2370 |
| 平均 margin（正确−最优错误） | 6.0342 | 6.0161 |
| **逐题预测一致率** | — | **60/60** |

答错的题完全一样（第 34、46 题）。**没有任何一题的答案因为 FP8 而改变**，只是
置信度有极轻微下降（平均 logprob −0.005，相对 margin 变化 −0.3%）。

### 3.5 长上下文检索（T4/T5）

| 运行 | 规模 | 针命中 | 多事实召回 (36 条) |
|---|---|---|---|
| FP16 | 270k | 8/8 | 1.0000 |
| FP8 @288k | 270k | 8/8 | 1.0000 |
| FP8 @480k | 270k | 8/8 | 1.0000 |
| FP8 @480k | 400k | 7/8 | 0.8056 |

**解读**：在两种配置都能达到的 270k 长度上，FP8 与 FP16 **完全持平且都是满分**。
400k 上的下降（7/8、29/36）无法与 FP16 对比（FP16 的池子上限是 290,240 token），
**不能据此判断是 FP8 造成还是长度本身造成**——它只是说明了 400k 任务本身更难。

### 3.6 真实业务任务（T7）

从施工日志按杆号抽取 4 天材料明细，与人工核对结果逐字段比对：

| | FP16 | FP8 |
|---|---|---|
| 字段准确率 | 0.5849 | **0.5849** |

**逐天、逐字段完全一致**。（该分数偏低是因为评测口径里杆号写法差异，如 `013号杆`
与 `13号杆`——两种配置受到的影响完全相同，属于评测问题而非模型问题。）

### 3.7 贪心生成一致性（T6）

| 对比 | 逐字相同 | 平均公共前缀 |
|---|---|---|
| 噪声下限 fp16 vs fp16b | 12/12 | 458 字符（整段） |
| fp16 vs fp8 | **0/12** | **26 字符** |
| fp8@288k vs fp8@480k | 12/12 | 502 字符 |

**解读**：12 个提示全部在约 26 个字符后分叉。这**不代表质量下降**——自回归生成中
任何微小的 logit 扰动都会在某一步翻转一个 token，之后轨迹完全发散。它说明的是
"FP8 与 FP16 不是逐位等价的两种部署"，对需要可复现输出的场景有意义。

### 3.8 性能（同一测试口径）

| 指标 | FP16 | FP8 | 变化 |
|---|---|---|---|
| 解码速度 | 139.8 tok/s | 144.2 tok/s | **+3.1%** |
| 冷 prefill（5.6 万 token） | 2,215 tok/s | 2,101 tok/s | **−5.2%** |
| 270k 冷 prefill 实测 | ~147 s | ~150 s | 持平 |
| 400k 冷 prefill 实测 | 不适用 | ~244 s | — |

解码变快来自 KV 读取字节减半；prefill 变慢来自软件 FP8 转换的 ALU 开销。

---

## 4. 结论

1. **任务层面没有可测的精度损失。** 选择题 60/60 预测一致、270k 检索满分持平、
   真实业务任务逐字段一致。
2. **但输出分布确实发生了小幅偏移。** 2.4%–4.6% 位置的首选 token 改变，对称 KL
   0.27–0.68，代码语料最敏感。
3. **困惑度变化 ≤0.8% 且符号不一致**，属于零均值扰动，不构成系统性退化。
4. **差异可唯一归因于 KV 数据类型**：同配置重复逐位相同、FP8 在 288k/480k 下逐位
   相同，两个对照都排除了噪声与 context 长度这两个混淆因素。

**建议**

- 需要 1.8 倍 KV 容量（本机即 290,240 → 521,024 token、context 480k）时，**可以
  放心启用 FP8 KV**。
- 对**代码类**长上下文任务保持关注：那是唯一显示出方向性变差的语料（PPL +0.37%、
  KL 0.68、top-1 一致率 95.5%）。
- 如果某个任务出现回归，回退只需 `--kv-cache-dtype auto`（代价是 context 掉回
  290,240 以内）。
- 依赖**精确 token 概率**的工作流（例如用 logprob 做置信度阈值判断、需要逐位可复现
  的输出）应当用 FP16：即使任务指标不变，分布已经不同。

## 5. 局限

1. **单一模型、单一硬件**：结论仅对 GLM-5.3-Flash + V100(SM70) 有效。
2. **T4/T5 在 ≤270k 达到天花板**：两种配置都满分，因此这两项**无法区分**精度差异，
   只能说明"没有严重退化"。真正有区分力的是 T1/T2/T3 与分布差异。
3. **选择题集为自建**（60 题），不是标准基准，用于 A/B 对照有效，但不是绝对能力评测。
4. **无缩放 E4M3**：这是 SM70 原始布局的设计（不带 per-block scale）。若将来改为
   带缩放方案，需重测。
5. **400k 探针没有 FP16 基线**（池上限所限），该数据只能作为 FP8 的绝对表现记录。
6. **对 <1% 的退化检测力有限**：困惑度差异本身就在 ±1% 量级。

## 6. 复现

```bash
# 测试组：/home/sg/acc/accuracy_ab.py，输出 acc_<tag>.json
#   需要先按 build_corpora.sh 构建语料
cd /home/sg/acc
/home/sg/sglang-v100-venv/bin/python accuracy_ab.py fp8_480k 480000   # FP8 生产配置
/home/sg/sglang-v100-venv/bin/python accuracy_ab.py fp16     288000   # FP16 基准

# 汇总（噪声下限、context 对照、分布差异）
/home/sg/sglang-v100-venv/bin/python /home/sg/acc/analyze.py   # -> summary.md
```

原始数据：`acc_fp16.json`、`acc_fp16b.json`、`acc_fp8_288k.json`、`acc_fp8_480k.json`。
