# Hy4-flash on 8× V100

Tencent Hy4-flash on 8× V100-SXM2-32GB, set up for one agentic coding session: a main conversation that grows towards 200k tokens, subagent conversations next to it, and many short tool-call turns. A turn that extends a cached conversation prefills only the new tokens. The model's own DFly drafter proposes a block of six tokens per step, and the target model checks each one before it is kept. The full 262,144-token context fits, with images and video.

## Get the model

The validated checkpoint is Avifenesh's NVFP4 quantisation of Hy4-flash (ModelOpt layout, 151 GB on disk). The DFly drafter and the vision tower are in the same repo. The checkpoint is not an official Tencent release; it carries the base model's Tencent Hy License 1.0.

```bash
pip install -U "huggingface_hub[cli]"
hf download Avifenesh/Hy4-flash-NVFP4 \
  --revision f9eac06cf022cfd3dc2e8ed423922466b2adcd87 \
  --local-dir ~/models/Hy4-flash-NVFP4

export HY4_MODEL=~/models/Hy4-flash-NVFP4
```

| | |
|---|---|
| checkpoint | [`Avifenesh/Hy4-flash-NVFP4`](https://huggingface.co/Avifenesh/Hy4-flash-NVFP4), revision `f9eac06c` |
| base model | [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash), ~220B MoE (~12B active), 512 routed experts |
| quantisation | NVFP4 routed experts run as W4A16 with FP16 activations; FP8 attention projections and Engram tables (fixed-order FP8 GEMM); FP16 KV cache |

## Talking to it

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":1024}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"hy4-flash-nvfp4","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'
```

The model thinks before it answers by default (`hunyuan` reasoning parser), and tool calls come back as structured `tool_calls` (`hunyuan` tool parser). An empty `content` with a large `completion_tokens` means the reply hit `max_tokens` while still thinking.

The chat template knows three levels: `no_think` (no thinking), `high` and `max`. The server maps `reasoning_effort` (OpenAI) or `output_config.effort` (Anthropic) onto them:

| requested | template level |
|---|---|
| nothing, `minimal`, `low`, `medium`, `high` | `high` |
| `xhigh`, `max` | `max` |
| `none` | `no_think` |

Anthropic `thinking: {"type": "disabled"}` selects `no_think`, and `enabled` or `adaptive` selects `high`; an `output_config.effort` in the same request wins. Claude Code sends its effort setting as `output_config.effort`, `xhigh` by default, so it runs at `max`.

## Measured performance

The reference recipe below: TP=8, DFly block 6, `--max-running-requests 1`, vision on. KV pool 286,720 tokens (28,672 of them in the sliding-window layers), context 262,144.

**Decode** (2026-11-24). A chat request for a short Python `merge_sorted` function, streamed through `/v1/chat/completions` (~580 generated tokens at temperature 0), 3 runs each; tok/s counts from the first streamed token to the last. Accept length is tokens per target forward.

| sampling | tok/s | accept length |
|---|---:|---:|
| temperature 0 | 196–197 | 4.11 |
| temperature 1 | 162–179 | 3.4–3.8 |

Past the first few requests of a launch, three launches agree within 0.5 tok/s at temperature 0; the whole reply takes 3.2 s including its 0.2 s first token. One or two early requests after a launch wait up to 1.4 s for their first token. At temperature 1 the rate follows each sampled reply's length and acceptance.

**Agent session** (2026-11-24). Through `/v1/chat/completions` with a `bash` tool and thinking on: source files as the first user message, then three turns that each append an assistant tool call and its result (~500–600 new tokens). TTFT is the first streamed token and includes rendering and tokenizing the whole chat on the server.

| context | cold first turn | tool turns 1 / 2 / 3 |
|---|---:|---:|
| 32,229 tokens | 7.7 s | 269 / 243 / 238 ms |
| 128,228 tokens | 32.5 s | 361 / 304 / 341 ms |
| 190,229 tokens | 50.4 s | 382 / 370 / 368 ms |

A cold prompt prefills at 3.8–4.2k tokens/s.

**Main session plus subagents** (2026-11-24). A 190k main conversation, then two fresh ~19k subagent conversations of three turns each, with a main turn after each subagent. The main turns stay cached:

| turn | TTFT |
|---|---:|
| main, cold / next turn | 50.0 s / 398 ms |
| subagent 1, cold / turns 2–3 | 4.52 s / 241, 242 ms |
| main | 371 ms |
| subagent 2, cold / turns 2–3 | 4.53 s / 239, 246 ms |
| main | 369 ms |

## Images and video

Images and videos go in as `image_url` and `video_url` content parts of a chat message (OpenAI API), or as `image` blocks (Anthropic API). The vision tower's weights stay in pinned host memory and are copied to the GPU for each pass, then freed, so they take no KV pool. Image preprocessing runs on the CPU, and encoded images are cached in host memory, so an image that spans several prefill chunks is encoded once.

**Measured** (2026-11-24). Requests with a short answer, image not seen before, the reference recipe. Every answer was correct.

| input | prompt tokens | time |
|---|---:|---:|
| image 448×448 | 264 | 0.4 s |
| image 2048×1536 | 3,174 | 1.3 s |
| image 3840×2160 | 4,182 | 2.0 s |
| video 640×360 | 3,872 | 1.8 s |

GPU memory: the largest request took at most 310 MiB more on the first GPU and about 2 MiB on the others.

| knob | default | effect |
|---|---|---|
| `HY4_VISION` | 1 | `0` serves text only (`--language-model-only`) and leaves a little more KV pool |
| `HY4_VIDEO_MAX_TOKENS` | 32768 | Token budget per video; a video above it is scaled down to fit. Every GPU holds a video's embedding while it is prefilled |
| `SGLANG_VLM_CACHE_SIZE_MB` | 512 | Encoded-image cache per GPU, in pinned host memory (`SGLANG_USE_HOST_VLM_CACHE=1`) |

## Accuracy

Measured with [sgl-eval](https://github.com/sgl-project/sgl-eval) 0.1.3 against the server, the model's recommended sampling (temperature 0.9, top_p 1.0), default thinking, DFly on:

| eval | score | settings |
|---|---:|---|
| GSM8K, 200 questions | 97.5% | max_tokens 16,384, 0 truncated (2026-11-23) |
| RULER2 at 32k, 20 per task | 87.7% | max_tokens 8,192; 11 of 240 answers hit the cap, 9 of them in `mv_niah_basic`. At max_tokens 32,768 the four `mv_niah` tasks scored 100 / 93.8 / 95 / 89.8 with none capped (2026-11-23) |

For comparison, the SGLang cookbook lists 96.4–97.0% on the full GSM8K for the FP8 and BF16 builds on Hopper and Blackwell GPUs (sgl-eval default sampling). With thinking on, give long-context requests a `max_tokens` of at least 32,768: on retrieval tasks the model copies what it finds into its thinking.

Prefill and decode logprobs of the same tokens agree to an average KL of 0.08 per token over 32 sequences of 512 generated tokens. Decoding on a cached prefix gives 0.10, within the spread of the per-sequence values, with no sequence standing out (2026-11-23). Greedy output with DFly is not token-for-token identical to greedy output without it.

## Reference recipe

The wrapper is the supported entry and the recipe the numbers above were measured on. DFly is on by default; `HY4_DFLY=0` turns it off.

```bash
export HY4_MODEL=~/models/Hy4-flash-NVFP4
bash scripts/serve_hy4_flash_nvfp4_v100.sh
```

Expanded (what that script runs). The env block is not implied by the CLI flags: it turns on the fixed-order GEMM and the two-level all-reduce.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_ALGO=allreduce:tree
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
export SGLANG_DSV41_HIER_AR=1
export SGLANG_DSV41_HIER_AR_CA=1
export SGLANG_DSV41_HIER_AR_PUSH=1
export SGLANG_USE_SM70_FIXED_ORDER_GEMM=1
export SGLANG_GRAMMAR_DISK_CACHE_DIR=~/.cache/sglang/grammar
export SGLANG_USE_HOST_VLM_CACHE=1
export SGLANG_FORCE_CPU_IMAGE_PREPROCESSING=1
export SGLANG_VLM_CACHE_SIZE_MB=512

python -m sglang.launch_server \
  --model-path "${HY4_MODEL}" \
  --served-model-name hy4-flash-nvfp4 \
  --trust-remote-code \
  --reasoning-parser hunyuan \
  --tool-call-parser hunyuan \
  --dtype float16 \
  --quantization modelopt_mixed \
  --fp4-gemm-backend marlin \
  --tensor-parallel-size 8 \
  --ep-size 1 \
  --attention-backend dsa \
  --kv-cache-dtype auto \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode 1 \
  --max-running-requests 1 \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --warmups sampling \
  --sleep-on-idle \
  --context-length 262144 \
  --mem-fraction-static 0.88 \
  --mm-process-config '{"video": {"max_pixels": 33554432}}' \
  --speculative-algorithm DFLY \
  --speculative-dfly-block-size 6 \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| DFly | on, block 6 | On an earlier build greedy decode went from 61.5 tok/s without DFly to 150 with it (block 6, accept length 4.0). The drafter costs about 3% of the KV pool: 296,640 tokens without it, 286,720 with |
| `SGLANG_USE_SM70_FIXED_ORDER_GEMM` | 1 | Dense fp16 linears up to 8 rows and all block-FP8 linears sum in a fixed order, so a row gives the same result in decode, the drafter and the DFly verify step. `0` goes back to cuBLAS / TurboMind |
| `--max-running-requests` | 1 | One agentic session; the decode and verify kernels are tuned for one request |
| `--mem-fraction-static` | 0.88 | Leaves a 286,720-token pool with DFly and vision on, above the 262,144 context |
| `--context-length` | 262,144 | The checkpoint allows 1M; 262,144 fits the pool with room for the output. The rotary table is built for this length only |
| prefill chunk | 2048 | The measured value |
| `--warmups sampling` | on | Builds FlashInfer's sampling module at startup, so the first sampled request does not wait for it |
| `--sleep-on-idle` | on | Without it the eight idle scheduler loops keep about 5 CPU cores busy |
| `--disable-custom-all-reduce` + `SGLANG_DSV41_HIER_AR*` | on | The 8-GPU mesh is two NVLink quads plus bridges. Custom all-reduce inside each quad, then across the bridge pair |

## Limitations

- One request at a time. A second request queues behind the first.
- A fresh subagent conversation pays its own cold prefill (about 4.5 s at 19k tokens).
- With many agents taking turns, the sliding-window pool (28,672 tokens) decides how many conversations stay cached: in a 14-agent round-robin test, 1 turn in 83 lost part of its cached prefix.
- Videos are capped at 32,768 tokens per video (`HY4_VIDEO_MAX_TOKENS`).
- Greedy output with DFly differs from greedy output without it (see [Accuracy](#accuracy)).
- Checked with the benchmarks above, GSM8K and RULER2. No multi-hour soak yet.

## PCIe V100s

The recipe is measured on SXM2. The two-level all-reduce (`SGLANG_DSV41_HIER_AR*`) assumes two NVLink quads joined by bridges; on PCIe cards set `SGLANG_DSV41_HIER_AR=0` before the script. Untested on an 8-card PCIe box; expect slower decode.
