# SenseNova-Loop-9B on 4× V100 (experimental)

## Get the model

The checkpoint is the official BF16 release, 18 GB, served in fp16. The repository is gated: accept its terms on its Hugging Face page and log in first. The model stays under its own license (Apache 2.0).

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf download sensenova/SenseNova-Loop-9B --local-dir ~/models/SenseNova-Loop-9B

export SENSENOVA_MODEL=~/models/SenseNova-Loop-9B
```

| | |
|---|---|
| checkpoint | [`sensenova/SenseNova-Loop-9B`](https://huggingface.co/sensenova/SenseNova-Loop-9B) |
| architecture | 32 layers, three linear-attention (Gated DeltaNet) layers to one full-attention layer; layers 4–27 run twice for each token, with two residual streams (mHC) |
| precision | fp16 weights and KV cache at runtime, mHC routers in fp32 |
| model code | upstream SGLang PR [#47281](https://github.com/sgl-project/sglang/pull/47281), plus fused Volta kernels for the mHC router and merge |

The fused kernels matter: without them each mHC layer ran about 830 small kernels per token (now 5) and decode reached 7 tok/s.

## Talking to it

Both API surfaces are native.

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sensenova-loop-9b","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"sensenova-loop-9b","max_tokens":512,"messages":[{"role":"user","content":"Hello"}]}'
```

Thinking is on by default. Turn it off with `"chat_template_kwargs": {"enable_thinking": false}` (OpenAI) or `"thinking": {"type": "disabled"}` (Anthropic). The chat template has no effort levels. An empty `content` alongside a large `completion_tokens` means the reply hit `max_tokens` while still thinking. Raise the limit.

Tool calls stream as `tool_calls` (OpenAI) and `tool_use` blocks (Anthropic). Image input was tested through chat completions.

**Sampling.** A request that sets no `top_k` gets top_k 20 from the serve script (`SENSENOVA_TOP_K`; -1 turns the cut off). Temperature 1.0 and top_p 0.95 come from the model's `generation_config.json`. A request's own values win. Without a top-k cut the model can draw a very unlikely token and then repeat it until `max_tokens`; see [Known issues](Known-Issues.md).

## Measured performance

2026-11-18 and 11-19, `scripts/serve_sensenova_loop_v100.sh`: 4× V100-SXM2-32GB, TP=4, no MTP, `--mem-fraction-static 0.85`, `--max-running-requests 12` (6 on 11-18). KV pool 854,272 tokens, 488 linear-attention state slots, context 262,144. Ready about 50 s after launch (once the JIT kernels are built). Host RAM about 15 GB per copy.

**Decode.** One stream, a code prompt (merge two sorted lists), up to 512 tokens: 70.2 tok/s at temperature 0, 69.4 at temperature 1 (11-19).

Several streams, 2026-11-19, ~19k-token prompts each, 384 tokens, temperature 0, summed over the streams:

| streams | copies | decode |
|---:|---:|---:|
| 1 | 1 | 68 tok/s |
| 12 | 1 | 617 tok/s |
| 24 | 2 (`SENSENOVA_DP=2`, 8 GPUs) | 1051 tok/s |

Before the Volta decode kernel for this head layout (four query heads per KV head, head size 256), 6 to 12 streams shared about 43 tok/s and one stream ran 64.6 tok/s.

**Time to first token.**

| request | TTFT |
|---|---:|
| short new prompt | 126 ms |
| ~3.6k-token new prompt | 528 ms |
| tool turn: cached 3.6k prefix plus a short new message | 113 ms |
| tool turn with a ~600-token tool result | 170 ms |

Uncached prefill runs at 5.2–5.7k tok/s up to 42k tokens (16.5k in 2.9 s, 42.5k in 8.2 s). A 245,137-token prompt takes 131 s cold. Five follow-up questions on that cached prompt took 4.6–6.2 s each, 13 answer tokens included; 1.4 s each with the new decode kernel (11-19).

**Accuracy.**

- gsm8k: 96.5% on 200 questions (sgl-eval 0.1.3, thinking on, temperature 1.0, top_p 0.95, max_tokens 16384; measured before the top_k 20 default). One answer hit the token limit.
- HLE (Humanity's Last Exam): 15.0% on 80 random text-only problems (95% interval 8.8–24.4%), thinking on, temperature 1.0, top_p 0.95, top_k 20, max_tokens 65,536, no tools. Graded with the official judge prompt, by Qwen3.8 instead of o3-mini. 19 of the 80 hit the token limit and count as wrong. Calibration error 76.
- Needle in a 245k-token prompt: 5 of 5 values found, at 5, 25, 50, 75 and 95% depth.
- Retrieval with thinking off: 6 of 6 random values at 45.8k and 93.8k tokens, 5 of 6 at 63.8k. When the values follow a pattern (`i*7 mod 1000`), the model computes them instead of looking them up and fails at 42k.
- Logprobs against the Hugging Face reference in fp32: mean difference 0.003–0.005 nats per token on four prompts; greedy output equals the reference in fp16.

**Images.** A 2048×1536 image (3,114 prompt tokens) answers in 1.1 s, a 3840×2160 image (8,202 tokens) in 4.2 s. Six of six test images were read correctly.

## Reference recipe

The wrapper is the supported entry.

```bash
export SENSENOVA_MODEL=~/models/SenseNova-Loop-9B
bash scripts/serve_sensenova_loop_v100.sh
```

Expanded (what that script runs). The env block is not implied by the CLI flags.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_NUMA_BIND_V2=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0

python -m sglang.launch_server \
  --trust-remote-code \
  --model-path "${SENSENOVA_MODEL}" \
  --served-model-name sensenova-loop-9b \
  --dtype float16 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype auto \
  --tensor-parallel-size 4 \
  --mem-fraction-static 0.85 \
  --context-length 262144 \
  --data-parallel-size 1 \
  --load-balance-method total_tokens \
  --max-running-requests 12 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --cuda-graph-max-bs-decode 12 \
  --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 \
  --disable-prefill-cuda-graph \
  --mamba-radix-cache-strategy extra_buffer \
  --enable-cache-report \
  --preferred-sampling-params '{"top_k": 20}' \
  --sleep-on-idle \
  --warmups sampling \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| `--dtype` | `float16` | V100 has no BF16 tensor cores. Activations stay far inside the fp16 range (largest under 700 in the fp32 reference); the mHC routers compute in fp32 |
| `--tensor-parallel-size` | 4 | The model has four KV heads, one per rank |
| `--max-running-requests` | 12 | Per copy (`SENSENOVA_MAX_RUNNING`). Decode throughput still grows at 12 streams; the KV pool and the 488 state slots hold more. CUDA graphs are captured for batch 1–12 |
| `--data-parallel-size` | 1 | `SENSENOVA_DP`: copies of the model behind one port, four GPUs each; `auto` runs one per four GPUs found. Each copy keeps its own prefix cache, so more than one suits batch jobs, not a long agent session. The script takes the first 4×DP GPUs unless `SENSENOVA_GPUS` lists them; each group of four should share an NVLink domain |
| `--preferred-sampling-params` | top_k 20 | Cuts the tail the stray-token repeat starts from ([Known issues](Known-Issues.md)). Fills only fields a request leaves unset |
| `--sleep-on-idle` | on | Without it each rank busy-spins a core while idle |
| `--warmups sampling` | on | Builds FlashInfer's sampling kernels at startup. Otherwise the first sampled request waits for the build (about 90 s when the cache is cold) |
| `--enable-cache-report` | on | Surfaces prefix-cache hits to Claude Code. The cache already hits without the flag; the flag only reports them |

## Limitations

- Experimental. It passed short checks (gsm8k, a 245k needle, tool calls, images, thinking on and off) and has not carried a long session.
- The model can switch to an unrelated token and repeat it until `max_tokens`. The fp32 reference implementation does the same on the same text, so this is the model, not the engine. top_k 20 makes the first draw much less likely. Details in [Known issues](Known-Issues.md).
- Untested on PCIe V100s.
