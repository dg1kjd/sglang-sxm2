# Qwen3.8-Flash-Next on 4× V100

## Get the model

The validated checkpoint is the NVFP4 quantisation of Qwen3.8-Flash-Next. 126 GB. It is the multimodal export, so the vision tower comes with it. `language_model_only: false` in `config.json` is the check. Other checkpoints of the same architecture may work; they are untested. The model stays under its own license.

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4

export FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
```

| | |
|---|---|
| checkpoint | [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) |
| base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| quantisation | NVFP4 W4A16 (modelopt), FP16 KV cache at runtime |

The launcher comments record why each OOM-sensitive flag is what it is. Read them before changing `--mem-fraction-static`, `--max-prefill-tokens`, or `--hicache-size`. On V100 (`--dtype float16`) the script passes `--ple-offload-embedding`, so the 51 GB PLE n-gram table lands in host memory. Without that offload the table is created on GPU and OOMs at load.

## Talking to it

Both API surfaces are native.

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38next-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen38next-nvfp4","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Image input works on both APIs. This model emits reasoning: an empty `content` alongside a large `completion_tokens` means the reply hit `max_tokens` while still inside a thinking block. Raise the limit.

## Measured performance

2026-11-12, `scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp`: 4× V100-SXM2-32GB, TP=4, MTP 3 steps / 4 draft tokens, `--mem-fraction-static 0.86`, `--max-running-requests 3`, temperature 0. KV pool 331,968 tokens, context 262,144. The chat template leaves thinking on, and these tests do not turn it off. The prompts and the agent session are the same as on the [GLM-5.3-Flash page](GLM-5.3-Flash.md#measured-performance).

**Decode.** Six chat prompts, 512 tokens each (math stopped at 294), streamed from `/generate`; tok/s excludes TTFT. Accept length is tokens per target forward.

| prompt | tok/s | accept length |
|---|---:|---:|
| code | 166.6 | 2.81 |
| refactor | 162.5 | 2.75 |
| explain | 155.2 | 2.61 |
| math | 195.6 | 3.30 |
| agent step | 173.5 | 2.94 |
| German prose | 165.4 | 2.78 |
| **all six** | **167.2** | |

The engine runs about 59 target forwards per second at one stream, so decode speed follows the accept length. Several streams share those forwards: running the six prompts as 2 or 3 parallel streams gives 153.3 and 193.3 tok/s in total, 83.6 and 70.4 per stream: two streams together are no faster than one.

**Agent session.** Through `/v1/chat/completions` with a `bash` tool: source files as the first user message, then three turns that each append an assistant tool call and its result (~400–900 new tokens). TTFT includes rendering and tokenizing the whole chat on the server. Each conversation starts with a unique system prompt, so the first turn is a full prefill (the disk tier of the host cache survives `/flush_cache`). The first request with a new set of tools compiles its tool-call grammar once; the rows below were measured after that.

| context | cold first turn | tool turns 1 / 2 / 3 |
|---|---:|---:|
| 32,361 tokens | 7.34 s | 340 / 322 / 298 ms |
| 128,360 tokens | 30.98 s | 450 / 599 / 516 ms |
| 190,361 tokens | 45.58 s | 498 / 506 / 484 ms |

Cold prefill runs at 4,410 tok/s at 32k and 4,180 tok/s at 190k. At 128k the second tool turn finds only the first message cached, not the first turn's addition, so it prefills ~1,500 tokens instead of ~600.

**Main session plus subagents.** A 190k main conversation, then two fresh ~19k subagent conversations of three turns each, with a main turn after each subagent. The main turns stay cached:

| turn | TTFT |
|---|---:|
| main, cold / next turn | 46.19 s / 496 ms |
| subagent 1, cold / turns 2–3 | 4.55 s / 376, 430 ms |
| main | 477 ms |
| subagent 2, cold / turns 2–3 | 4.55 s / 356, 365 ms |
| main | 487 ms |

## Reference recipe

The wrapper is the supported entry. `mtp` is the recipe the numbers above were measured on.

```bash
export FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4
bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp
```

Expanded (what that script runs for `mtp`). The env block is not implied by the CLI flags. `SGLANG_NUMA_BIND_V2=0` keeps the host cache interleaved; the file tier must sit on a real disk, not a tmpfs `/tmp`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_SM70_QWEN_FUSIONS=1
export SGLANG_SM70_QSA_DENSE_PREFILL_MAX_TOKENS=8192
export SGLANG_NUMA_BIND_V2=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$HOME/hicache_storage
export SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE=1
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=37.5G
export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=64G

python -m sglang.launch_server \
  --model-path "${FLASH_NEXT_MODEL}" \
  --served-model-name qwen38next-nvfp4 \
  --trust-remote-code \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend tilelang_fa_v100 \
  --linear-attn-prefill-backend tilelang \
  --linear-attn-decode-backend triton \
  --kv-cache-dtype auto \
  --tensor-parallel-size 4 \
  --context-length 262144 \
  --mem-fraction-static 0.86 \
  --max-running-requests 3 \
  --max-mamba-cache-size 20 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-hierarchical-cache \
  --hicache-size 8 \
  --hicache-mem-layout page_first \
  --hicache-storage-backend file \
  --hicache-write-policy write_back \
  --sleep-on-idle \
  --cuda-graph-max-bs-decode 3 \
  --cuda-graph-bs-decode 1 2 3 \
  --disable-prefill-cuda-graph \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-full-memory-ratio 0.2 \
  --enable-cache-report \
  --enable-metrics \
  --ple-offload-embedding \
  --warmups sampling \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "${FLASH_NEXT_MODEL}" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| MTP | on, 3 steps / 4 draft tokens | The measured recipe. `target` is the same server with no draft head |
| `--kv-cache-dtype` | `auto` (FP16 on V100) | The fast sm70 decode kernel reads only the selected top-k, so FP16 costs no decode speed and accepts better than `fp8_e5m2`. The 1-byte cache is `FLASH_NEXT_EXTRA_ARGS='--kv-cache-dtype fp8_e5m2'` plus a lower mem-fraction |
| `--mem-fraction-static` | 0.86 | 0.88 OOMed prefill under a beyond-spec 32k / np 8 load. 0.86 costs about 12% of the KV pool and still holds long agent contexts at 3 streams |
| `--max-running-requests` | 3 | Aggregate decode peaks here. A fourth stream adds no throughput and stretches TTFT. CUDA graphs are captured for batch 1–3 only |
| prefill chunk | 4096 | An 8k GDN chunk OOMed in the headroom left after the KV pool. Long prompts still run, in 4k chunks |
| `--hicache-size` | 8 per rank | Host KV and host Mamba, so total is `N × 2 × 4` ranks = 64 GB. N=16 left the box near thrashing. `page_first` is required for the Mamba host pool |
| disk-tier metadata cache | on | Without it every disk-tier lookup lists the whole storage directory, so requests wait longer as it fills: a 600-token tool result waited 92–119 ms before prefill at 50–68k files, 20 ms with the cache |
| disk-tier page epoch | 2 | The file tier outlives the engine and takes any stored page of the right size as valid. The serve script removes this model's pages once when its page epoch changes. Epoch 2 (2026-11) is the QSA pending-ring fix: with MTP on, earlier trees could store wrong index keys for decoded tokens. If you launch by hand, empty the storage directory once after upgrading |
| disk-tier size | 37.5G per rank, 64G free | The file tier has no size limit of its own; a 245k-token session adds tens of GB. `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` caps each TP rank's pages (every rank keeps its own), so the directory holds up to 150 GB at TP 4; a rank that reaches it trims to 90%. `SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE` deletes the least recently used pages before a write would leave less than 64G free on the filesystem. With only the free-space limit, the tier filled a 2 TB disk down to it (277 GB) |
| write policy | `write_back` | `write_through` saturated the disk and made prefix readback collapse. Disk writes happen on eviction and shutdown |
| `--ple-offload-embedding` | on | The 51 GB n-gram table stays in host RAM. On GPU it OOMs at load |
| `--disable-prefill-cuda-graph` | on | Upstream turns breakable prefill graphs on for this model. Their capture does not fit next to the KV pool sized above (the draft's capture ran out of memory) |
| `--sleep-on-idle` | on | Without it each rank busy-spins a core while idle |
| `--warmups sampling` | on | Builds FlashInfer's sampling kernels at startup. Otherwise the first sampled request waits for the build (about 90 s when the cache is cold) |
| `--enable-cache-report` | on | Surfaces prefix-cache hits to Claude Code. The cache already hits without the flag; the flag only reports them |

`target` drops the four `--speculative-*` lines.

## PCIe V100s

Four PCIe-only V100s with P2P (no NVLink): export `SGLANG_CUSTOM_AR_ALLOW_PCIE=1` before the script. NCCL already uses PCIe P2P between cards behind one switch, so leave `NCCL_P2P_LEVEL` unset; `NVL` would send its traffic through host memory. Leave both unset on an NVLink mesh.

## Limitations

- Stability was hammered, not soaked: about an hour of agentic load at 1, 2 and 4 streams plus a beyond-spec 32k-token / 8-stream phase, no crash and no wrong output at `--mem-fraction-static 0.86`. No multi-day soak.
- Greedy output is not bit-identical across cache states. The radix cache replays an approximate linear-attention state for a cached prefix, so a prompt on a token decision boundary can come out differently cold and warm. Every output is a valid completion.
- The dense NVFP4 linear path is unverified. It matters only for checkpoints that quantise weights outside the MoE experts; this one does not.
