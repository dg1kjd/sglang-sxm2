<div align="center" id="top">

# sglang-sxm2

*"You got any more of them chinese models?"-RELEASE*

**An agentic-coding inference engine for N× V100-SXM2.**

Current open-weight models on Volta GPUs, with long context, tool calls, and the native Anthropic Messages API, so Claude Code connects directly.
SXM2 (NVLink) is only needed for speed. Any set of 32 GB V100s works, PCIe cards included.

</div>

---

## Models

| Model | GPUs | Context | Use it for |
|---|---|---|---|
| **[GLM-5.3-Flash](docs/v100/GLM-5.3-Flash.md)**, 320B MoE, NVFP4, MTP | 8× V100 32 GB | 240k | **Agentic coding.** The primary model, with image and video input |
| **[Qwen3.8-Flash-Next](docs/v100/Qwen3.8-Flash-Next.md)**, 125B MoE, NVFP4, MTP | 4× V100 32 GB | 262k | Coding and chat on four cards, image input, up to 3 parallel requests |
| **[DeepSeek-V4.1-Flash](docs/v100/DeepSeek-V4.1-Flash.md)**, official MXFP8 + MXFP4, DSpark | 8× V100 32 GB, ~300 GiB host RAM | 256k | **Best reasoning and world knowledge** of the four. The full official checkpoint, for agentic coding where answer quality matters more than speed |
| **[MiniMax-H3](docs/v100/MiniMax-H3.md)**, video + audio | 4× V100 32 GB | 4–15 s clips | Text-to-video, keyframes, reference image/video/audio |

Each model page has the full measurements, the exact launch recipe, and why each flag has its value. Other models may load, but they are untested here.

**Which one fits?**
- **8 cards:** GLM-5.3-Flash for fast turns, with images and video. DeepSeek-V4.1-Flash for the hardest problems, if the host has the RAM.
- **4 cards:** Qwen3.8-Flash-Next for text and images, MiniMax-H3 for video.
- **16 GB cards:** none of the recipes fit. Qwen's NVFP4 weights alone take ~22 GB per GPU on four cards.

## Performance

Measured on 8× V100-SXM2-32GB (two NVLink quads joined by bridges), one request at a time unless noted, temperature 0.

### GLM-5.3-Flash: fast tool-call turns

In an agent loop most turns append a tool result of a few hundred tokens to a long conversation. Those turns only prefill the new tokens, so the answer starts in well under a second up to 128k context and in about a second at 190k.

| Conversation length | First turn (cold) | Each following tool-call turn |
|---|---:|---:|
| 32k tokens | 16 s | 0.52–0.60 s |
| 128k tokens | 67 s | 0.62–0.71 s |
| 190k tokens | 103 s | 1.03–1.15 s |

- **Decode:** ~177 tok/s (154–199 across six prompt types), with MTP accepting ~2.9 tokens per step.
- **Images and video cost no context.** The vision tower runs from host memory. A 2048×1536 image takes ~3 s, a 3840×2160 one ~6 s, a 60 s 720p video ~23 s.
- **Subagents don't evict the main session.** Two conversations stay cached side by side. With a 190k main session, a fresh 17k subagent costs ~8.5 s once, then ~0.5 s per turn, and the next main turn answers in ~1 s without prefilling the main session again.

### Qwen3.8-Flash-Next: fast tool-call turns on four cards

The same agent-loop test as GLM, on half the GPUs.

| Conversation length | First turn (cold) | Each following tool-call turn |
|---|---:|---:|
| 32k tokens | 7 s | 0.30–0.34 s |
| 128k tokens | 31 s | 0.45–0.60 s |
| 190k tokens | 46 s | 0.48–0.51 s |

- **Decode:** ~167 tok/s for one stream (155–196 across the same six prompts as GLM), with MTP accepting ~2.9 tokens per step. Up to three streams run at once, but two together are no faster than one (153 tok/s total); three reach 193.
- **Subagents don't evict the main session.** With a 190k main session, a fresh 19k subagent costs ~4.5 s once, then 0.36–0.43 s per turn, and the next main turn still answers in ~0.5 s.

### DeepSeek-V4.1-Flash: the most capable model, running locally

DeepSeek-V4.1-Flash is the strongest model here for reasoning and world knowledge. It runs from the official checkpoint, with Engram tables in host RAM, experts spilled to the host, and sparse attention, on hardware it was never meant for. As far as we know, no other engine runs it on V100s, let alone for local agentic coding. It has carried a multi-hour Claude Code session.

It is slow: about 11 tok/s on short code, about 7 on prose, and ~560 tok/s warm prefill. The resident conversation is never re-prefilled. A turn that adds a few dozen tokens reaches its first token in about 1 s, one that adds a 700-token tool result in about 3 s.

### MiniMax-H3

On 4 GPUs, a 5 s 960×544 clip takes 50 steps of 10–12 s each. A 5 s 1344×768 clip takes about 25 minutes.

### What makes the turns fast

- **Prefix cache:** a turn that extends a cached conversation prefills only the new tokens.
- **Volta kernels for MTP verify:** they check 2–4 draft tokens per step and give each row the same result as batch 1.
- **Tokenizer cache:** the server caches tokenized prompt pieces, so a long chat is not re-tokenized from scratch every turn.
- **Startup warmup:** the kernels for a cache hit and for a grammar-constrained tool-call decode load at startup, not on the first real tool turn.

## Requirements

- **GPUs:** V100 with **32 GB**. SXM2 with NVLink is the measured setup. PCIe V100s run the same scripts, slower; GLM and Qwen need the interconnect switches under "PCIe V100s" on their pages.
- **Host RAM:** GLM ~23 GB in use, up to ~6 GB more with images (vision tower and image cache, pinned). Qwen ~134 GB in use with the host KV tier, so plan for 160 GB. DeepSeek needs ~190 GiB on 1 GiB hugepages for Engram plus ~104 GiB of pinned expert spill. H3 keeps the text encoder and VAEs on the host.
- **Disk:** GLM 190 GB, Qwen 126 GB, DeepSeek 476 GB, plus the H3 checkpoint.
- **CUDA 12.8 or 12.9.** CUDA 13 dropped Volta.
- **GCC ≤ 14** with a working `cc1plus`. CUDA 12.9 rejects GCC 15, which many distros now default to.
- **Python 3.12**, Linux. `ffmpeg` and `ffprobe` for MiniMax-H3.

## Install

```bash
git clone https://github.com/dg1kjd/sglang-sxm2.git
cd sglang-sxm2

# System deps, Python env, patched FlashInfer, TurboMind, sglang-kernel, Marlin.
# About an hour, most of it nvcc.
bash scripts/install_v100.sh
conda activate sglang-v100

# Check that the Volta kernels registered.
bash scripts/smoke_v100.sh
```

Run the smoke check. A server that is missing the Volta kernels still starts and answers, but its MoE layers return zeros. [docs/v100/INSTALL.md](docs/v100/INSTALL.md) walks through each step and what to do when one fails, including running the server as a systemd service.

### Docker

The image builds the same stack from source. No pre-built image is published. You need Docker with the NVIDIA Container Toolkit and a host driver from the R570 series or newer (the image uses CUDA 12.8).

```bash
docker compose -f docker/v100-compose.yaml build glm53  # about an hour; one image for all four
docker compose -f docker/v100-compose.yaml up glm53     # or qwen38, dsv41, h3
```

Checkpoints are read from `MODELS_DIR` (default `~/models`) under the directory names used in [Run](#run). JIT kernels and caches persist in a Docker volume, so only the first launch compiles (about 10 minutes for GLM). Overrides such as `SGLANG_V100_PORT` or `GLM53_GPUS` pass through from the shell. DeepSeek-V4.1-Flash still needs the 1 GiB hugepages reserved on the host. [INSTALL.md](docs/v100/INSTALL.md#docker) describes how the image and the compose file are put together and how to run each model.

## Run

One language model at a time. All three bind `0.0.0.0:11435` (`SGLANG_V100_HOST`, `SGLANG_V100_PORT`). The first launch compiles JIT kernels for several minutes.

```bash
pip install -U "huggingface_hub[cli]"

# GLM-5.3-Flash, 8 GPUs
hf download RadixArk/GLM-5.3-Flash-NVFP4 --local-dir ~/models/GLM-5.3-Flash-NVFP4
GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4 bash scripts/serve_glm53_flash_nvfp4_v100.sh

# Qwen3.8-Flash-Next, 4 GPUs, MTP on
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4
FLASH_NEXT_MODEL=~/models/Qwen3.8-Flash-Next-NVFP4 bash scripts/serve_qwen38_flash_next_nvfp4_v100.sh mtp

# DeepSeek-V4.1-Flash, 8 GPUs (reserve 1 GiB hugepages and turn MGLRU off first, see its page)
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir ~/models/DeepSeek-V4.1-Flash
MODEL_PATH=~/models/DeepSeek-V4.1-Flash bash scripts/serve_dsv41_v100.sh

# MiniMax-H3 video, 4 GPUs, port 30010
hf download MiniMaxAI/MiniMax-H3 --local-dir ~/models/MiniMax-H3
H3_MODEL=~/models/MiniMax-H3 bash scripts/serve_minimax_h3_v100.sh
```

The server is ready when the log says `The server is fired up and ready to roll!`.

## Connect

OpenAI-compatible and Anthropic Messages endpoints are both native.

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:11435
export ANTHROPIC_AUTH_TOKEN=placeholder
export API_TIMEOUT_MS=3000000
export CLAUDE_CODE_ATTRIBUTION_HEADER=0   # keeps the prompt prefix stable, so turns hit the cache
claude
```

These models reason before they answer. An empty `content` with a large `completion_tokens` means the reply hit `max_tokens` while still thinking, so raise the limit. [Reasoning effort](docs/v100/Reasoning-Effort.md) lists how each model reads Claude Code's effort setting.

## Limitations

- **One agent session per server** for GLM and DeepSeek (`--max-running-requests 1`). Further requests queue. Qwen takes up to three streams.
- **DeepSeek-V4.1-Flash keeps one conversation resident.** A new conversation prefills from scratch. The last four are kept on disk, so returning to one of them loads it back instead of prefilling it again. A long uncached suffix runs at about 300 tok/s.
- **GLM tool turns at 190k got slower** (1.03–1.15 s, 0.60–0.77 s on 2026-10-02) somewhere between 10-02 and 11-11; 32k and 128k are unchanged. Not found yet.
- **Not soaked for days.** Qwen ran an hour of mixed load tests and DeepSeek a multi-hour Claude Code session. GLM is checked with benchmarks and GSM8K, with no multi-hour soak yet.
- **Volta has no bf16 and no FP4/FP8 hardware.** Everything runs as fp16 with weight-only 4/8-bit kernels, so numerics differ slightly from a Hopper run of the same checkpoint.
- **Claude Code can show "No response from the API after …"** while a long fresh prompt is still prefilling (seen on DeepSeek-V4.1-Flash with Claude Code's system prompt). With `API_TIMEOUT_MS` set as above it keeps waiting. Without it, Claude Code retries the request, and long prefills can make it unusable.

## Relationship to SGLang

This is a fork of [SGLang](https://github.com/sgl-project/sglang) that is synced with upstream `main` regularly. Upstream's engine (radix and hierarchical caches, speculative decoding, the OpenAI and Anthropic servers, parsers) is used as-is wherever possible. This fork adds the Volta kernels, the model support listed above, and serving recipes tuned for one agentic session.

Report bugs in the Volta path here. Report bugs in SGLang itself upstream.

## Credits

Built on [SGLang](https://github.com/sgl-project/sglang) (Apache 2.0, SGLang Team), which does the hard part.

Incorporates work from:

- **[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100)**: the original Volta port of SGLang, including the TileLang attention backend, the QSA and GDN kernels, the NVFP4 path for hardware without FP4, the TurboMind sm70 integration and the PLE host offload. Also [haohervchb/flashinfer](https://github.com/haohervchb/flashinfer) (sm70 FlashInfer) and [GooseLLM](https://github.com/haohervchb/GooseLLM) (TileLang FlashAttention for V100). Their November follow-ups make `can_p2p` check every peer and validate the PCIe byte cap, and their V100 backport of the QSA pending-ring fix pointed at the parts an older base lacks. Ported from their Qwen3.8 work after that: QSA chunk-prefill reads an FP16 KV pool directly, the fused shared-expert gate runs 2 and 4 rows, and a test for a prefill shorter than one compression group. Their findings that the Triton vision attention is the slow part of image requests on Volta, and that image preprocessing can be kept off the GPU, led to the TileLang vision backend and the `SGLANG_FORCE_CPU_IMAGE_PREPROCESSING` fix here. Ported from their later Qwen3.8 decode and prefill work: NVFP4 MoE decode in two kernels instead of four, the fused QSA indexer prep with the fp16 RoPE cache, and long QSA prefill rows scored on one TP rank each.
- **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)**: the vLLM fork for V100 whose TurboMind sm70 block-FP8 and FP16 MoE backend the build compiles.
- **[Horacio Vico](https://github.com/hvico)** ([hvico/sglang-V100](https://github.com/hvico/sglang-V100)): the opt-in custom all-reduce over PCIe P2P for GPUs without NVLink (`SGLANG_CUSTOM_AR_ALLOW_PCIE`).
- **[ltarcher](https://github.com/ltarcher)** ([ltarcher/sglang-v100-qwen3.8-flash-next](https://github.com/ltarcher/sglang-v100-qwen3.8-flash-next)): the import guards that treat an installed but unloadable DeepGEMM wheel as missing, and the measurement showing that `NCCL_P2P_LEVEL=NVL` slows PCIe-only cards, after which the scripts leave that choice to NCCL.
- **[Divy Vasal](https://github.com/divyvasal)** ([sgl-project/sglang#42812](https://github.com/sgl-project/sglang/pull/42812)): DSpark's accept step returns copies instead of views of buffers that the next verify step overwrites, so tensor-parallel ranks cannot commit different token counts under the overlap scheduler.
- **[Andy Liu](https://github.com/andyl98)** ([sgl-project/sglang#46125](https://github.com/sgl-project/sglang/pull/46125)): sizes the QSA pending-key ring for speculative verify and draft extend, which overwrote group members before they were compressed (Qwen3.8-Flash-Next with MTP).
- **[aaahzb](https://github.com/aaahzb)** ([aaahzb/sglang-sxm2](https://github.com/aaahzb/sglang-sxm2)): install fixes from a fresh CUDA 12.8 setup (libssl-dev, CMake 3.26, the TurboMind source subset, the Marlin bf16 patch on CUDA 12.8+).
- **[That80-20](https://github.com/That80-20)** ([issue #5](https://github.com/dg1kjd/sglang-sxm2/issues/5)): found that a fresh install missed a TurboMind header (`csrc/sm70_tile_runtime_signal.cuh`) and that the installer stopped on a host without Rust, with the fix for both.
- **[frankier](https://github.com/frankier)** ([frankier/sglang-sxm2](https://github.com/frankier/sglang-sxm2)): their GLM-5.3 vision work suggested checking at launch that the image processor imports, and running the vision tower one image at a time.

Volta kernels by: [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy), [marlin_v100](https://github.com/zhinianqin/marlin_v100), [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100), [v100-skinny](https://github.com/dnv2003/v100-skinny) (QPN8 FP8 decode GEMV), [CUTLASS](https://github.com/NVIDIA/cutlass), [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [TileLang](https://github.com/tile-ai/tilelang).

None of these are redistributed here. The installer fetches them at pinned revisions. See [NOTICE](NOTICE) for licenses.

## License

Apache 2.0, inherited from SGLang. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is an independent project. It is not affiliated with, endorsed by, or supported by the SGLang project, LMSYS, NVIDIA, or the model authors. It is provided as is, without warranty of any kind (Apache 2.0, Section 7). It drives hardware its vendor no longer supports, so validate it on your own setup before you rely on it.

No model weights are distributed here. Each checkpoint stays under its own license.

## Contact

Issues and pull requests are the preferred channel. For anything that does not belong in public: `git@jens-david-consulting.com`.
