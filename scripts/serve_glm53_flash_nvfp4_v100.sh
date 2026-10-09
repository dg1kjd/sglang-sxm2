#!/usr/bin/env bash
# Serve RadixArk/GLM-5.3-Flash-NVFP4 (W4A16 Marlin, FP16 activations; FP16 or
# FP8 E4M3 KV) on 8x V100-SXM2-32GB, one request at a time, MTP on by default.
# Experts are TP-sliced (GLM53_EP_SIZE=1): every rank does the same MoE work
# per token. With EP8 the other ranks waited on the one holding most routes.
#
# Usage (from the repo root):
#   GLM53_MODEL=/path/to/GLM-5.3-Flash-NVFP4 bash scripts/serve_glm53_flash_nvfp4_v100.sh
#
# Env overrides:
#   GLM53_MODEL=/path/to/GLM-5.3-Flash-NVFP4   (required)
#   SGLANG_V100_VENV=/path/to/venv             (default: $HOME/sglang-v100-venv)
#   SGLANG_V100_HOST=0.0.0.0  SGLANG_V100_PORT=11435   (shared with the other V100 scripts)
#   GLM53_GPUS=0,1,2,3,4,5,6,7
#   GLM53_MTP_STEPS=3          (0 turns MTP off)
#   GLM53_CONTEXT_LENGTH / GLM53_MEM_FRACTION   (defaults follow GLM53_MTP_STEPS)
#   GLM53_KV_CACHE_DTYPE=fp8_e4m3   (default auto = FP16; fp8 roughly doubles the
#                                    pool, so the context default scales with it)
#   GLM53_MAX_RUNNING=1         (increase with Mamba slots for real batching)
#   GLM53_MAMBA_SLOTS=12        (4-request batch needs at least 20 slots)
#   GLM53_CUDA_GRAPH_BS_DECODE=1 (or use GLM53_EXTRA_ARGS for a list, e.g. 1 2 4)
#   GLM53_EP_SIZE=1
#   GLM53_CHUNKED_PREFILL / GLM53_MAX_PREFILL_TOKENS   (default 4096)
#   GLM53_EXTRA_ARGS="..."     (appended to the launch_server arguments)
#
# Prefill chunking: 4096 measured 5.9-8.1% faster end to end than 2048 on long
# prompts (8x V100-SXM2, 2026-10-05); 8192 OOMs at mem-fraction 0.945.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$VENV}}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
MODEL="${GLM53_MODEL:-}"
[[ -n "$MODEL" ]] || { echo "set GLM53_MODEL to the GLM-5.3-Flash-NVFP4 checkout" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GLM53_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
# NCCL_P2P_LEVEL stays unset: NCCL picks NVLink where it exists and PCIe P2P
# behind a PCIe switch; forcing NVL routes PCIe-only cards through host memory.
export NCCL_ALGO="${NCCL_ALGO:-allreduce:tree}"
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
# Batch-1 decode on Volta HMMA for the NVFP4 dense, shared and routed experts.
export SGLANG_SM70_GLM_NVFP4_GEMV="${SGLANG_SM70_GLM_NVFP4_GEMV:-1}"
export SGLANG_SM70_GLM_NVFP4_MOE_DECODE="${SGLANG_SM70_GLM_NVFP4_MOE_DECODE:-1}"
# TP8 custom all-reduce is off (the mesh is two NVLink quads plus bridges);
# reduce inside each quad with custom AR, then across the bridge pair.
export SGLANG_DSV41_HIER_AR="${SGLANG_DSV41_HIER_AR:-1}"
export SGLANG_DSV41_HIER_AR_CA="${SGLANG_DSV41_HIER_AR_CA:-1}"
# Both steps in one push kernel, bitwise equal to the two custom-AR launches.
export SGLANG_DSV41_HIER_AR_PUSH="${SGLANG_DSV41_HIER_AR_PUSH:-1}"
# Batch-1 FP16 attention projections on the native GEMV instead of cuBLAS.
export SGLANG_SM70_DENSE_GEMV="${SGLANG_SM70_DENSE_GEMV:-1}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

# --kv-cache-dtype fp8_e4m3 stores the DSA latent as unscaled E4M3 bytes; the SM70
# sparse MLA kernel widens them back to fp16 while staging rows into shared memory.
# Per token over the 11 DSA layers that is 12,716 -> 7,084 B, so the KV pool roughly
# doubles. The Hopper flashmla_kv backend does not exist on sm_70, so this path has
# to be asked for explicitly; without the env var fp8_e4m3 is rejected at argument
# resolution rather than picking a backend that cannot load.
GLM53_KV_CACHE_DTYPE="${GLM53_KV_CACHE_DTYPE:-auto}"
GLM53_MAX_RUNNING="${GLM53_MAX_RUNNING:-1}"
GLM53_MAMBA_SLOTS="${GLM53_MAMBA_SLOTS:-12}"
GLM53_CUDA_GRAPH_BS_DECODE="${GLM53_CUDA_GRAPH_BS_DECODE:-1}"
read -r -a CUDA_GRAPH_BS_DECODE <<< "$GLM53_CUDA_GRAPH_BS_DECODE"
if [[ "$GLM53_KV_CACHE_DTYPE" == "fp8_e4m3" ]]; then
  export SGLANG_SM70_DSA_FP8_KV="${SGLANG_SM70_DSA_FP8_KV:-1}"
fi

# MTP: GLM53_MTP_STEPS=N (default 3) drafts N tokens per step with the checkpoint's
# layer 45; 0 turns it off. The draft's BF16 routed experts are quantized to NVFP4
# at load (2.4 -> 1.06 GB per rank); verify still decides every token.
# Context defaults stay below the KV pool each setting leaves (MTP 242880 at 0.935,
# no MTP 225600 at 0.88), so prompt + completion always fits.
# A running request pins 5 KDA state slots in the no-MTP path. The scheduler caps
# max_running_requests at max_mamba_cache_size // 5, so a real batch of four needs
# --max-mamba-cache-size 20. MTP uses a different, larger per-request footprint.
GLM53_MTP_STEPS="${GLM53_MTP_STEPS:-3}"
SPEC_ARGS=()
MEM_FRACTION_DEFAULT=0.88
CONTEXT_LENGTH_DEFAULT=223232
if [[ "$GLM53_MTP_STEPS" -gt 0 ]]; then
  export SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE="${SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE:-1}"
  MEM_FRACTION_DEFAULT=0.935
  CONTEXT_LENGTH_DEFAULT=240640
  if [[ "$GLM53_KV_CACHE_DTYPE" == "fp8_e4m3" ]]; then
    # Measured on 8x V100-SXM2 with GLM-5.3-Flash-NVFP4: a 521,024-token pool at
    # 0.945, so this context plus a completion always fits. FP16 tops out at
    # 290,240 tokens, which is why 240640 is the default above.
    MEM_FRACTION_DEFAULT=0.945
    CONTEXT_LENGTH_DEFAULT=480000
  fi
  SPEC_ARGS=(--speculative-algorithm EAGLE --speculative-draft-model-path "$MODEL"
    --speculative-num-steps "$GLM53_MTP_STEPS" --speculative-eagle-topk 1
    --speculative-num-draft-tokens "$((GLM53_MTP_STEPS + 1))")
fi

# --sleep-on-idle: without it the eight idle scheduler loops keep ~5.3 cores
# busy (measured 2026-10-05).
exec "$VENV/bin/python" -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$MODEL" \
  --served-model-name glm53-flash-nvfp4 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --language-only \
  --tensor-parallel-size 8 \
  --ep-size "${GLM53_EP_SIZE:-1}" \
  --attention-backend dsa \
  --linear-attn-backend triton \
  --kv-cache-dtype "$GLM53_KV_CACHE_DTYPE" \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE[@]}" \
  --max-running-requests "$GLM53_MAX_RUNNING" \
  --max-mamba-cache-size "$GLM53_MAMBA_SLOTS" \
  --mamba-max-states-per-path 2 \
  --mamba-full-memory-ratio 0.15 \
  --mamba-radix-cache-strategy extra_buffer \
  --chunked-prefill-size "${GLM53_CHUNKED_PREFILL:-4096}" \
  --max-prefill-tokens "${GLM53_MAX_PREFILL_TOKENS:-4096}" \
  --warmups prefix_reuse,sampling \
  --sleep-on-idle \
  --context-length "${GLM53_CONTEXT_LENGTH:-$CONTEXT_LENGTH_DEFAULT}" \
  --mem-fraction-static "${GLM53_MEM_FRACTION:-$MEM_FRACTION_DEFAULT}" \
  --host "${SGLANG_V100_HOST:-0.0.0.0}" \
  --port "${SGLANG_V100_PORT:-11435}" \
  "${SPEC_ARGS[@]}" ${GLM53_EXTRA_ARGS:-}
