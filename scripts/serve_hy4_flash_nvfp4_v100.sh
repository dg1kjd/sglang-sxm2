#!/usr/bin/env bash
# Serve Tencent Hy4-flash from an NVFP4 checkpoint (W4A16 experts, block-FP8
# attention projections, FP8 Engram tables) on 8x V100-SXM2-32GB, FP16
# activations and KV, one request at a time.
#
# Usage (from the repo root):
#   HY4_MODEL=/path/to/Hy4-flash-NVFP4 bash scripts/serve_hy4_flash_nvfp4_v100.sh
#
# Env overrides:
#   HY4_MODEL=/path/to/Hy4-flash-NVFP4         (required)
#   SGLANG_V100_VENV=/path/to/venv             (default: $HOME/sglang-v100-venv)
#   SGLANG_V100_HOST=0.0.0.0  SGLANG_V100_PORT=11435   (shared with the other V100 scripts)
#   HY4_GPUS=0,1,2,3,4,5,6,7
#   HY4_DFLY=1                 (0 turns the DFly drafter off)
#   HY4_VISION=1               (0 serves text only)
#   HY4_VIDEO_MAX_TOKENS=32768 (per video)
#   HY4_CONTEXT_LENGTH / HY4_MEM_FRACTION
#   HY4_EP_SIZE=1
#   HY4_EXTRA_ARGS="..."       (appended to the launch_server arguments)
#   SGLANG_USE_SM70_FIXED_ORDER_GEMM=1   (0: cuBLAS / TurboMind for the dense
#                              fp16 and block-FP8 linears and lm_head at every
#                              M; their rows then change with the batch size)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$VENV}}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
MODEL="${HY4_MODEL:-}"
[[ -n "$MODEL" ]] || { echo "set HY4_MODEL to the Hy4-flash NVFP4 checkout" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${HY4_GPUS:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-14"
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_ALGO="${NCCL_ALGO:-allreduce:tree}"
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
# TP8 custom all-reduce is off (the mesh is two NVLink quads plus bridges);
# reduce inside each quad with custom AR, then across the bridge pair.
export SGLANG_DSV41_HIER_AR="${SGLANG_DSV41_HIER_AR:-1}"
export SGLANG_DSV41_HIER_AR_CA="${SGLANG_DSV41_HIER_AR_CA:-1}"
export SGLANG_DSV41_HIER_AR_PUSH="${SGLANG_DSV41_HIER_AR_PUSH:-1}"
# Batch-invariant dense GEMM up to 8 rows: a row of every dense linear and of
# the logits is the same in the verify step (M = 6), the drafter and decode
# (M = 1). Prefill (more rows) keeps cuBLAS for the fp16 linears.
export SGLANG_USE_SM70_FIXED_ORDER_GEMM="${SGLANG_USE_SM70_FIXED_ORDER_GEMM:-1}"
export SGLANG_GRAMMAR_DISK_CACHE_DIR="${SGLANG_GRAMMAR_DISK_CACHE_DIR-$HOME/.cache/sglang/grammar}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

HY4_DFLY="${HY4_DFLY:-1}"
SPEC_ARGS=()
if [[ "$HY4_DFLY" != 0 ]]; then
  SPEC_ARGS=(--speculative-algorithm DFLY --speculative-dfly-block-size 6)
fi

# Vision: the tower's weights (~250 MB per rank, ~18k tokens of KV pool) stay
# in pinned host memory and are copied in for each pass of at most one
# full-size image (hunyuan_v4_flash.py, finalize_after_quant_processing).
HY4_VISION="${HY4_VISION:-1}"
VISION_ARGS=()
# The embedding cache keeps an image that spans several prefill chunks from
# being encoded again for each chunk; in host memory it costs no KV pool.
export SGLANG_USE_HOST_VLM_CACHE="${SGLANG_USE_HOST_VLM_CACHE:-1}"
if [[ "$HY4_VISION" == 0 ]]; then
  VISION_ARGS=(--language-model-only)
else
  # Image preprocessing on the GPU would put a CUDA context on the first card.
  export SGLANG_FORCE_CPU_IMAGE_PREPROCESSING="${SGLANG_FORCE_CPU_IMAGE_PREPROCESSING:-1}"
  # 512 MB per rank holds about twenty full-size images or two capped videos.
  export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-512}"
  # The checkpoint lets one video take 60000 tokens. Every rank holds a
  # video's embedding on the GPU while it is prefilled, so cap it. The video
  # processor counts its budget in pixels, 32 x 32 per token.
  VISION_ARGS=(--mm-process-config
    "{\"video\": {\"max_pixels\": $(( ${HY4_VIDEO_MAX_TOKENS:-32768} * 1024 ))}}")
fi

# --sleep-on-idle: without it the eight idle scheduler loops keep ~5 cores busy.
exec "$VENV/bin/python" -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$MODEL" \
  --served-model-name hy4-flash-nvfp4 \
  --reasoning-parser hunyuan \
  --tool-call-parser hunyuan \
  --dtype float16 \
  --quantization modelopt_mixed \
  --fp4-gemm-backend marlin \
  --tensor-parallel-size 8 \
  --ep-size "${HY4_EP_SIZE:-1}" \
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
  --context-length "${HY4_CONTEXT_LENGTH:-262144}" \
  --mem-fraction-static "${HY4_MEM_FRACTION:-0.88}" \
  --host "${SGLANG_V100_HOST:-0.0.0.0}" \
  --port "${SGLANG_V100_PORT:-11435}" \
  "${VISION_ARGS[@]}" "${SPEC_ARGS[@]}" ${HY4_EXTRA_ARGS:-}
