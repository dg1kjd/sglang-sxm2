#!/usr/bin/env bash
# Serve sensenova/SenseNova-Loop-9B (BF16 checkpoint, served in fp16) on four
# V100s with tensor parallelism 4. No MTP.
#
# Usage (from the repo root):
#   SENSENOVA_MODEL=/path/to/SenseNova-Loop-9B bash scripts/serve_sensenova_loop_v100.sh
#
# Env overrides:
#   SENSENOVA_MODEL=/path/to/SenseNova-Loop-9B   (required)
#   SGLANG_V100_VENV=/path/to/venv               (default: $HOME/sglang-v100-venv)
#   SENSENOVA_DP=1                               (copies of the model behind one port, 4 GPUs each.
#                                                 auto: one copy per 4 GPUs found. Each copy keeps its
#                                                 own prefix cache, so more than 1 suits batch jobs,
#                                                 not one long agent session)
#   SENSENOVA_GPUS=0,1,2,3                       (default: the first 4*DP GPUs; consecutive groups of
#                                                 four should share an NVLink domain)
#   SENSENOVA_MAX_RUNNING=12                     (running requests per copy)
#   SGLANG_V100_HOST=0.0.0.0  SGLANG_V100_PORT=11435
#   SENSENOVA_MEM_FRACTION=0.85
#   SENSENOVA_TOP_K=20                           (top_k for requests that set none; -1: no cut)
#   SENSENOVA_EXTRA_ARGS='...'                   (appended last; last wins)
set -euo pipefail

VENV="${SGLANG_V100_VENV:-$HOME/sglang-v100-venv}"
[[ -x "$VENV/bin/python" ]] || VENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-$VENV}}"
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV; set SGLANG_V100_VENV" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${SENSENOVA_MODEL:-}"
[[ -n "$MODEL" ]] || { echo "set SENSENOVA_MODEL to the SenseNova-Loop-9B checkout" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "no config.json at $MODEL" >&2; exit 1; }

DP="${SENSENOVA_DP:-1}"
if [[ -n "${SENSENOVA_GPUS:-}" ]]; then
  IFS=, read -ra gpus <<< "$SENSENOVA_GPUS"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
[[ "$DP" == auto ]] && DP=$(( ${#gpus[@]} / 4 ))
[[ "$DP" =~ ^[1-9][0-9]*$ ]] || { echo "SENSENOVA_DP must be a positive number or auto; ${#gpus[@]} GPUs found (4 per copy)" >&2; exit 1; }
if [[ -z "${SENSENOVA_GPUS:-}" ]]; then
  gpus=("${gpus[@]:0:$((4 * DP))}")
fi
(( ${#gpus[@]} == 4 * DP )) || { echo "SENSENOVA_DP=$DP needs $((4 * DP)) GPUs, have ${#gpus[@]} (${gpus[*]})" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]}")"
echo "SenseNova: $DP copies x TP 4 on GPUs $CUDA_VISIBLE_DEVICES"
MAX_RUNNING="${SENSENOVA_MAX_RUNNING:-12}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The SM70 JIT kernels shell out to the venv's `ninja`.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
# CUDA 12.x accepts host GCC up to 14; pick the newest complete one. The
# runtime JIT runs a bare nvcc, which reads NVCC_PREPEND_FLAGS, not CUDAHOSTCXX.
# All serve scripts share this block: FlashInfer's JIT cache is keyed by version
# and arch only, so a different compiler command rebuilds it on every switch.
for _v in 14 13 12; do
  if [[ -x "/usr/bin/g++-$_v" ]] && ls /usr/libexec/gcc/*/$_v/cc1plus >/dev/null 2>&1; then
    export CC="/usr/bin/gcc-$_v" CXX="/usr/bin/g++-$_v" CUDAHOSTCXX="/usr/bin/g++-$_v"
    export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-$_v"
    break
  fi
done
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="$REPO/python"

args=(
  --trust-remote-code
  --model-path "$MODEL"
  --served-model-name sensenova-loop-9b
  # V100 has no BF16 tensor cores. Activations stay far inside the fp16 range
  # (largest |activation| ~520 in the HF reference); the mHC routers run in fp32.
  --dtype float16
  --reasoning-parser auto
  --tool-call-parser auto
  --attention-backend tilelang_fa_v100
  --linear-attn-prefill-backend tilelang
  --linear-attn-decode-backend triton
  --kv-cache-dtype auto
  # Four KV heads: TP 4 gives each rank one.
  --tensor-parallel-size 4
  --data-parallel-size "$DP"
  --load-balance-method total_tokens
  --host "${SGLANG_V100_HOST:-0.0.0.0}"
  --port "${SGLANG_V100_PORT:-11435}"
  --mem-fraction-static "${SENSENOVA_MEM_FRACTION:-0.85}"
  --context-length 262144
  --max-running-requests "$MAX_RUNNING"
  --chunked-prefill-size 4096
  --max-prefill-tokens 4096
  --cuda-graph-max-bs-decode "$MAX_RUNNING"
  --cuda-graph-bs-decode $(seq 1 "$MAX_RUNNING")
  --disable-prefill-cuda-graph
  --mamba-radix-cache-strategy extra_buffer
  --enable-cache-report
  # Without a top-k cut the model can draw a token hundreds of ranks down and then
  # repeat it until max_tokens (docs/v100/Known-Issues.md). A request's own top_k wins.
  --preferred-sampling-params "{\"top_k\": ${SENSENOVA_TOP_K:-20}}"
  --sleep-on-idle
  # Builds FlashInfer's sampling kernels before serving (JIT, ~90 s when cold).
  --warmups sampling
)

if [[ -n "${SENSENOVA_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$SENSENOVA_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

cd "$REPO"
exec "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
