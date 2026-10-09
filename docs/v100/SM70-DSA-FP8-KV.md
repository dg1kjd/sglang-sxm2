# FP8 E4M3 KV cache for DSA models on SM70 (V100)

Upstream locks the DSA/MLA KV cache to FP16 on Volta: the SM70 sparse MLA kernel
reads the pool directly, and `--kv-cache-dtype fp8_e4m3` otherwise resolves to the
Hopper-only `flashmla_kv` backend, which does not exist on sm_70.

This change lets the 11 DSA/MLA layers of GLM-5.3-Flash store their 512-d latent as
unscaled E4M3 bytes on V100. Volta has no FP8 arithmetic and no hardware FP8->FP16
convert, so rows are widened with a software conversion while they are staged into
shared memory; the softmax and accumulate loops are unchanged.

## Capacity

Per token over the 11 DSA layers (the latent plus the uint8 indexer K cache):

| | FP16 | FP8 E4M3 |
|---|---|---|
| latent 512 | 1024 B | 512 B |
| indexer K (128 + scale) | 132 B | 132 B |
| per layer | 1156 B | 644 B |
| x11 layers | 12,716 B | 7,084 B |

At `--mem-fraction-static 0.945` the pool grows from 290,240 to 521,024 tokens, which
supports `--context-length 480000` (FP16 tops out at 290,240).

## Enabling

```bash
export SGLANG_SM70_DSA_FP8_KV=1
python -m sglang.launch_server ... --kv-cache-dtype fp8_e4m3 --context-length 480000
```

Without the env var, SM70 + `fp8_e4m3` fails at argument resolution instead of
selecting a backend that cannot load.

## What changed

- `kernels/jit/csrc/attention/sparse_mla_sm70.cuh`: `kv` accepts fp16 or uint8 (E4M3);
  new `<kVec16, kFp8>` instance; the fp8 branch widens eight bytes per load with
  `__nv_cvt_fp8x2_to_halfraw2(..., __NV_E4M3)`. The fp16 code paths are unchanged.
- `kernels/ops/attention/sparse_mla_sm70.py`: accepts `float8_e4m3fn` (viewed as uint8).
- `arg_groups/overrides.py`: SM70 + `fp8_e4m3` keeps the FP16 path backend names so no
  FlashMLA metadata is built during CUDA graph capture.
- `mem_cache/kv_cache_configurator.py`: SM70 keeps the raw 512-wide layout instead of
  the scale-packed FlashMLA one. Without this the write path reaches a Triton FP8 kernel
  that Triton cannot compile for sm_70 (`fp8e4nv` not supported).
- `srt/environ.py`: `SGLANG_SM70_DSA_FP8_KV` (default off).

## Verification (8x V100-SXM2, GLM-5.3-Flash-NVFP4, TP8, MTP 3/4)

- Kernel unit test, 48 cases: the FP8 path is bit-identical to the fp16 path reading the
  same already-quantized data; quantization error rel-L2 <= 0.052, cosine >= 0.9986.
- Eight short prompts: outputs identical to FP16.
- Needle recall at 94k and 188k prompt tokens: found in both. 234k at 0.15 depth: missed
  in both (model behaviour, not the KV dtype).
- Real extraction task (4 days of construction logs, 34k tokens): field accuracy 0.5849 in
  both, identical per day.
- Speed, same harness: decode 139.8 -> 144.2 tok/s; cold prefill (56k) 2215 -> 2101 tok/s
  (-5%). The prefill cost is the software FP8 widening; decode gains from halving the KV
  read traffic.
- 480k context: needle found at 375,201 and 412,703 prompt tokens.

## Reverting

`--kv-cache-dtype auto`. FP16 context is limited to 290,240 tokens.
