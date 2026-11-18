"""SM70 SenseNova-Loop mHC router and merge kernels against the torch reference.

Not a CI test. For T in 1, 2, 3, 7 and 4096, random and realistic inputs:
  - pre/post/mix against the fp32 torch router (the model's torch path) and an
    fp64 reference; the kernel must sit at least as close to fp64 as torch fp32;
  - branch and merged in fp16 ulps against torch, and bitwise against the torch
    formula fed with the kernel's own coefficients;
  - the fused norm bitwise against sgl_kernel gemma_rmsnorm;
  - merge_route bitwise against merge followed by route;
  - each token's outputs bitwise equal whatever T.
Then times one mHC layer (attention read, FFN read, FFN update) against the
torch chain:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_sensenova_hc.py [model dir]

With a model directory (SenseNova-Loop-9B) the realistic cases use its router
and norm weights.
"""

import json
import os
import sys

import torch
from torch.nn import functional as F

from sglang.kernels.ops.elementwise.sm70_sensenova_hc import (
    sensenova_hc_merge,
    sensenova_hc_merge_route,
    sensenova_hc_route,
)

ITERS = 20
RMS_EPS = 1e-6
NORM_EPS = 1e-6
TOKENS = (1, 2, 3, 7, 4096)


def torch_coeffs(x, weight, bias, dtype, iterations=ITERS):
    flat = x.to(dtype)
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    logits = F.linear(normalized, weight.to(dtype), bias.to(dtype))
    pre = logits[..., :2].sigmoid()
    post = 2 * logits[..., 2:4].sigmoid()
    mix = logits[..., 4:].unflatten(-1, (2, 2))
    for _ in range(iterations):
        mix = mix - torch.logsumexp(mix, dim=-1, keepdim=True)
        mix = mix - torch.logsumexp(mix, dim=-2, keepdim=True)
    return pre, post, mix.exp()


def torch_branch(x, pre):
    return torch.bmm(pre.to(x.dtype).unsqueeze(-2), x.unflatten(-1, (2, -1))).squeeze(
        -2
    )


def torch_merge(branch, residual, mix, post):
    residual = residual.unflatten(-1, (2, -1))
    mixed = torch.bmm(mix.to(residual.dtype), residual)
    update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
    return (mixed + update).flatten(-2)


def _bits(t):
    return (
        t.contiguous().view(torch.int16)
        if t.dtype == torch.float16
        else t.contiguous().view(torch.int32)
    )


def _ulps(a, b):
    def order(t):
        v = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(v < 0, -(v & 0x7FFF), v)

    return (order(a) - order(b)).abs()


def _assert_bitwise(a, b, what):
    if torch.equal(_bits(a), _bits(b)):
        return
    diff = _bits(a) != _bits(b)
    first = tuple(diff.nonzero()[0].tolist())
    worst = int(_ulps(a, b).max()) if a.dtype == torch.float16 else _max_abs(a, b)
    raise AssertionError(
        f"{what}: {int(diff.sum())} elements differ, worst {worst}, "
        f"first at {first}: {a[first].item()} vs {b[first].item()}"
    )


def _term_ulps(a, b, terms):
    """|a - b| in fp16 ulps of the largest term summed into the element."""
    scale = torch.stack([t.abs().float() for t in terms]).amax(0).clamp_min(2.0**-14)
    return (a.float() - b.float()).abs() / torch.exp2(
        torch.floor(torch.log2(scale)) - 10
    )


def _fp16_report(k, t, x, terms):
    """Elements where kernel and torch differ (max in term ulps), then elements off
    the same formula fed fp64 coefficients: kernel / torch."""
    diff = _bits(k) != _bits(t)
    worst = _term_ulps(k, t, terms).max().item()
    return (
        f"{int(diff.sum())} ({worst:.2g})",
        f"{int((_bits(k) != _bits(x)).sum())}/{int((_bits(t) != _bits(x)).sum())}",
    )


def _max_abs(a, b):
    return (a.double() - b.double()).abs().max().item()


def _streams(kind, tokens, gen):
    if kind == "random":
        return (torch.randn(tokens, 8192, device="cuda", generator=gen) * 3).half()
    x = torch.randn(tokens, 8192, device="cuda", generator=gen) * 0.6
    cols = torch.randperm(8192, device="cuda", generator=gen)[:32]
    x[:, cols] = (torch.rand(tokens, 32, device="cuda", generator=gen) * 2 - 1) * 520
    if kind == "expanded":
        x[:, 4096:] = x[:, :4096]
    return x.half()


def _model_weights(model_dir):
    from safetensors import safe_open

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))[
        "weight_map"
    ]

    def get(name):
        with safe_open(os.path.join(model_dir, index[name]), "pt") as f:
            return f.get_tensor(name).cuda()

    prefix = "model.language_model.layers"
    routers = [
        (
            f"layer {i} {kind}",
            get(f"{prefix}.{i}.mhc_{kind}.weight").float(),
            get(f"{prefix}.{i}.mhc_{kind}.bias").float(),
        )
        for i in (4, 15, 27)
        for kind in ("attn", "ffn")
    ]
    norms = {
        f"layer {i} {kind}": get(f"{prefix}.{i}.{kind}.weight").half()
        for i in (4, 27)
        for kind in ("input_layernorm", "post_attention_layernorm")
    }
    return routers, norms


def _synthetic_weights(gen):
    bias = torch.tensor([-1.1, 1.1, 0.0, 0.0, 2.7, 0.0, 0.0, 2.7], device="cuda")
    routers = [
        ("synthetic", torch.randn(8, 8192, device="cuda", generator=gen) * 0.0026, bias)
    ]
    return routers, {
        "randn": (torch.randn(4096, device="cuda", generator=gen) * 0.2).half()
    }


def _norm_weights(gen):
    special = torch.randn(4096, device="cuda", generator=gen).half()
    special[::97] = 0.0
    special[1::97] = -0.0
    special[2::97] = 65504.0
    special[3::97] = -1.0
    return {
        "signed zeros, extremes, -1": special,
        "zeros": torch.zeros(4096, device="cuda", dtype=torch.float16),
    }


def _cases(model_dir, gen):
    routers, norms = _model_weights(model_dir) if model_dir else _synthetic_weights(gen)
    norms.update(_norm_weights(gen))
    rand_w = torch.randn(8, 8192, device="cuda", generator=gen) * 0.02
    cases = [
        ("random", "random", rand_w, torch.randn(8, device="cuda", generator=gen) * 2)
    ]
    for name, w, b in routers:
        cases.append((f"realistic {name}", "realistic", w, b))
        cases.append((f"expanded {name}", "expanded", w, b))
    # Mix logits spread 300 (linear-space loop, slow convergence) and 1600 (log-space loop).
    for spread in (300.0, 1600.0):
        b = torch.zeros(8, device="cuda")
        b[4:] = torch.tensor(
            [spread / 2, -spread / 2, -spread / 2, spread / 4], device="cuda"
        )
        cases.append((f"mix spread {spread:.0f}", "random", rand_w * 0.1, b))
    return cases, norms


class Errors:
    def __init__(self):
        self.rows = []

    def add(self, *row):
        self.rows.append(row)

    def print(self):
        print(
            "\n| case | T | pre k-t | post k-t | mix k-t | pre k/t-fp64 | post k/t-fp64 | mix k/t-fp64 "
            "| branch k!=t (term ulps) | branch off fp64-coeff k/t | merged k!=t (term ulps) "
            "| merged off fp64-coeff k/t |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            print("| " + " | ".join(str(v) for v in r) + " |")


def check(model_dir) -> None:
    from sgl_kernel import gemma_rmsnorm

    gen = torch.Generator(device="cuda").manual_seed(7)
    cases, norms = _cases(model_dir, gen)
    errors = Errors()
    for case, kind, weight, bias in cases:
        for tokens in TOKENS:
            x = _streams(kind, tokens, gen)
            pre, post, mix = torch_coeffs(x, weight, bias, torch.float32)
            pre64, post64, mix64 = torch_coeffs(x, weight, bias, torch.float64)
            branch_t = torch_branch(x, pre)

            branch_k, pre_k, post_k, mix_k = sensenova_hc_route(
                x, weight, bias, ITERS, RMS_EPS
            )
            e_k = [
                _max_abs(a, b)
                for a, b in zip((pre_k, post_k, mix_k), (pre64, post64, mix64))
            ]
            e_t = [
                _max_abs(a, b) for a, b in zip((pre, post, mix), (pre64, post64, mix64))
            ]
            assert all(k <= t for k, t in zip(e_k, e_t)), (
                f"{case} T={tokens}: kernel {e_k} vs torch {e_t}"
            )
            _assert_bitwise(
                branch_k, torch_branch(x, pre_k), f"{case} T={tokens}: branch"
            )
            p16 = pre.half().float()
            branch_report = _fp16_report(
                branch_k,
                branch_t,
                torch_branch(x, pre64),
                (p16[:, :1] * x[:, :4096].float(), p16[:, 1:] * x[:, 4096:].float()),
            )

            # FFN read: merge the attention output with this router's coefficients, route again.
            attn_out = (
                torch.randn(tokens, 4096, device="cuda", generator=gen) * 2
            ).half()
            merged_t = torch_merge(attn_out, x, mix, post)
            merged_k = sensenova_hc_merge(attn_out, x, mix_k, post_k)
            _assert_bitwise(
                merged_k,
                torch_merge(attn_out, x, mix_k, post_k),
                f"{case} T={tokens}: merge",
            )
            m16, q16, r = (
                mix.half().float(),
                post.half().float(),
                x.float().unflatten(-1, (2, -1)),
            )
            terms = [
                torch.stack(
                    [
                        m16[:, j, 0:1] * r[:, 0],
                        m16[:, j, 1:2] * r[:, 1],
                        q16[:, j : j + 1] * attn_out.float(),
                    ],
                    0,
                )
                for j in range(2)
            ]
            merged_report = _fp16_report(
                merged_k,
                merged_t,
                torch_merge(attn_out, x, mix64, post64),
                [torch.cat([t[i] for t in terms], -1) for i in range(3)],
            )
            nw = next(iter(norms.values()))
            fused = sensenova_hc_merge_route(
                attn_out, x, mix_k, post_k, weight, bias, ITERS, RMS_EPS, nw, NORM_EPS
            )
            plain = sensenova_hc_route(merged_k, weight, bias, ITERS, RMS_EPS)
            _assert_bitwise(
                fused[1], merged_k, f"{case} T={tokens}: merge_route merged"
            )
            for name, a, b in zip(("pre", "post", "mix"), fused[2:], plain[1:]):
                _assert_bitwise(a, b, f"{case} T={tokens}: merge_route {name}")
            _assert_bitwise(
                fused[0],
                gemma_rmsnorm(plain[0], nw, NORM_EPS),
                f"{case} T={tokens}: merge_route norm",
            )

            for wname, w in norms.items():
                normed = sensenova_hc_route(
                    x, weight, bias, ITERS, RMS_EPS, w, NORM_EPS
                )[0]
                want = gemma_rmsnorm(branch_k, w, NORM_EPS)
                _assert_bitwise(normed, want, f"{case} T={tokens} norm {wname}")

            if tokens > 1:
                for t in sorted({0, 1, tokens // 2, tokens - 1}):
                    one = sensenova_hc_route(x[t : t + 1], weight, bias, ITERS, RMS_EPS)
                    for a, b in zip(one, (branch_k, pre_k, post_k, mix_k)):
                        _assert_bitwise(
                            a, b[t : t + 1], f"{case} T={tokens}: row {t} varies"
                        )

            errors.add(
                case,
                tokens,
                *(
                    "%.1e" % _max_abs(a, b)
                    for a, b in zip((pre_k, post_k, mix_k), (pre, post, mix))
                ),
                *("%.1e/%.1e" % (k, t) for k, t in zip(e_k, e_t)),
                *branch_report,
                *merged_report,
            )
    errors.print()
    print(
        "\nkernel <= torch fp32 error vs fp64 for pre/post/mix; branch, merge, merge_route and "
        f"the fused norm ({len(norms)} weights) bitwise as specified; rows batch invariant"
    )


def check_iterations(model_dir) -> None:
    gen = torch.Generator(device="cuda").manual_seed(5)
    cases, _ = _cases(model_dir, gen)
    for iterations in (1, 2, 3, 50):
        worst_k = worst_t = 0.0
        for case, kind, weight, bias in cases:
            x = _streams(kind, 64, gen)
            mix = torch_coeffs(x, weight, bias, torch.float32, iterations)[2]
            mix64 = torch_coeffs(x, weight, bias, torch.float64, iterations)[2]
            mix_k = sensenova_hc_route(x, weight, bias, iterations, RMS_EPS)[3]
            e_k, e_t = _max_abs(mix_k, mix64), _max_abs(mix, mix64)
            assert e_k <= e_t, (
                f"{case} iterations={iterations}: mix {e_k} vs torch {e_t}"
            )
            worst_k, worst_t = max(worst_k, e_k), max(worst_t, e_t)
        print(
            f"iterations={iterations}: mix max err vs fp64 kernel {worst_k:.1e} torch fp32 {worst_t:.1e} "
            f"({len(cases)} cases, T=64)"
        )


def _graph_time(fn, reps=20):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    best = 1e9
    for _ in range(20):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / reps * 1e3)
    return best


def _eager_time(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3


def time_layer() -> None:
    from sgl_kernel import gemma_rmsnorm

    gen = torch.Generator(device="cuda").manual_seed(11)
    w_attn, w_ffn = (
        torch.randn(8, 8192, device="cuda", generator=gen) * 0.0026 for _ in range(2)
    )
    bias = torch.tensor([-1.1, 1.1, 0.0, 0.0, 2.7, 0.0, 0.0, 2.7], device="cuda")
    nw = (torch.randn(4096, device="cuda", generator=gen) * 0.2).half()
    print(
        "\n| T | torch chain us/layer | kernels us/layer | route_norm us | merge_route_norm us | merge us |"
    )
    print("|---|---|---|---|---|---|")
    for tokens in (1, 2, 3, 4096):
        x = _streams("realistic", tokens, gen)
        attn_out = torch.randn(tokens, 4096, device="cuda", generator=gen).half()
        ffn_out = torch.randn(tokens, 4096, device="cuda", generator=gen).half()

        def torch_layer():
            pre, post, mix = torch_coeffs(x, w_attn, bias, torch.float32)
            gemma_rmsnorm(torch_branch(x, pre), nw, NORM_EPS)
            merged = torch_merge(attn_out, x, mix, post)
            pre, post, mix = torch_coeffs(merged, w_ffn, bias, torch.float32)
            gemma_rmsnorm(torch_branch(merged, pre), nw, NORM_EPS)
            torch_merge(ffn_out, merged, mix, post)

        _, _, post_a, mix_a = sensenova_hc_route(
            x, w_attn, bias, ITERS, RMS_EPS, nw, NORM_EPS
        )

        def route():
            sensenova_hc_route(x, w_attn, bias, ITERS, RMS_EPS, nw, NORM_EPS)

        def merge_route():
            sensenova_hc_merge_route(
                attn_out, x, mix_a, post_a, w_ffn, bias, ITERS, RMS_EPS, nw, NORM_EPS
            )

        def merge():
            sensenova_hc_merge(ffn_out, x, mix_a, post_a)

        def kernel_layer():
            _, _, post, mix = sensenova_hc_route(
                x, w_attn, bias, ITERS, RMS_EPS, nw, NORM_EPS
            )
            _, merged, _, post, mix = sensenova_hc_merge_route(
                attn_out, x, mix, post, w_ffn, bias, ITERS, RMS_EPS, nw, NORM_EPS
            )
            sensenova_hc_merge(ffn_out, merged, mix, post)

        timer = _graph_time if tokens <= 3 else _eager_time
        times = [
            timer(f) for f in (torch_layer, kernel_layer, route, merge_route, merge)
        ]
        print(f"| {tokens} | " + " | ".join("%.1f" % t for t in times) + " |")


def main() -> None:
    model_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
    check(model_dir)
    check_iterations(model_dir)
    time_layer()


if __name__ == "__main__":
    main()
