from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .model_shapes import TextModelShape, get_builtin_shape
from .results import ResultRow, append_rows


class MLXUnavailableError(RuntimeError):
    pass


def _require_mlx():
    try:
        import mlx.core as mx
        return mx
    except Exception as exc:  # pragma: no cover - import path depends on host env
        raise MLXUnavailableError(
            "MLX is not importable. Install the MLX Python package in the target environment "
            "before running this benchmark."
        ) from exc


@dataclass(frozen=True)
class BenchStats:
    median_ms: float
    min_ms: float
    max_ms: float


def _to_numpy(x) -> np.ndarray:
    return np.array(x)


def _eval(mx, *values) -> None:
    if len(values) == 1:
        mx.eval(values[0])
    else:
        mx.eval(*values)


def _time_op(mx, fn, *, warmup: int, iters: int) -> BenchStats:
    for _ in range(warmup):
        out = fn()
        _eval(mx, out)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        _eval(mx, out)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    arr = np.array(samples, dtype=np.float64)
    return BenchStats(
        median_ms=float(np.median(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
    )


def _assert_close(actual, expected, *, atol: float, rtol: float, label: str) -> None:
    a = _to_numpy(actual).astype(np.float32, copy=False)
    b = _to_numpy(expected).astype(np.float32, copy=False)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.abs(a - b)
        raise AssertionError(
            f"{label}: max_abs={float(diff.max()):.6e}, mean_abs={float(diff.mean()):.6e}, "
            f"atol={atol}, rtol={rtol}"
        )


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


def ref_quantized_matmul(mx, x, wq, scales, biases, *, transpose: bool, group_size: int, bits: int, mode: str):
    w = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits, mode=mode)
    if transpose:
        w = mx.swapaxes(w, -1, -2)
    return mx.matmul(x, w)


def ref_rms_norm(mx, x, weight, eps: float):
    x2 = x.astype(mx.float32)
    denom = mx.rsqrt(mx.mean(mx.square(x2), axis=-1, keepdims=True) + eps)
    out = x * denom.astype(x.dtype)
    if weight is not None:
        out = out * weight
    return out


def ref_sdpa(mx, q, k, v, *, scale: float, causal: bool):
    scores = mx.matmul(q.astype(mx.float32), mx.swapaxes(k.astype(mx.float32), -1, -2)) * scale
    scores_np = _to_numpy(scores)
    if causal:
        q_len = scores_np.shape[-2]
        kv_len = scores_np.shape[-1]
        mask = np.triu(np.ones((q_len, kv_len), dtype=bool), k=1 + (kv_len - q_len))
        scores_np = np.where(mask, -np.inf, scores_np)
    probs = _softmax_np(scores_np, axis=-1)
    probs_mx = mx.array(probs, dtype=mx.float32)
    return mx.matmul(probs_mx, v.astype(mx.float32)).astype(v.dtype)


def ref_rope(mx, x, *, dims: int, traditional: bool, base: float, scale: float, offset: int):
    if traditional:
        raise NotImplementedError("This scaffold implements only traditional=False RoPE.")
    x_np = _to_numpy(x).astype(np.float32, copy=False)
    seq_len = x_np.shape[-2]
    d = dims
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    positions = (np.arange(seq_len, dtype=np.float32) + float(offset)) * float(scale)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs)
    sin = np.sin(freqs)

    out = x_np.copy()
    head = out[..., :d]
    x_even = head[..., 0::2]
    x_odd = head[..., 1::2]
    head[..., 0::2] = x_even * cos - x_odd * sin
    head[..., 1::2] = x_even * sin + x_odd * cos
    out[..., :d] = head
    return mx.array(out, dtype=x.dtype)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _dtype_from_name(mx, name: str):
    mapping = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Choose from {sorted(mapping)}") from exc


def _load_shape(args: argparse.Namespace) -> TextModelShape:
    if args.config_json:
        return TextModelShape.from_hf_config_json(args.config_json, name=args.model_shape or None)
    if not args.model_shape:
        raise ValueError("Either --model-shape or --config-json is required.")
    return get_builtin_shape(args.model_shape)


def _quantized_case(mx, *, m: int, n: int, k: int, dtype_name: str, bits: int, group_size: int, seed: int):
    gen = _rng(seed)
    dtype = _dtype_from_name(mx, dtype_name)
    x = mx.array(gen.standard_normal((m, k), dtype=np.float32), dtype=dtype)
    w_dense = mx.array(gen.standard_normal((n, k), dtype=np.float32), dtype=dtype)
    wq, scales, biases = mx.quantize(w_dense, group_size=group_size, bits=bits, mode="affine")
    return x, w_dense, wq, scales, biases


def _rms_case(mx, *, rows: int, dims: int, dtype_name: str, seed: int):
    gen = _rng(seed)
    dtype = _dtype_from_name(mx, dtype_name)
    x = mx.array(gen.standard_normal((rows, dims), dtype=np.float32), dtype=dtype)
    w = mx.array(gen.standard_normal((dims,), dtype=np.float32), dtype=dtype)
    return x, w


def _rope_case(mx, *, heads: int, seq_len: int, head_dim: int, batch: int, dtype_name: str, seed: int):
    gen = _rng(seed)
    dtype = _dtype_from_name(mx, dtype_name)
    x = mx.array(gen.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32), dtype=dtype)
    return x


def _sdpa_case(mx, *, batch: int, q_heads: int, kv_heads: int, q_len: int, kv_len: int, head_dim: int, dtype_name: str, seed: int):
    gen = _rng(seed)
    dtype = _dtype_from_name(mx, dtype_name)
    q = mx.array(gen.standard_normal((batch, q_heads, q_len, head_dim), dtype=np.float32), dtype=dtype)
    k = mx.array(gen.standard_normal((batch, kv_heads, kv_len, head_dim), dtype=np.float32), dtype=dtype)
    v = mx.array(gen.standard_normal((batch, kv_heads, kv_len, head_dim), dtype=np.float32), dtype=dtype)
    return q, k, v


def bench_quantized_matmul(args: argparse.Namespace) -> list[ResultRow]:
    mx = _require_mlx()
    shape = _load_shape(args)
    tokens = 1 if args.phase == "decode" else args.prefill_tokens
    rows: list[ResultRow] = []
    atol = 2e-2 if args.dtype != "float32" else 1e-4
    rtol = 2e-2 if args.dtype != "float32" else 1e-4

    for idx, case in enumerate(shape.quantized_linear_cases(tokens), start=1):
        x, _w_dense, wq, scales, biases = _quantized_case(
            mx,
            m=case.m,
            n=case.n,
            k=case.k,
            dtype_name=args.dtype,
            bits=args.bits,
            group_size=args.group_size,
            seed=idx,
        )
        actual = mx.quantized_matmul(
            x,
            wq,
            scales,
            biases,
            transpose=case.transpose,
            group_size=args.group_size,
            bits=args.bits,
            mode="affine",
        )
        expected = ref_quantized_matmul(
            mx,
            x,
            wq,
            scales,
            biases,
            transpose=case.transpose,
            group_size=args.group_size,
            bits=args.bits,
            mode="affine",
        )
        _eval(mx, actual, expected)
        _assert_close(actual, expected, atol=atol, rtol=rtol, label=case.op_name)

        stats = _time_op(
            mx,
            lambda: mx.quantized_matmul(
                x,
                wq,
                scales,
                biases,
                transpose=case.transpose,
                group_size=args.group_size,
                bits=args.bits,
                mode="affine",
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        flops = 2.0 * case.m * case.n * case.k
        tflops = flops / (stats.median_ms / 1000.0) / 1e12
        rows.append(
            ResultRow(
                experiment=idx,
                tag=args.phase,
                kernel_type="quantized_matmul",
                throughput_tflops=tflops,
                latency_us=stats.median_ms * 1000.0,
                pct_peak=0.0,
                speedup_vs_baseline=1.0,
                correctness="PASS",
                peak_vram_mb=0.0,
                description=f"{shape.name}:{case.op_name}:M{case.m}:N{case.n}:K{case.k}:bits{args.bits}:gs{args.group_size}",
            )
        )
    return rows


def bench_rms_norm(args: argparse.Namespace) -> list[ResultRow]:
    mx = _require_mlx()
    shape = _load_shape(args)
    rows = 1 if args.phase == "decode" else args.prefill_tokens
    x, w = _rms_case(mx, rows=rows, dims=shape.hidden_size, dtype_name=args.dtype, seed=1)
    actual = mx.fast.rms_norm(x, w, args.eps)
    expected = ref_rms_norm(mx, x, w, args.eps)
    _eval(mx, actual, expected)
    _assert_close(actual, expected, atol=2e-2, rtol=2e-2, label="rms_norm")
    stats = _time_op(mx, lambda: mx.fast.rms_norm(x, w, args.eps), warmup=args.warmup, iters=args.iters)
    flops = 6.0 * rows * shape.hidden_size
    tflops = flops / (stats.median_ms / 1000.0) / 1e12
    return [
        ResultRow(
            experiment=1,
            tag=args.phase,
            kernel_type="rms_norm",
            throughput_tflops=tflops,
            latency_us=stats.median_ms * 1000.0,
            pct_peak=0.0,
            speedup_vs_baseline=1.0,
            correctness="PASS",
            peak_vram_mb=0.0,
            description=f"{shape.name}:rows{rows}:dims{shape.hidden_size}",
        )
    ]


def bench_rope(args: argparse.Namespace) -> list[ResultRow]:
    mx = _require_mlx()
    shape = _load_shape(args)
    seq_len = 1 if args.phase == "decode" else args.prefill_tokens
    x = _rope_case(
        mx,
        heads=shape.num_attention_heads,
        seq_len=seq_len,
        head_dim=shape.head_dim,
        batch=1,
        dtype_name=args.dtype,
        seed=1,
    )
    actual = mx.fast.rope(
        x,
        shape.head_dim,
        traditional=False,
        base=args.base,
        scale=args.rope_scale,
        offset=args.offset,
    )
    expected = ref_rope(
        mx,
        x,
        dims=shape.head_dim,
        traditional=False,
        base=args.base,
        scale=args.rope_scale,
        offset=args.offset,
    )
    _eval(mx, actual, expected)
    _assert_close(actual, expected, atol=2e-2, rtol=2e-2, label="rope")
    stats = _time_op(
        mx,
        lambda: mx.fast.rope(
            x,
            shape.head_dim,
            traditional=False,
            base=args.base,
            scale=args.rope_scale,
            offset=args.offset,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    flops = 6.0 * 1 * shape.num_attention_heads * seq_len * shape.head_dim
    tflops = flops / (stats.median_ms / 1000.0) / 1e12
    return [
        ResultRow(
            experiment=1,
            tag=args.phase,
            kernel_type="rope",
            throughput_tflops=tflops,
            latency_us=stats.median_ms * 1000.0,
            pct_peak=0.0,
            speedup_vs_baseline=1.0,
            correctness="PASS",
            peak_vram_mb=0.0,
            description=f"{shape.name}:heads{shape.num_attention_heads}:seq{seq_len}:head_dim{shape.head_dim}",
        )
    ]


def bench_sdpa(args: argparse.Namespace) -> list[ResultRow]:
    mx = _require_mlx()
    shape = _load_shape(args)
    q_len = 1 if args.phase == "decode" else min(args.prefill_tokens, 8)
    kv_len = args.context_len if args.phase == "decode" else args.prefill_tokens
    q, k, v = _sdpa_case(
        mx,
        batch=1,
        q_heads=shape.num_attention_heads,
        kv_heads=shape.num_key_value_heads,
        q_len=q_len,
        kv_len=kv_len,
        head_dim=shape.head_dim,
        dtype_name=args.dtype,
        seed=1,
    )
    scale = 1.0 / math.sqrt(shape.head_dim)
    actual = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if args.causal else None)
    expected = ref_sdpa(mx, q, k, v, scale=scale, causal=args.causal)
    _eval(mx, actual, expected)
    _assert_close(actual, expected, atol=3e-2, rtol=3e-2, label="sdpa")
    stats = _time_op(
        mx,
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if args.causal else None),
        warmup=args.warmup,
        iters=args.iters,
    )
    flops = 4.0 * 1 * shape.num_attention_heads * (q_len * kv_len) * shape.head_dim
    tflops = flops / (stats.median_ms / 1000.0) / 1e12
    return [
        ResultRow(
            experiment=1,
            tag=args.phase,
            kernel_type="sdpa",
            throughput_tflops=tflops,
            latency_us=stats.median_ms * 1000.0,
            pct_peak=0.0,
            speedup_vs_baseline=1.0,
            correctness="PASS",
            peak_vram_mb=0.0,
            description=f"{shape.name}:q{q_len}:kv{kv_len}:q_heads{shape.num_attention_heads}:kv_heads{shape.num_key_value_heads}:head_dim{shape.head_dim}",
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MLX hot-path ops through public MLX APIs.")
    parser.add_argument("--op", choices=("quantized_matmul", "rms_norm", "rope", "sdpa"), required=True)
    parser.add_argument("--model-shape", default=None, help="Builtin model shape name.")
    parser.add_argument("--config-json", default=None, help="Path to a config.json to derive shapes from.")
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--base", type=float, default=10000.0)
    parser.add_argument("--rope-scale", type=float, default=1.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output-tsv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.op == "quantized_matmul":
        rows = bench_quantized_matmul(args)
    elif args.op == "rms_norm":
        rows = bench_rms_norm(args)
    elif args.op == "rope":
        rows = bench_rope(args)
    elif args.op == "sdpa":
        rows = bench_sdpa(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported op {args.op!r}")

    for row in rows:
        print(
            f"{row.kernel_type:18}  latency_us={row.latency_us:10.2f}  "
            f"tflops={row.throughput_tflops:8.4f}  {row.description}"
        )

    if args.output_tsv:
        append_rows(args.output_tsv, rows)


if __name__ == "__main__":
    main()
