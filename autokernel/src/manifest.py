from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Sequence

ROI = Literal["critical", "important", "supporting", "optional"]
Phase = Literal["decode", "prefill", "both"]


@dataclass(frozen=True)
class DispatchRule:
    predicate: str
    rationale: str


@dataclass(frozen=True)
class MLXTarget:
    name: str
    roi: ROI
    phase: Phase
    public_ops: tuple[str, ...]
    source_files: tuple[str, ...]
    hot_functions: tuple[str, ...]
    dispatch_rules: tuple[DispatchRule, ...]
    notes: str = ""


TARGETS: Dict[str, MLXTarget] = {
    "quantized_dispatch_decode": MLXTarget(
        name="quantized_dispatch_decode",
        roi="critical",
        phase="decode",
        public_ops=("mlx.core.quantized_matmul",),
        source_files=(
            "mlx/backend/metal/quantized.cpp",
            "mlx/backend/metal/kernels/quantized.h",
            "mlx/backend/metal/kernels/quantized.metal",
            "mlx/backend/metal/kernels/quantized_nax.h",
            "mlx/backend/metal/kernels/quantized_nax.metal",
            "mlx/backend/metal/kernels/quantized_utils.h",
        ),
        hot_functions=(
            "QuantizedMatmul::eval_gpu",
            "dispatch_qmv",
            "qmv",
            "qmv_quad",
            "qmm",
            "qvm",
            "qvm_split_k",
            "qdot",
            "qdot_safe",
        ),
        dispatch_rules=(
            DispatchRule("transpose == true", "decode-style linear path for quantized_matmul default usage"),
            DispatchRule("M < vector_limit", "decode / tiny-batch path stays in qmv/qvm family"),
            DispatchRule("N % 8 == 0 and K % 512 == 0", "required for qmv_fast"),
            DispatchRule("((K == 128) or (K == 64 and bits >= 2)) and bits power-of-two", "required for qmv_quad"),
        ),
        notes="Highest-ROI target. Includes dispatch and inner loops.",
    ),
    "quantized_dispatch_prefill": MLXTarget(
        name="quantized_dispatch_prefill",
        roi="critical",
        phase="prefill",
        public_ops=("mlx.core.quantized_matmul",),
        source_files=(
            "mlx/backend/metal/quantized.cpp",
            "mlx/backend/metal/kernels/quantized.h",
            "mlx/backend/metal/kernels/quantized.metal",
            "mlx/backend/metal/kernels/quantized_nax.h",
            "mlx/backend/metal/kernels/quantized_nax.metal",
            "mlx/backend/metal/kernels/quantized_utils.h",
        ),
        hot_functions=(
            "QuantizedMatmul::eval_gpu",
            "qmm",
            "qmm_t_impl",
            "qmm_n_impl",
            "QuantizedBlockLoader",
            "gemm_loop_aligned",
            "gemm_loop_unaligned",
            "gemm_loop_finalize",
        ),
        dispatch_rules=(
            DispatchRule("M >= vector_limit", "prefill path routes to qmm"),
            DispatchRule("transpose == true and K % 64 == 0 and NAX available", "NAX qmm path becomes eligible"),
        ),
        notes="Second half of the quantized path. Prefill and batched prompt processing.",
    ),
    "quantized_qvm_split_k_large_k": MLXTarget(
        name="quantized_qvm_split_k_large_k",
        roi="important",
        phase="decode",
        public_ops=("mlx.core.quantized_matmul",),
        source_files=(
            "mlx/backend/metal/quantized.cpp",
            "mlx/backend/metal/kernels/quantized.h",
            "mlx/backend/metal/kernels/quantized.metal",
        ),
        hot_functions=("qvm_split_k", "affine_qvm_split_k"),
        dispatch_rules=(
            DispatchRule("transpose == false", "non-transposed path only"),
            DispatchRule("M < 4", "vector path only when non-transposed"),
            DispatchRule("K >= 1024", "required for split-k qvm"),
        ),
        notes="Only matters if the model or benchmark uses non-transposed quantized matmul.",
    ),
    "sdpa_vector_decode": MLXTarget(
        name="sdpa_vector_decode",
        roi="critical",
        phase="decode",
        public_ops=("mlx.core.fast.scaled_dot_product_attention",),
        source_files=(
            "mlx/backend/metal/scaled_dot_product_attention.cpp",
            "mlx/backend/metal/kernels/scaled_dot_product_attention.metal",
            "mlx/backend/metal/kernels/sdpa_vector.h",
        ),
        hot_functions=("ScaledDotProductAttention::eval_gpu", "sdpa_vector"),
        dispatch_rules=(
            DispatchRule("query_sequence_length <= 8", "vector path only"),
            DispatchRule("query_head_dim == value_head_dim", "required"),
            DispatchRule("head_dim in {64, 96, 128, 256}", "vector supported head dims"),
            DispatchRule("(query_sequence_length * gqa_factor) <= 32", "vector path limit"),
        ),
        notes="Primary decode attention target.",
    ),
    "sdpa_vector_2pass_long_kv": MLXTarget(
        name="sdpa_vector_2pass_long_kv",
        roi="important",
        phase="decode",
        public_ops=("mlx.core.fast.scaled_dot_product_attention",),
        source_files=(
            "mlx/backend/metal/scaled_dot_product_attention.cpp",
            "mlx/backend/metal/kernels/scaled_dot_product_attention.metal",
            "mlx/backend/metal/kernels/sdpa_vector.h",
        ),
        hot_functions=("sdpa_vector_2pass", "sdpa_vector_2pass_1", "sdpa_vector_2pass_2"),
        dispatch_rules=(
            DispatchRule("short query, longer KV", "long-context decode stress"),
            DispatchRule("query_sequence_length <= 8", "still on vector family"),
        ),
        notes="Important for long-context decode and grouped-query attention.",
    ),
    "rms_norm_decode": MLXTarget(
        name="rms_norm_decode",
        roi="important",
        phase="both",
        public_ops=("mlx.core.fast.rms_norm",),
        source_files=(
            "mlx/backend/metal/normalization.cpp",
            "mlx/backend/metal/kernels/rms_norm.metal",
        ),
        hot_functions=("RMSNorm::eval_gpu", "rms_single_row", "rms_looped"),
        dispatch_rules=(
            DispatchRule("axis_size <= RMS_LOOPED_LIMIT", "single-row/block path"),
            DispatchRule("axis_size > RMS_LOOPED_LIMIT", "looped path"),
        ),
        notes="Small per-call cost, but extremely frequent in decode.",
    ),
    "rope_single_token": MLXTarget(
        name="rope_single_token",
        roi="important",
        phase="decode",
        public_ops=("mlx.core.fast.rope",),
        source_files=(
            "mlx/backend/metal/rope.cpp",
            "mlx/backend/metal/kernels/rope.metal",
        ),
        hot_functions=("RoPE::eval_gpu", "rope", "rope_freqs"),
        dispatch_rules=(
            DispatchRule("T == 1 and contiguous and scalar offset", "single-token inference special case"),
            DispatchRule("with_freqs / large variants separate", "benchmark separately if used"),
        ),
        notes="Per-layer decode overhead target.",
    ),
    "gemv_logits": MLXTarget(
        name="gemv_logits",
        roi="supporting",
        phase="both",
        public_ops=("mlx.core.matmul",),
        source_files=(
            "mlx/backend/metal/matmul.cpp",
            "mlx/backend/metal/kernels/gemv.metal",
        ),
        hot_functions=("gemv", "gemv_t"),
        dispatch_rules=(
            DispatchRule("min(M, N) == 1", "gemv path from matmul.cpp"),
        ),
        notes="Useful for logits / narrow matrices if real traces show it matters.",
    ),
    "gemv_masked_optional": MLXTarget(
        name="gemv_masked_optional",
        roi="optional",
        phase="both",
        public_ops=("mlx.core.matmul",),
        source_files=(
            "mlx/backend/metal/kernels/gemv_masked.h",
            "mlx/backend/metal/kernels/gemv_masked.metal",
        ),
        hot_functions=("gemv_masked", "gemv_t_masked"),
        dispatch_rules=(),
        notes="Only prioritize if masked GEMV shows up in traces.",
    ),
    "softmax_standalone": MLXTarget(
        name="softmax_standalone",
        roi="supporting",
        phase="both",
        public_ops=("mlx.core.softmax",),
        source_files=(
            "mlx/backend/metal/softmax.cpp",
            "mlx/backend/metal/kernels/softmax.h",
            "mlx/backend/metal/kernels/softmax.metal",
        ),
        hot_functions=("Softmax::eval_gpu", "softmax_single_row", "softmax_looped"),
        dispatch_rules=(
            DispatchRule("axis_size > 4096", "looped path"),
            DispatchRule("axis_size <= 4096", "block path"),
        ),
        notes="Not part of SDPA in the current MLX Metal implementation.",
    ),
    "layer_norm_optional": MLXTarget(
        name="layer_norm_optional",
        roi="optional",
        phase="both",
        public_ops=("mlx.core.fast.layer_norm",),
        source_files=("mlx/backend/metal/kernels/layer_norm.metal",),
        hot_functions=("layer_norm_single_row", "layer_norm_looped"),
        dispatch_rules=(),
        notes="Only if the target model uses LayerNorm.",
    ),
    "hadamard_optional": MLXTarget(
        name="hadamard_optional",
        roi="optional",
        phase="both",
        public_ops=("mlx.core.hadamard_transform",),
        source_files=(
            "mlx/backend/metal/hadamard.cpp",
            "mlx/backend/metal/kernels/hadamard.h",
        ),
        hot_functions=("hadamard_n", "hadamard_m"),
        dispatch_rules=(),
        notes="Only if the quantization / rotation pipeline actually uses it in inference.",
    ),
}


def get_target(name: str) -> MLXTarget:
    try:
        return TARGETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TARGETS))
        raise KeyError(f"Unknown MLX target {name!r}. Available: {available}") from exc


def list_targets() -> Sequence[MLXTarget]:
    return tuple(TARGETS[name] for name in sorted(TARGETS))


if __name__ == "__main__":
    for target in list_targets():
        print(f"{target.name:28}  {target.roi:10}  phase={target.phase}")
