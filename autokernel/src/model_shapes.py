from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence


@dataclass(frozen=True)
class LinearCase:
    op_name: str
    m: int
    n: int
    k: int
    transpose: bool = True


@dataclass(frozen=True)
class AttentionCase:
    q_heads: int
    kv_heads: int
    q_len: int
    kv_len: int
    head_dim: int
    batch: int = 1


@dataclass(frozen=True)
class RopeCase:
    heads: int
    kv_heads: int
    seq_len: int
    head_dim: int
    batch: int = 1


@dataclass(frozen=True)
class NormCase:
    rows: int
    dims: int


@dataclass(frozen=True)
class TextModelShape:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.num_key_value_heads

    def quantized_linear_cases(self, tokens: int) -> Sequence[LinearCase]:
        h = self.hidden_size
        i = self.intermediate_size
        kv = self.kv_dim
        return (
            LinearCase("q_proj", m=tokens, n=h, k=h, transpose=True),
            LinearCase("k_proj", m=tokens, n=kv, k=h, transpose=True),
            LinearCase("v_proj", m=tokens, n=kv, k=h, transpose=True),
            LinearCase("o_proj", m=tokens, n=h, k=h, transpose=True),
            LinearCase("gate_proj", m=tokens, n=i, k=h, transpose=True),
            LinearCase("up_proj", m=tokens, n=i, k=h, transpose=True),
            LinearCase("down_proj", m=tokens, n=h, k=i, transpose=True),
        )

    def qvm_nontranspose_case(self, tokens: int) -> LinearCase:
        return LinearCase(
            op_name="nontranspose_probe",
            m=tokens,
            n=self.hidden_size,
            k=self.hidden_size,
            transpose=False,
        )

    def rms_norm_case(self, rows: int) -> NormCase:
        return NormCase(rows=rows, dims=self.hidden_size)

    def rope_case(self, seq_len: int) -> RopeCase:
        return RopeCase(
            heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            seq_len=seq_len,
            head_dim=self.head_dim,
        )

    def attention_case(self, q_len: int, kv_len: int) -> AttentionCase:
        return AttentionCase(
            q_heads=self.num_attention_heads,
            kv_heads=self.num_key_value_heads,
            q_len=q_len,
            kv_len=kv_len,
            head_dim=self.head_dim,
        )

    @classmethod
    def from_hf_config_json(cls, path: str | Path, *, name: str | None = None) -> "TextModelShape":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=name or data.get("_name_or_path", Path(path).stem),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
        )


# Best-effort public-config presets for immediate benchmarking.
# Prefer loading the exact model's config.json in production.
_BUILTINS: Dict[str, TextModelShape] = {
    "bonsai_8b": TextModelShape(
        name="bonsai_8b",
        hidden_size=4096,
        intermediate_size=12288,
        num_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "gemma4_e2b": TextModelShape(
        name="gemma4_e2b",
        hidden_size=1536,
        intermediate_size=6144,
        num_layers=35,
        num_attention_heads=8,
        num_key_value_heads=1,
    ),
    "gemma4_e4b": TextModelShape(
        name="gemma4_e4b",
        hidden_size=2560,
        intermediate_size=10240,
        num_layers=42,
        num_attention_heads=20,
        num_key_value_heads=2,
    ),
}


def get_builtin_shape(name: str) -> TextModelShape:
    try:
        return _BUILTINS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BUILTINS))
        raise KeyError(f"Unknown builtin shape {name!r}. Available: {available}") from exc


if __name__ == "__main__":
    for key, shape in _BUILTINS.items():
        print(key, shape)
