from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .manifest import MLXTarget, get_target, list_targets


@dataclass(frozen=True)
class MLXBuildConfig:
    mlx_repo_root: Path
    build_dir: Path
    python_executable: str = sys.executable
    build_target: str | None = None
    jobs: int | None = None


class MLXBackendAdapter:
    """
    Source-tree backend adapter for an AutoKernel -> MLX port.

    The design assumption is simple:
    - edit a validated target's MLX source files
    - rebuild the MLX checkout
    - benchmark through the public MLX Python API
    """

    def __init__(self, config: MLXBuildConfig) -> None:
        self.config = config

    def enumerate_targets(self) -> Sequence[MLXTarget]:
        return list_targets()

    def resolve_target(self, name: str) -> MLXTarget:
        return get_target(name)

    def build(self) -> None:
        cmd = [
            self.config.python_executable,
            "-m",
            "autokernel_mlx.build_mlx",
            "--repo-root",
            str(self.config.mlx_repo_root),
            "--build-dir",
            str(self.config.build_dir),
        ]
        if self.config.build_target:
            cmd.extend(["--target", self.config.build_target])
        if self.config.jobs is not None:
            cmd.extend(["--jobs", str(self.config.jobs)])
        subprocess.run(cmd, check=True)

    def benchmark_target(
        self,
        target_name: str,
        *,
        model_shape: str,
        phase: str,
        bits: int = 4,
        group_size: int = 128,
        output_tsv: str | None = None,
    ) -> None:
        target = self.resolve_target(target_name)
        op = _default_op_for_target(target)
        cmd = [
            self.config.python_executable,
            "-m",
            "autokernel_mlx.bench_mlx",
            "--op",
            op,
            "--model-shape",
            model_shape,
            "--phase",
            phase,
            "--bits",
            str(bits),
            "--group-size",
            str(group_size),
        ]
        if output_tsv:
            cmd.extend(["--output-tsv", output_tsv])
        subprocess.run(cmd, check=True)

    def target_payload(self, target_name: str) -> dict:
        target = self.resolve_target(target_name)
        return {
            "name": target.name,
            "roi": target.roi,
            "phase": target.phase,
            "public_ops": list(target.public_ops),
            "source_files": list(target.source_files),
            "hot_functions": list(target.hot_functions),
            "dispatch_rules": [
                {"predicate": rule.predicate, "rationale": rule.rationale}
                for rule in target.dispatch_rules
            ],
            "notes": target.notes,
        }


def _default_op_for_target(target: MLXTarget) -> str:
    mapping = {
        "mlx.core.quantized_matmul": "quantized_matmul",
        "mlx.core.fast.scaled_dot_product_attention": "sdpa",
        "mlx.core.fast.rms_norm": "rms_norm",
        "mlx.core.fast.rope": "rope",
        "mlx.core.matmul": "matmul",
        "mlx.core.softmax": "softmax",
        "mlx.core.fast.layer_norm": "layer_norm",
    }
    for public_op in target.public_ops:
        if public_op in mapping:
            return mapping[public_op]
    raise ValueError(f"No benchmark op mapping for target {target.name!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the MLX backend manifest.")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--show-target", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_targets:
        for target in list_targets():
            print(f"{target.name:28}  roi={target.roi:10} phase={target.phase}")
        return
    if args.show_target:
        payload = MLXBackendAdapter(
            MLXBuildConfig(mlx_repo_root=Path("."), build_dir=Path("build"))
        ).target_payload(args.show_target)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("Use --list-targets or --show-target <name>.")


if __name__ == "__main__":
    main()
