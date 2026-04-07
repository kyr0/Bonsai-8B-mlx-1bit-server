"""AutoKernel orchestrator — runs the full experiment loop.

Coordinates: patch → compile → benchmark → record → revert.

Usage:
    # Establish baseline (must be run first)
    python -m autokernel.src.orchestrate baseline

    # Run a single experiment from prepared candidate files
    python -m autokernel.src.orchestrate experiment \\
        --target quantized_dispatch_decode \\
        --candidate-dir candidates/1 \\
        --description "Increased tile size in qmv"

    # Run experiment from a standalone kernel file
    python -m autokernel.src.orchestrate experiment \\
        --target sdpa_vector_decode \\
        --kernel-file mlx/backend/metal/kernels/sdpa_vector.h:path/to/modified.h \\
        --description "Doubled BN block size"
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

from autokernel.src.manifest import get_target
from autokernel.src import manifest_manager
from autokernel.src import patch as patch_mod
from autokernel.src.compile_mlx import compile_mlx, CompileResult
from autokernel.src.run_calibration import (
    run_calibration,
    save_result,
    load_result,
    CalibrationResult,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AUTOKERNEL_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = AUTOKERNEL_DIR / "candidates"
MLX_REPO = AUTOKERNEL_DIR.parent / "mlx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(experiment_id: int) -> Path:
    """Create and return the candidate directory for an experiment."""
    d = CANDIDATES_DIR / str(experiment_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _backup_baseline_kernels(target_name: str, candidate_dir: Path) -> None:
    """Copy the current (baseline) kernel files to the candidate dir for reference."""
    target = get_target(target_name)
    for relpath in target.source_files:
        src = MLX_REPO / relpath
        if src.exists():
            dest = candidate_dir / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))


def _save_experiment_meta(candidate_dir: Path, meta: dict) -> None:
    """Write experiment metadata alongside results."""
    path = candidate_dir / "experiment_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def run_baseline(
    *,
    model: str = "prism-ml/Bonsai-8B-mlx-1bit",
    target: str = "quantized_dispatch_decode",
    mlx_repo: Path = MLX_REPO,
) -> Path:
    """Establish baseline performance (experiment 0).

    Returns path to candidates/0/result.json.
    """
    print("=" * 60)
    print("AutoKernel — Establishing Baseline")
    print("=" * 60)

    # Ensure the MLX tree is clean
    src_files = list(get_target(target).source_files)
    if patch_mod.is_tree_dirty(mlx_repo, src_files):
        print("WARNING: MLX tree is dirty — reverting to HEAD first")
        patch_mod.revert_to_baseline(mlx_repo, src_files)

    candidate_dir = _ensure_dirs(0)

    # Backup baseline kernel files
    _backup_baseline_kernels(target, candidate_dir)

    # Run calibration
    print("\nRunning calibration (baseline) ...\n")
    result = run_calibration(
        model_name=model,
        experiment_id=0,
        target=target,
        description="baseline (unmodified MLX)",
    )

    result_path = candidate_dir / "result.json"
    save_result(result, result_path)

    # Register in manifest
    manifest_manager.add_experiment(result_path, target)

    print(f"\nBaseline tok/s: {result.avg_tok_s}")
    print(f"Result: {result_path}")
    return result_path


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    target: str,
    candidate_dir: Path | None = None,
    kernel_files: dict[str, Path] | None = None,
    description: str = "",
    model: str = "prism-ml/Bonsai-8B-mlx-1bit",
    mlx_repo: Path = MLX_REPO,
    experiment_id: int | None = None,
    skip_compile: bool = False,
) -> Path:
    """Run a single experiment.

    Either *candidate_dir* (containing kernel files at their relative paths)
    or *kernel_files* (mapping relpath → local file) must be provided.

    Always reverts the MLX tree to baseline afterwards, even on failure.

    Returns path to the result.json.
    """
    manifest = manifest_manager.load_manifest()

    if experiment_id is None:
        experiment_id = manifest_manager.get_next_id(manifest)

    exp_dir = _ensure_dirs(experiment_id)

    print("=" * 60)
    print(f"AutoKernel — Experiment {experiment_id}")
    print(f"  Target: {target}")
    print(f"  Description: {description}")
    print("=" * 60)

    target_obj = get_target(target)
    src_files = list(target_obj.source_files)

    # --- Step 1: Prepare candidate files in exp_dir ---
    if kernel_files:
        for relpath, local_file in kernel_files.items():
            dest = exp_dir / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(local_file), str(dest))
    elif candidate_dir and candidate_dir != exp_dir:
        # Copy from provided candidate dir
        for relpath in src_files:
            src = candidate_dir / relpath
            if src.exists():
                dest = exp_dir / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dest))

    # --- Save metadata ---
    _save_experiment_meta(exp_dir, {
        "experiment_id": experiment_id,
        "target": target,
        "description": description,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    result_path = exp_dir / "result.json"

    try:
        # --- Step 2: Create patch ---
        print("\nCreating patch ...")
        patch_path = patch_mod.create_patch(exp_dir, mlx_repo, target)
        if patch_path.stat().st_size == 0:
            print("WARNING: No changes detected — candidate files are identical to baseline")

        # --- Step 3: Apply patch ---
        print("Applying patch to MLX source tree ...")
        patch_mod.apply_patch(patch_path, mlx_repo)

        # --- Step 4: Compile MLX ---
        if not skip_compile:
            print("\nCompiling MLX ...")
            compile_result = compile_mlx(mlx_repo)
            if not compile_result.success:
                raise RuntimeError(
                    f"MLX compilation failed ({compile_result.method}): {compile_result.error}"
                )
            print(f"Compiled via {compile_result.method} in {compile_result.elapsed_s:.1f}s")
        else:
            print("Skipping compilation (--skip-compile)")

        # --- Step 5: Run calibration ---
        print("\nRunning calibration ...")
        baseline_path = CANDIDATES_DIR / "0" / "result.json"
        baseline_data = load_result(baseline_path) if baseline_path.exists() else None

        result = run_calibration(
            model_name=model,
            experiment_id=experiment_id,
            target=target,
            description=description,
            baseline_result=baseline_data,
        )

        save_result(result, result_path)

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        # Write a failure result
        fail_result = {
            "experiment": experiment_id,
            "target": target,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "avg_tok_s": 0,
            "total_tokens": 0,
            "total_time_s": 0,
            "correctness": "FAIL",
            "description": f"FAILED: {description} — {e}",
            "prompts": [],
        }
        with open(result_path, "w") as f:
            json.dump(fail_result, f, indent=2)

    finally:
        # --- Step 6: ALWAYS revert ---
        print("\nReverting MLX source tree to baseline ...")
        patch_mod.revert_to_baseline(mlx_repo, src_files)

        # Re-compile baseline if we compiled the experiment
        if not skip_compile:
            print("Re-compiling baseline MLX ...")
            compile_mlx(mlx_repo)

    # --- Step 7: Update manifest ---
    manifest_manager.add_experiment(result_path, target)

    print(f"\nExperiment {experiment_id} complete.")
    print(manifest_manager.summary())
    return result_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoKernel orchestrator — experiment loop for MLX kernel optimization"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # -- baseline --
    p_base = sub.add_parser("baseline", help="Establish baseline performance")
    p_base.add_argument("--model", default="prism-ml/Bonsai-8B-mlx-1bit")
    p_base.add_argument("--target", default="quantized_dispatch_decode")
    p_base.add_argument("--mlx-repo", default=str(MLX_REPO))

    # -- experiment --
    p_exp = sub.add_parser("experiment", help="Run a single experiment")
    p_exp.add_argument("--target", required=True, help="Optimization target name")
    p_exp.add_argument("--candidate-dir", default=None, help="Dir with candidate kernel files")
    p_exp.add_argument(
        "--kernel-file",
        action="append",
        default=[],
        help="relpath:local_file mapping (repeatable)",
    )
    p_exp.add_argument("--description", default="", help="Experiment description")
    p_exp.add_argument("--model", default="prism-ml/Bonsai-8B-mlx-1bit")
    p_exp.add_argument("--mlx-repo", default=str(MLX_REPO))
    p_exp.add_argument("--experiment-id", type=int, default=None)
    p_exp.add_argument("--skip-compile", action="store_true")

    # -- status --
    sub.add_parser("status", help="Print manifest summary")

    args = parser.parse_args()

    if args.action == "baseline":
        run_baseline(
            model=args.model,
            target=args.target,
            mlx_repo=Path(args.mlx_repo).resolve(),
        )
    elif args.action == "experiment":
        kernel_files = None
        if args.kernel_file:
            kernel_files = {}
            for mapping in args.kernel_file:
                relpath, local_path = mapping.split(":", 1)
                kernel_files[relpath] = Path(local_path).resolve()

        candidate_dir = Path(args.candidate_dir).resolve() if args.candidate_dir else None

        run_experiment(
            target=args.target,
            candidate_dir=candidate_dir,
            kernel_files=kernel_files,
            description=args.description,
            model=args.model,
            mlx_repo=Path(args.mlx_repo).resolve(),
            experiment_id=args.experiment_id,
            skip_compile=args.skip_compile,
        )
    elif args.action == "status":
        print(manifest_manager.summary())


if __name__ == "__main__":
    main()
