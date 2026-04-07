"""Idempotent patch / unpatch utilities for MLX kernel experiments.

Manages applying candidate kernel files to the MLX source tree and reverting
back to the git-clean baseline.  All operations check preconditions first so
they are safe to call repeatedly.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def _run(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def source_files_for_target(target_name: str) -> tuple[str, ...]:
    """Return the relative source file paths for a given manifest target."""
    from autokernel.src.manifest import get_target
    return get_target(target_name).source_files


def is_tree_dirty(mlx_repo: Path, files: Sequence[str] | None = None) -> bool:
    """Return True if any of *files* (relative to *mlx_repo*) differ from HEAD."""
    cmd = ["git", "diff", "--quiet", "HEAD", "--"]
    if files:
        cmd.extend(files)
    result = _run(cmd, cwd=mlx_repo, check=False)
    return result.returncode != 0


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------

def revert_to_baseline(mlx_repo: Path, files: Sequence[str] | None = None) -> None:
    """Restore *files* (or the whole tree) to the git HEAD state."""
    cmd = ["git", "checkout", "HEAD", "--"]
    if files:
        cmd.extend(files)
    else:
        cmd.append(".")
    _run(cmd, cwd=mlx_repo)


# ---------------------------------------------------------------------------
# Patch creation
# ---------------------------------------------------------------------------

def create_patch(candidate_dir: Path, mlx_repo: Path, target_name: str) -> Path:
    """Create a unified diff patch for modified kernel files.

    Copies candidate kernel files into the MLX tree, generates a patch via
    ``git diff``, then reverts the tree.  The patch is written to
    ``candidate_dir/experiment.patch``.

    Returns the path to the patch file.
    """
    src_files = source_files_for_target(target_name)
    patch_path = candidate_dir / "experiment.patch"

    # Ensure the tree is clean before we start
    if is_tree_dirty(mlx_repo, list(src_files)):
        revert_to_baseline(mlx_repo, list(src_files))

    # Copy candidate files into the tree
    modified: list[str] = []
    for relpath in src_files:
        candidate_file = candidate_dir / relpath
        if candidate_file.exists():
            dest = mlx_repo / relpath
            shutil.copy2(str(candidate_file), str(dest))
            modified.append(relpath)

    if not modified:
        raise FileNotFoundError(
            f"No candidate kernel files found in {candidate_dir} for target {target_name}. "
            f"Expected one or more of: {src_files}"
        )

    # Generate the diff
    result = _run(["git", "diff", "HEAD", "--"] + modified, cwd=mlx_repo)
    patch_content = result.stdout

    if not patch_content.strip():
        # No actual changes — write an empty patch
        patch_path.write_text("")
        revert_to_baseline(mlx_repo, modified)
        return patch_path

    patch_path.write_text(patch_content)

    # Revert tree so caller controls when to apply
    revert_to_baseline(mlx_repo, modified)
    return patch_path


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def is_patch_applied(patch_path: Path, mlx_repo: Path) -> bool:
    """Check if a patch is already fully applied (reverse-apply dry-run fails
    iff the forward patch is already present)."""
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        return False
    result = _run(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=mlx_repo,
        check=False,
    )
    return result.returncode == 0


def apply_patch(patch_path: Path, mlx_repo: Path) -> None:
    """Apply *patch_path* to the MLX tree idempotently.

    Does nothing if the patch is already applied.  Raises on conflict.
    """
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        return  # nothing to apply

    if is_patch_applied(patch_path, mlx_repo):
        return  # already applied

    # Dry-run first
    _run(["git", "apply", "--check", str(patch_path)], cwd=mlx_repo)
    # Apply for real
    _run(["git", "apply", str(patch_path)], cwd=mlx_repo)


def unapply_patch(patch_path: Path, mlx_repo: Path) -> None:
    """Reverse-apply *patch_path*.  No-op if not currently applied."""
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        return

    if not is_patch_applied(patch_path, mlx_repo):
        return  # not applied — nothing to do

    _run(["git", "apply", "--reverse", str(patch_path)], cwd=mlx_repo)


# ---------------------------------------------------------------------------
# High-level: install candidate files directly (alternative to patch flow)
# ---------------------------------------------------------------------------

def install_candidate_files(candidate_dir: Path, mlx_repo: Path, target_name: str) -> list[str]:
    """Copy candidate kernel files into the MLX tree.  Returns list of
    relative paths that were modified."""
    src_files = source_files_for_target(target_name)
    modified: list[str] = []
    for relpath in src_files:
        candidate_file = candidate_dir / relpath
        if candidate_file.exists():
            dest = mlx_repo / relpath
            shutil.copy2(str(candidate_file), str(dest))
            modified.append(relpath)
    return modified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLX kernel patch management")
    sub = parser.add_subparsers(dest="action", required=True)

    p_status = sub.add_parser("status", help="Check if MLX tree is dirty")
    p_status.add_argument("--mlx-repo", required=True)
    p_status.add_argument("--target", default=None)

    p_revert = sub.add_parser("revert", help="Revert kernel files to baseline")
    p_revert.add_argument("--mlx-repo", required=True)
    p_revert.add_argument("--target", default=None)

    p_apply = sub.add_parser("apply", help="Apply a patch file")
    p_apply.add_argument("--mlx-repo", required=True)
    p_apply.add_argument("patch", help="Path to patch file")

    p_unapply = sub.add_parser("unapply", help="Reverse-apply a patch file")
    p_unapply.add_argument("--mlx-repo", required=True)
    p_unapply.add_argument("patch", help="Path to patch file")

    args = parser.parse_args()
    mlx_repo = Path(args.mlx_repo).resolve()

    if args.action == "status":
        files = list(source_files_for_target(args.target)) if args.target else None
        dirty = is_tree_dirty(mlx_repo, files)
        print(f"dirty={dirty}")
    elif args.action == "revert":
        files = list(source_files_for_target(args.target)) if args.target else None
        revert_to_baseline(mlx_repo, files)
        print("Reverted.")
    elif args.action == "apply":
        apply_patch(Path(args.patch).resolve(), mlx_repo)
        print("Applied.")
    elif args.action == "unapply":
        unapply_patch(Path(args.patch).resolve(), mlx_repo)
        print("Unapplied.")


if __name__ == "__main__":
    main()
