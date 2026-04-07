"""MLX compilation wrapper.

Rebuilds the local MLX checkout and installs it into the active venv.
Tries editable install first (fast incremental rebuild); falls back to
full wheel build if Metal kernels do not update.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class CompileResult:
    success: bool
    elapsed_s: float
    method: str  # "editable" | "wheel"
    error: str = ""


def _run(cmd: Sequence[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _pip_exe() -> str:
    """Return the pip executable in the current venv."""
    venv = Path(sys.prefix)
    pip = venv / "bin" / "pip"
    if pip.exists():
        return str(pip)
    return "pip"


def _uv_exe() -> str | None:
    """Return path to uv if available."""
    result = subprocess.run(["which", "uv"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _verify_import() -> bool:
    """Verify that ``import mlx`` succeeds after install."""
    try:
        if "mlx" in sys.modules:
            del sys.modules["mlx"]
        if "mlx.core" in sys.modules:
            del sys.modules["mlx.core"]
        importlib.invalidate_caches()
        result = subprocess.run(
            [sys.executable, "-c", "import mlx.core; print(mlx.core.__file__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def compile_editable(mlx_repo: Path) -> CompileResult:
    """Try ``pip install -e <mlx_repo>`` (editable / incremental)."""
    t0 = time.monotonic()
    pip = _pip_exe()
    result = _run([pip, "install", "-e", str(mlx_repo), "--no-build-isolation"], cwd=mlx_repo, timeout=600)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        return CompileResult(
            success=False,
            elapsed_s=elapsed,
            method="editable",
            error=result.stderr[-2000:] if result.stderr else "unknown error",
        )
    if not _verify_import():
        return CompileResult(
            success=False,
            elapsed_s=elapsed,
            method="editable",
            error="pip install succeeded but `import mlx` failed",
        )
    return CompileResult(success=True, elapsed_s=elapsed, method="editable")


def compile_wheel(mlx_repo: Path) -> CompileResult:
    """Full wheel build via ``uv pip install`` or ``pip install``."""
    t0 = time.monotonic()
    uv = _uv_exe()
    if uv:
        cmd = [uv, "pip", "install", str(mlx_repo)]
    else:
        cmd = [_pip_exe(), "install", str(mlx_repo)]
    result = _run(cmd, cwd=mlx_repo, timeout=900)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        return CompileResult(
            success=False,
            elapsed_s=elapsed,
            method="wheel",
            error=result.stderr[-2000:] if result.stderr else "unknown error",
        )
    if not _verify_import():
        return CompileResult(
            success=False,
            elapsed_s=elapsed,
            method="wheel",
            error="install succeeded but `import mlx` failed",
        )
    return CompileResult(success=True, elapsed_s=elapsed, method="wheel")


def compile_mlx(mlx_repo: Path | str = "../mlx", *, prefer_editable: bool = True) -> CompileResult:
    """Compile and install MLX from source.

    Tries editable install first for speed.  Falls back to full wheel
    build if editable fails.
    """
    mlx_repo = Path(mlx_repo).resolve()
    if not (mlx_repo / "setup.py").exists():
        return CompileResult(
            success=False, elapsed_s=0, method="none",
            error=f"No setup.py found at {mlx_repo}",
        )

    if prefer_editable:
        result = compile_editable(mlx_repo)
        if result.success:
            return result
        print(f"[compile_mlx] editable install failed ({result.error[:200]}), trying wheel build …")

    return compile_wheel(mlx_repo)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compile and install MLX from source")
    parser.add_argument("--mlx-repo", default="../mlx", help="Path to MLX repository root")
    parser.add_argument("--wheel-only", action="store_true", help="Skip editable, go straight to wheel")
    args = parser.parse_args()

    result = compile_mlx(args.mlx_repo, prefer_editable=not args.wheel_only)
    if result.success:
        print(f"[OK] MLX compiled via {result.method} in {result.elapsed_s:.1f}s")
    else:
        print(f"[FAIL] MLX compilation failed ({result.method}): {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
