from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BuildRequest:
    repo_root: Path
    build_dir: Path
    build_type: str = "Release"
    cmake_args: tuple[str, ...] = ()
    target: str | None = None
    jobs: int | None = None


def run_command(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    pretty = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {pretty}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def configure(req: BuildRequest) -> None:
    req.build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmake",
        "-S", str(req.repo_root),
        "-B", str(req.build_dir),
        f"-DCMAKE_BUILD_TYPE={req.build_type}",
        *req.cmake_args,
    ]
    run_command(cmd)


def build(req: BuildRequest) -> None:
    cmd = ["cmake", "--build", str(req.build_dir)]
    if req.target:
        cmd.extend(["--target", req.target])
    if req.jobs:
        cmd.extend(["--parallel", str(req.jobs)])
    run_command(cmd)


def rebuild(req: BuildRequest) -> None:
    configure(req)
    build(req)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure / build an MLX checkout.")
    parser.add_argument("--repo-root", required=True, help="Path to the MLX repository root.")
    parser.add_argument("--build-dir", default="build", help="Build directory path.")
    parser.add_argument("--build-type", default="Release", help="CMake build type.")
    parser.add_argument("--target", default=None, help="Optional specific build target.")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel build jobs.")
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="Extra argument to pass to CMake configure step. Repeat as needed.",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Only run the CMake configure step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    req = BuildRequest(
        repo_root=Path(args.repo_root).resolve(),
        build_dir=Path(args.build_dir).resolve(),
        build_type=args.build_type,
        cmake_args=tuple(args.cmake_arg),
        target=args.target,
        jobs=args.jobs,
    )
    if args.configure_only:
        configure(req)
    else:
        rebuild(req)


if __name__ == "__main__":
    main()
