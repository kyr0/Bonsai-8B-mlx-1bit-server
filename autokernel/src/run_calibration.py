"""Run Bonsai model with calibration prompts and measure tok/s.

Uses mlx-lm Python API directly (no server) for isolated, deterministic
benchmarking of inference throughput after kernel changes.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "prism-ml/Bonsai-8B-mlx-1bit"
DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "calibration" / "calibration_bonsai.json"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMP = 0.0          # deterministic for correctness
DEFAULT_REPETITIONS = 1     # repeat each prompt N times, take median


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    id: int
    prompt: str
    response: str
    tokens: int
    time_s: float
    tok_per_sec: float
    baseline_tokens: int | None = None
    correctness: str = "PASS"  # PASS | FAIL | SKIP


@dataclass
class CalibrationResult:
    experiment: int
    target: str
    timestamp: str
    model: str
    avg_tok_s: float
    total_tokens: int
    total_time_s: float
    correctness: str  # PASS | FAIL
    description: str
    prompts: list[dict]


# ---------------------------------------------------------------------------
# Calibration JSON loader
# ---------------------------------------------------------------------------

def load_calibration(path: Path | str) -> list[dict]:
    """Load calibration prompts from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Calibration file must be a non-empty JSON array: {path}")
    return data


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_calibration(
    *,
    model_name: str = DEFAULT_MODEL,
    calibration_path: Path | str = DEFAULT_CALIBRATION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temp: float = DEFAULT_TEMP,
    experiment_id: int = 0,
    target: str = "",
    description: str = "baseline",
    baseline_result: CalibrationResult | dict | None = None,
) -> CalibrationResult:
    """Run all calibration prompts and return structured results."""

    # Import mlx_lm here so import errors surface clearly
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        print(f"ERROR: Cannot import mlx_lm: {e}", file=sys.stderr)
        print("Install with: uv pip install mlx-lm", file=sys.stderr)
        sys.exit(1)

    sampler = make_sampler(temp=temp)

    calibration = load_calibration(calibration_path)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Parse baseline for correctness comparison
    baseline_prompts: dict[int, dict] | None = None
    if baseline_result is not None:
        bl = baseline_result if isinstance(baseline_result, dict) else asdict(baseline_result)
        baseline_prompts = {p["id"]: p for p in bl.get("prompts", [])}

    # Load model
    print(f"Loading model: {model_name} ...")
    t_load_start = time.monotonic()
    model, tokenizer = load(model_name)
    t_load = time.monotonic() - t_load_start
    print(f"Model loaded in {t_load:.1f}s")

    # Run prompts
    prompt_results: list[PromptResult] = []
    total_tokens = 0
    total_gen_time = 0.0
    all_correct = True

    for entry in calibration:
        pid = entry["id"]
        prompt_text = entry["prompt"]

        print(f"  [{pid}] {prompt_text[:60]}...", end="", flush=True)

        try:
            # Apply chat template if tokenizer supports it
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt_text

            t0 = time.monotonic()
            response = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            t1 = time.monotonic()

            elapsed = t1 - t0
            # Count tokens in the response
            resp_tokens = len(tokenizer.encode(response))
            tok_s = resp_tokens / elapsed if elapsed > 0 else 0.0

            # Correctness check
            correctness = "PASS"
            baseline_tok = None

            if baseline_prompts and pid in baseline_prompts:
                bl_entry = baseline_prompts[pid]
                baseline_tok = bl_entry.get("tokens", 0)
                # Check: non-empty response
                if not response.strip():
                    correctness = "FAIL"
                    all_correct = False
                # Check: no dramatic token count divergence (>3x or <0.2x)
                elif baseline_tok and baseline_tok > 5:
                    ratio = resp_tokens / baseline_tok
                    if ratio > 3.0 or ratio < 0.2:
                        correctness = "FAIL"
                        all_correct = False
            elif not response.strip():
                correctness = "FAIL"
                all_correct = False

            pr = PromptResult(
                id=pid,
                prompt=prompt_text,
                response=response[:500],  # truncate for storage
                tokens=resp_tokens,
                time_s=round(elapsed, 3),
                tok_per_sec=round(tok_s, 1),
                baseline_tokens=baseline_tok,
                correctness=correctness,
            )
            prompt_results.append(pr)
            total_tokens += resp_tokens
            total_gen_time += elapsed

            status = "OK" if correctness == "PASS" else "FAIL"
            print(f" {resp_tokens} tok in {elapsed:.2f}s ({tok_s:.1f} tok/s) [{status}]")

        except Exception as e:
            print(f" ERROR: {e}")
            pr = PromptResult(
                id=pid,
                prompt=prompt_text,
                response=f"ERROR: {e}",
                tokens=0,
                time_s=0,
                tok_per_sec=0,
                correctness="FAIL",
            )
            prompt_results.append(pr)
            all_correct = False

    avg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0.0

    result = CalibrationResult(
        experiment=experiment_id,
        target=target,
        timestamp=timestamp,
        model=model_name,
        avg_tok_s=round(avg_tok_s, 2),
        total_tokens=total_tokens,
        total_time_s=round(total_gen_time, 3),
        correctness="PASS" if all_correct else "FAIL",
        description=description,
        prompts=[asdict(p) for p in prompt_results],
    )

    print(f"\n  Average: {avg_tok_s:.1f} tok/s | Total: {total_tokens} tokens in {total_gen_time:.1f}s")
    print(f"  Correctness: {result.correctness}")

    return result


def save_result(result: CalibrationResult, path: Path | str) -> None:
    """Write result to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"Result saved to {path}")


def load_result(path: Path | str) -> dict:
    """Load a result JSON. Returns raw dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Bonsai calibration benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Calibration JSON path")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--experiment-id", type=int, default=0)
    parser.add_argument("--target", default="")
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--output", default=None, help="Output result.json path")
    parser.add_argument("--baseline", default=None, help="Baseline result.json for correctness comparison")
    args = parser.parse_args()

    baseline = None
    if args.baseline:
        baseline = load_result(args.baseline)

    result = run_calibration(
        model_name=args.model,
        calibration_path=Path(args.calibration),
        max_tokens=args.max_tokens,
        temp=args.temp,
        experiment_id=args.experiment_id,
        target=args.target,
        description=args.description,
        baseline_result=baseline,
    )

    output_path = args.output or f"candidates/{args.experiment_id}/result.json"
    save_result(result, Path(output_path))


if __name__ == "__main__":
    main()
