"""AutoKernel — Analysis & visualization of experiment results.

Reads candidates/manifest.json and individual result.json files to produce:
  - Terminal summary of all experiments
  - Research frontier tracking (running max tok/s)
  - Suggestions for next experiments

Usage:
    python -m autokernel.src.analysis
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


CANDIDATES_DIR = Path(__file__).resolve().parent.parent / "candidates"
MANIFEST_PATH = CANDIDATES_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_result(experiment_id: int) -> dict | None:
    path = CANDIDATES_DIR / str(experiment_id) / "result.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(exp: dict) -> str:
    """Classify experiment: 'kept', 'failed', or 'reverted'."""
    if exp.get("correctness") != "PASS":
        return "failed"
    speedup = exp.get("speedup", 0)
    if speedup >= 1.0:
        return "kept"
    return "reverted"


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(manifest: dict) -> None:
    targets = manifest.get("targets", {})

    print()
    print("=" * 60)
    print("  AutoKernel — Session Summary")
    print("=" * 60)

    if not targets:
        print("  No experiments recorded yet.")
        print()
        return

    for tname, tdata in sorted(targets.items()):
        experiments = tdata.get("experiments", [])
        print(f"\n  Target: {tname}")
        print(f"  {'=' * 45}")

        n_total = len(experiments)
        n_kept = sum(1 for e in experiments if classify(e) == "kept")
        n_failed = sum(1 for e in experiments if classify(e) == "failed")
        n_reverted = sum(1 for e in experiments if classify(e) == "reverted")

        # Baseline
        baseline = None
        for e in experiments:
            if e.get("id") == 0:
                baseline = e
                break

        baseline_tok_s = baseline.get("avg_tok_s", 0) if baseline else 0
        best = None
        for e in experiments:
            if e.get("correctness") == "PASS":
                if best is None or e.get("avg_tok_s", 0) > best.get("avg_tok_s", 0):
                    best = e

        if baseline_tok_s:
            print(f"  Baseline:          {baseline_tok_s:.1f} tok/s")
        if best:
            print(f"  Current best:      {best['avg_tok_s']:.1f} tok/s (exp {best['id']}, {best.get('speedup', 0):.3f}x)")
        print(f"  Experiments:       {n_total}")
        print(f"  Kept (faster):     {n_kept}")
        print(f"  Reverted (slower): {n_reverted}")
        print(f"  Failed:            {n_failed}")

        # Research frontier
        if n_total > 1:
            print(f"\n  Experiment history:")
            for exp in experiments:
                tag = classify(exp)
                marker = {"kept": "+", "reverted": "~", "failed": "X"}[tag]
                desc = exp.get("description", "")[:50]
                tok_s = exp.get("avg_tok_s", 0)
                speedup = exp.get("speedup", 0)
                print(f"    [{marker}] exp {exp['id']:>3}: {tok_s:6.1f} tok/s  {speedup:5.3f}x  {desc}")

        # Suggestions
        suggestions = _generate_suggestions(experiments, baseline_tok_s, best)
        if suggestions:
            print(f"\n  Suggestions:")
            for s in suggestions:
                print(f"    - {s}")

    overall = manifest.get("best_overall")
    if overall:
        print(f"\n  {'=' * 45}")
        print(f"  Best overall: experiment {overall['id']} — {overall.get('speedup', 0):.3f}x speedup")

    print()
    print("=" * 60)
    print()


def _generate_suggestions(
    experiments: list[dict],
    baseline_tok_s: float,
    best: dict | None,
) -> list[str]:
    suggestions = []
    n_total = len(experiments)

    if n_total <= 1:
        return ["Run some experiments to generate suggestions."]

    n_failed = sum(1 for e in experiments if classify(e) == "failed")

    # High failure rate
    if n_total > 2 and n_failed / n_total > 0.4:
        suggestions.append(
            f"High failure rate ({n_failed}/{n_total}). "
            "Make smaller, more conservative kernel changes."
        )

    # Speedup analysis
    if best and baseline_tok_s > 0:
        speedup = best.get("speedup", 1.0)
        if speedup < 1.02:
            suggestions.append(
                "No significant speedup yet. Consider: "
                "different tile sizes, loop unrolling, memory access patterns."
            )
        elif speedup < 1.10:
            suggestions.append(
                "Modest gains. Try: SIMD utilization, threadgroup sizing, "
                "reducing shared memory pressure."
            )
        else:
            suggestions.append(
                "Good speedup achieved! Consider: fine-tuning for specific "
                "sequence lengths, profiling with Metal System Trace."
            )

    # Plateau detection
    recent = experiments[-5:] if len(experiments) >= 5 else []
    if len(recent) >= 5 and all(classify(e) in ("reverted", "failed") for e in recent):
        suggestions.append(
            "Last 5 experiments all reverted/failed — possible plateau. "
            "Try a fundamentally different approach."
        )

    if not suggestions:
        suggestions.append("Continue iterating with kernel modifications.")

    return suggestions


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(manifest: dict) -> str:
    """Generate a markdown report."""
    lines = [
        "# AutoKernel Session Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    targets = manifest.get("targets", {})
    for tname, tdata in sorted(targets.items()):
        experiments = tdata.get("experiments", [])
        n_total = len(experiments)
        n_kept = sum(1 for e in experiments if classify(e) == "kept")
        n_failed = sum(1 for e in experiments if classify(e) == "failed")

        lines.append(f"## Target: {tname}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total experiments | {n_total} |")
        lines.append(f"| Kept (faster) | {n_kept} |")
        lines.append(f"| Failed | {n_failed} |")

        baseline = next((e for e in experiments if e.get("id") == 0), None)
        if baseline:
            lines.append(f"| Baseline tok/s | {baseline.get('avg_tok_s', 0):.1f} |")

        best = None
        for e in experiments:
            if e.get("correctness") == "PASS":
                if best is None or e.get("avg_tok_s", 0) > best.get("avg_tok_s", 0):
                    best = e
        if best:
            lines.append(f"| Best tok/s | {best['avg_tok_s']:.1f} |")
            lines.append(f"| Best speedup | {best.get('speedup', 0):.3f}x |")

        lines.append("")

        # Per-experiment table
        if experiments:
            lines.append("### Experiments")
            lines.append("")
            lines.append("| ID | tok/s | Speedup | Status | Description |")
            lines.append("|----|-------|---------|--------|-------------|")
            for exp in experiments:
                status = classify(exp).upper()
                lines.append(
                    f"| {exp['id']} | {exp.get('avg_tok_s', 0):.1f} | "
                    f"{exp.get('speedup', 0):.3f}x | {status} | "
                    f"{exp.get('description', '')[:60]} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = load_manifest()
    if manifest is None:
        print("No manifest found. Run baseline first: python -m autokernel.src.orchestrate baseline")
        return

    targets = manifest.get("targets", {})
    total_experiments = sum(len(t.get("experiments", [])) for t in targets.values())

    if total_experiments == 0:
        print("No experiments recorded yet.")
        return

    print_summary(manifest)

    # Also write report to file
    report = generate_report(manifest)
    report_path = CANDIDATES_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
