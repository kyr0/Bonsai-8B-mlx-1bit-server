"""Manage candidates/manifest.json — the index of all AutoKernel experiments.

The manifest tracks every experiment per optimization target, sorted by
speedup (best-first) so the orchestrator and analysis tools can quickly
find the current best candidate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CANDIDATES_DIR = Path(__file__).resolve().parent.parent / "candidates"
MANIFEST_PATH = CANDIDATES_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _empty_manifest() -> dict:
    return {"targets": {}, "best_overall": None, "next_id": 1}


def _sort_experiments(experiments: list[dict]) -> list[dict]:
    """Sort experiments: PASS first, then by speedup descending."""
    def key(e: dict) -> tuple:
        correct = 0 if e.get("correctness") == "PASS" else 1
        speedup = -(e.get("speedup", 0.0))
        return (correct, speedup)
    return sorted(experiments, key=key)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    """Load manifest from disk, or return an empty one if missing."""
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return _empty_manifest()


def save_manifest(manifest: dict, path: Path | str = MANIFEST_PATH) -> None:
    """Write manifest to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_next_id(manifest: dict | None = None) -> int:
    """Return the next experiment ID."""
    if manifest is None:
        manifest = load_manifest()
    return manifest.get("next_id", 1)


def add_experiment(
    result_path: Path | str,
    target_name: str,
    *,
    manifest: dict | None = None,
    manifest_path: Path | str = MANIFEST_PATH,
) -> dict:
    """Add an experiment from its result.json to the manifest.

    - Loads result.json
    - Creates the target entry if not present
    - Appends the experiment
    - Re-sorts experiments (best first)
    - Updates best_overall and next_id
    - Saves manifest
    - Returns the updated manifest
    """
    result_path = Path(result_path)
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    if manifest is None:
        manifest = load_manifest(manifest_path)

    exp_id = result.get("experiment", 0)
    avg_tok_s = result.get("avg_tok_s", 0.0)
    correctness = result.get("correctness", "FAIL")
    description = result.get("description", "")

    # Compute speedup relative to baseline (experiment 0)
    baseline_tok_s = _get_baseline_tok_s(manifest, target_name)
    if baseline_tok_s and baseline_tok_s > 0:
        speedup = round(avg_tok_s / baseline_tok_s, 4)
    else:
        speedup = 1.0

    entry = {
        "id": exp_id,
        "speedup": speedup,
        "avg_tok_s": avg_tok_s,
        "correctness": correctness,
        "description": description,
        "result_path": str(result_path),
    }

    # Ensure target exists in manifest
    targets = manifest.setdefault("targets", {})
    target_data = targets.setdefault(target_name, {"experiments": []})
    experiments = target_data["experiments"]

    # Avoid duplicates
    experiments = [e for e in experiments if e.get("id") != exp_id]
    experiments.append(entry)
    target_data["experiments"] = _sort_experiments(experiments)

    # Update best_overall
    manifest["best_overall"] = _compute_best_overall(manifest)

    # Update next_id
    all_ids = _all_experiment_ids(manifest)
    manifest["next_id"] = max(all_ids) + 1 if all_ids else 1

    save_manifest(manifest, manifest_path)
    return manifest


def _get_baseline_tok_s(manifest: dict, target_name: str) -> float | None:
    """Get the baseline (experiment 0) tok/s for the target."""
    targets = manifest.get("targets", {})
    target_data = targets.get(target_name, {})
    for exp in target_data.get("experiments", []):
        if exp.get("id") == 0:
            return exp.get("avg_tok_s")
    # Fall back to global baseline
    for tname, tdata in targets.items():
        for exp in tdata.get("experiments", []):
            if exp.get("id") == 0:
                return exp.get("avg_tok_s")
    return None


def _compute_best_overall(manifest: dict) -> dict | None:
    """Find the best experiment across all targets (highest speedup, correct)."""
    best = None
    for target_data in manifest.get("targets", {}).values():
        for exp in target_data.get("experiments", []):
            if exp.get("correctness") != "PASS":
                continue
            if best is None or exp.get("speedup", 0) > best.get("speedup", 0):
                best = exp
    return best


def _all_experiment_ids(manifest: dict) -> list[int]:
    """Collect all experiment IDs across all targets."""
    ids = set()
    for target_data in manifest.get("targets", {}).values():
        for exp in target_data.get("experiments", []):
            eid = exp.get("id")
            if eid is not None:
                ids.add(eid)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_best(target_name: str, *, manifest: dict | None = None) -> dict | None:
    """Return the best experiment for a given target (or None)."""
    if manifest is None:
        manifest = load_manifest()
    target_data = manifest.get("targets", {}).get(target_name, {})
    experiments = target_data.get("experiments", [])
    for exp in experiments:
        if exp.get("correctness") == "PASS":
            return exp
    return None


def summary(manifest: dict | None = None) -> str:
    """Return a brief text summary of the manifest."""
    if manifest is None:
        manifest = load_manifest()

    lines = ["AutoKernel Experiment Manifest", "=" * 40]
    targets = manifest.get("targets", {})
    if not targets:
        lines.append("No experiments recorded yet.")
        return "\n".join(lines)

    for tname, tdata in sorted(targets.items()):
        exps = tdata.get("experiments", [])
        n_pass = sum(1 for e in exps if e.get("correctness") == "PASS")
        n_fail = sum(1 for e in exps if e.get("correctness") != "PASS")
        best = get_best(tname, manifest=manifest)
        best_str = f"{best['speedup']:.3f}x ({best['avg_tok_s']:.1f} tok/s)" if best else "none"
        lines.append(f"\n  {tname}:")
        lines.append(f"    experiments: {len(exps)} ({n_pass} PASS, {n_fail} FAIL)")
        lines.append(f"    best: {best_str}")

    overall = manifest.get("best_overall")
    if overall:
        lines.append(f"\n  Best overall: exp {overall['id']} — {overall['speedup']:.3f}x")

    lines.append(f"  Next experiment ID: {manifest.get('next_id', 1)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AutoKernel manifest manager")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Print manifest summary")

    p_add = sub.add_parser("add", help="Add experiment result to manifest")
    p_add.add_argument("result_json", help="Path to result.json")
    p_add.add_argument("--target", required=True, help="Target name")

    p_best = sub.add_parser("best", help="Show best experiment for a target")
    p_best.add_argument("--target", required=True)

    p_next = sub.add_parser("next-id", help="Print the next experiment ID")

    args = parser.parse_args()

    if args.action == "status":
        print(summary())
    elif args.action == "add":
        add_experiment(args.result_json, args.target)
        print(summary())
    elif args.action == "best":
        b = get_best(args.target)
        if b:
            print(json.dumps(b, indent=2))
        else:
            print(f"No passing experiments for target {args.target}")
    elif args.action == "next-id":
        print(get_next_id())


if __name__ == "__main__":
    main()
