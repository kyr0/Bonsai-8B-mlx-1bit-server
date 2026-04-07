from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

RESULT_COLUMNS = (
    "experiment",
    "tag",
    "kernel_type",
    "throughput_tflops",
    "latency_us",
    "pct_peak",
    "speedup_vs_baseline",
    "correctness",
    "peak_vram_mb",
    "description",
)


@dataclass
class ResultRow:
    experiment: int
    tag: str
    kernel_type: str
    throughput_tflops: float
    latency_us: float
    pct_peak: float
    speedup_vs_baseline: float
    correctness: str
    peak_vram_mb: float
    description: str


def ensure_results_tsv(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
            writer.writeheader()
    return path


def append_rows(path: str | Path, rows: Iterable[ResultRow]) -> None:
    path = ensure_results_tsv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        for row in rows:
            writer.writerow(asdict(row))
