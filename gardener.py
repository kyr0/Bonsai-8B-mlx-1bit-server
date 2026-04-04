#!/usr/bin/env python3
"""bonsai-gardener: watches server memory and restarts when idle + bloated."""

import argparse
import logging
import os
import re
import signal
import subprocess
import time

log = logging.getLogger("gardener")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GARDENER_PID_FILE = os.path.join(SCRIPT_DIR, ".gardener.pid")


def get_server_pid(pid_file: str) -> int | None:
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def get_footprint_kb(pid: int) -> int | None:
    """Return physical footprint in KB (includes Metal/GPU on macOS)."""
    try:
        out = subprocess.check_output(
            ["footprint", str(pid)], text=True, stderr=subprocess.DEVNULL,
        )
        m = re.search(r"Footprint:\s+([\d.]+)\s+(KB|MB|GB)", out)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            return int(val * {"KB": 1, "MB": 1024, "GB": 1048576}[unit])
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def seconds_since_modified(path: str) -> float:
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return 999999.0


def fmt_mb(kb: int) -> str:
    return f"{kb // 1024} MB"


def make(target: str) -> None:
    log.info("  running: make %s", target)
    result = subprocess.run(
        ["make", "-C", SCRIPT_DIR, target],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        log.info("  [make] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            log.warning("  [make] %s", line)


def main() -> None:
    parser = argparse.ArgumentParser(description="bonsai-gardener memory watchdog")
    parser.add_argument("--pid-file", default=os.path.join(SCRIPT_DIR, ".bonsai.pid"))
    parser.add_argument("--log-file", default=os.path.join(SCRIPT_DIR, "bonsai.log"))
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (s)")
    parser.add_argument("--idle-timeout", type=int, default=300, help="Idle threshold (s)")
    parser.add_argument("--threshold", type=int, default=20, help="Memory pressure %% over baseline")
    parser.add_argument("--baseline-start", type=int, default=10, help="Baseline window start (s)")
    parser.add_argument("--baseline-end", type=int, default=20, help="Baseline window end (s)")
    args = parser.parse_args()

    logging.basicConfig(
        format="[gardener %(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Write our PID
    with open(GARDENER_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    restart_count = 0

    def cleanup(signum, frame):
        try:
            os.remove(GARDENER_PID_FILE)
        except OSError:
            pass
        log.info("stopped (%d restart(s) performed)", restart_count)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    log.info(
        "starting  (interval=%ds, idle=%ds, threshold=%d%%)",
        args.interval, args.idle_timeout, args.threshold,
    )
    log.info(
        "baseline window: %ds – %ds after start", args.baseline_start, args.baseline_end
    )

    start_time = time.monotonic()
    baseline_rss = 0
    baseline_samples = 0
    baseline_set = False

    while True:
        time.sleep(args.interval)

        # ── get server PID ───────────────────────────────────────
        server_pid = get_server_pid(args.pid_file)
        if server_pid is None:
            baseline_set = False
            baseline_rss = 0
            baseline_samples = 0
            start_time = time.monotonic()
            continue

        current_rss = get_footprint_kb(server_pid)
        if current_rss is None or current_rss == 0:
            continue

        elapsed = time.monotonic() - start_time

        # ── baseline sampling window ─────────────────────────────
        if not baseline_set:
            if args.baseline_start <= elapsed < args.baseline_end:
                baseline_rss += current_rss
                baseline_samples += 1
            elif elapsed >= args.baseline_end:
                if baseline_samples > 0:
                    baseline_rss = baseline_rss // baseline_samples
                    baseline_set = True
                    log.info(
                        "baseline established: %s (%d samples)",
                        fmt_mb(baseline_rss), baseline_samples,
                    )
                else:
                    log.info("no baseline samples collected — retrying window")
                    start_time = time.monotonic()
                    baseline_rss = 0
            continue

        # ── monitoring ───────────────────────────────────────────
        idle_secs = seconds_since_modified(args.log_file)
        pressure = ((current_rss - baseline_rss) * 100) // baseline_rss

        # ── gardening decision ───────────────────────────────────
        if idle_secs >= args.idle_timeout and pressure >= args.threshold:
            log.info(
                "!! memory pressure %d%% >= %d%% and idle %.0fs >= %ds",
                pressure, args.threshold, idle_secs, args.idle_timeout,
            )
            log.info(
                "!! rss=%s  baseline=%s  — restarting server ...",
                fmt_mb(current_rss), fmt_mb(baseline_rss),
            )
            restart_count += 1

            make("_server-stop")
            make("_server-start")

            baseline_set = False
            baseline_rss = 0
            baseline_samples = 0
            start_time = time.monotonic()

            log.info("server restarted — re-establishing baseline")


if __name__ == "__main__":
    main()
