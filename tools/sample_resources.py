#!/usr/bin/env python3
"""Sample CPU, RAM and MPS memory while a MOTH-50 training run is active.

macOS keeps no CPU/RAM history, so this writes one JSON line per sample; the
dashboard generator (`tools/resource_dashboard.py`) reads it. MPS memory and the
current step come from the trainer's ``train.log.jsonl``, which is the only
resource the trainer itself records.

    python tools/sample_resources.py --out runs/E-011/resource-samples.jsonl

The training PID is rediscovered on every sample, so a sequential seed-0 →
seed-1 shell loop is covered without restarting the sampler.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PAGE_SIZE = 16384  # Apple silicon, confirmed from `vm_stat`
LOG_GLOB = "runs/E-011/*/train.log.jsonl"


def run(*command: str) -> str:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=10).stdout


def training_pid() -> int | None:
    for line in run("ps", "-axo", "pid=,command=").splitlines():
        pid, _, command = line.strip().partition(" ")
        executable = command.split(maxsplit=1)[0] if command else ""
        if "-m training.train" in command and "python" in executable:
            return int(pid)
    return None


def latest_step() -> tuple[int | None, float | None, str | None]:
    """Newest training-log line across E-011 seed directories."""
    newest = None
    for path in Path(".").glob(LOG_GLOB):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        candidate = (path.stat().st_mtime, path, lines[-1])
        if newest is None or candidate[0] > newest[0]:
            newest = candidate
    if newest is None:
        return None, None, None
    path = newest[1]
    try:
        row = json.loads(newest[2])
    except json.JSONDecodeError:
        return None, None, str(path)
    return row.get("step"), row.get("memory_gb"), str(path)


def cpu_and_load() -> dict:
    # Two samples one second apart; the second is the current second, not the
    # uptime average (`-c` is the repeat count, `-w` the wait).
    output = run("iostat", "-c", "2", "-w", "1")
    fields = None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 9:
            fields = parts
    if fields is None:
        return {}
    return {
        "sys_user_pct": float(fields[3]),
        "sys_system_pct": float(fields[4]),
        "sys_idle_pct": float(fields[5]),
        "load_1m": float(fields[6]),
        "load_5m": float(fields[7]),
        "load_15m": float(fields[8]),
    }


def memory() -> dict:
    stats = {}
    for line in run("vm_stat").splitlines():
        match = re.match(r"(.+?):\s+(\d+)\.", line)
        if match:
            stats[match.group(1)] = int(match.group(2)) * PAGE_SIZE
    used = stats.get("Pages active", 0) + stats.get("Pages wired down", 0)
    used += stats.get("Pages occupied by compressor", 0)
    return {
        "mem_used_bytes": used,
        "mem_active_bytes": stats.get("Pages active", 0),
        "mem_wired_bytes": stats.get("Pages wired down", 0),
        "mem_compressed_bytes": stats.get("Pages occupied by compressor", 0),
        "mem_free_bytes": stats.get("Pages free", 0),
    }


def swap() -> dict:
    match = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", run("sysctl", "-n", "vm.swapusage"))
    if not match:
        return {}
    return {
        "swap_total_bytes": int(float(match.group(1)) * 1e6),
        "swap_used_bytes": int(float(match.group(2)) * 1e6),
    }


def process(pid: int | None) -> dict:
    if pid is None:
        return {"pid": None, "proc_cpu_pct": None, "proc_rss_bytes": None}
    output = run("ps", "-p", str(pid), "-o", "%cpu=,rss=").strip()
    if not output:
        return {"pid": pid, "proc_cpu_pct": None, "proc_rss_bytes": None}
    cpu, rss = output.split()
    return {"pid": pid, "proc_cpu_pct": float(cpu), "proc_rss_bytes": int(rss) * 1024}


def thermal() -> dict:
    match = re.search(r"CPU_Speed_Limit = (\d+)", run("pmset", "-g", "therm"))
    return {"cpu_speed_limit": int(match.group(1)) if match else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sample_resources.py", description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = run until stopped")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    missing = 0
    with args.out.open("a", encoding="utf-8") as out:
        while True:
            pid = training_pid()
            step, mps_gb, log = latest_step()
            sample = {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "epoch": time.time(),
                "step": step,
                "mps_gb": mps_gb,
                "log": log,
                **process(pid),
                **cpu_and_load(),
                **memory(),
                **swap(),
                **thermal(),
            }
            out.write(json.dumps(sample) + "\n")
            out.flush()
            samples += 1
            if args.max_samples and samples >= args.max_samples:
                return 0
            missing = missing + 1 if pid is None else 0
            if missing >= 120:  # 30 min with no trainer: both seeds are done
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
