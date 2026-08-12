#!/usr/bin/env python3
"""Sample process and host resource counters for one PDCat experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024
        values[key] = value
    return values


def read_proc_stat(pid: int) -> tuple[str, int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return fields[2], int(fields[13]), int(fields[14])
    except (OSError, IndexError, ValueError):
        return None


def temperatures() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            temperature = int((zone / "temp").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        try:
            kind = (zone / "type").read_text(encoding="utf-8").strip()
        except OSError:
            kind = zone.name
        samples.append(
            {
                "source": "thermal_zone",
                "name": zone.name,
                "type": kind,
                "millicelsius": temperature,
            }
        )
    for path in sorted(
        Path("/sys/class/nvme").glob("nvme*/device/hwmon/hwmon*/temp*_input")
    ):
        try:
            temperature = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        label_path = path.with_name(path.name.replace("_input", "_label"))
        try:
            label = label_path.read_text(encoding="utf-8").strip()
        except OSError:
            label = path.stem
        samples.append(
            {
                "source": "nvme_hwmon",
                "name": str(path),
                "type": label,
                "millicelsius": temperature,
            }
        )
    return samples


def power_rails() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            device_name = (hwmon / "name").read_text(encoding="utf-8").strip()
        except OSError:
            device_name = hwmon.name
        for path in sorted(hwmon.glob("power*_input")):
            try:
                microwatts = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            label_path = path.with_name(path.name.replace("_input", "_label"))
            try:
                label = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                label = path.stem
            samples.append(
                {
                    "source": "hwmon",
                    "device": device_name,
                    "name": path.name,
                    "label": label,
                    "microwatts": microwatts,
                }
            )
    return samples


def sample(
    pid: int,
    run_id: str,
    index: int,
    previous: tuple[float, int, int] | None,
) -> tuple[dict[str, Any] | None, tuple[float, int, int] | None]:
    proc_dir = Path(f"/proc/{pid}")
    stat = read_proc_stat(pid)
    if not proc_dir.exists() or stat is None or stat[0] == "Z":
        return None, previous
    now = time.monotonic()
    _, user_ticks, system_ticks = stat
    cpu_percent = None
    if previous is not None:
        previous_time, previous_user, previous_system = previous
        elapsed = now - previous_time
        if elapsed > 0:
            clock_ticks = os.sysconf("SC_CLK_TCK")
            cpu_percent = (
                100.0
                * ((user_ticks - previous_user) + (system_ticks - previous_system))
                / clock_ticks
                / elapsed
            )

    status = read_key_values(proc_dir / "status")
    process_io = read_key_values(proc_dir / "io")
    meminfo = read_key_values(Path("/proc/meminfo"))
    value = {
        "schema_version": 1,
        "event": "resource_sample",
        "run_id": run_id,
        "sample_index": index,
        "wall_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": pid,
        "process": {
            "rss_bytes": status.get("VmRSS"),
            "rss_peak_bytes": status.get("VmHWM"),
            "virtual_bytes": status.get("VmSize"),
            "threads": status.get("Threads"),
            "cpu_user_ticks": user_ticks,
            "cpu_system_ticks": system_ticks,
            "cpu_percent_one_core": cpu_percent,
            "read_bytes": process_io.get("read_bytes"),
            "write_bytes": process_io.get("write_bytes"),
            "cancelled_write_bytes": process_io.get("cancelled_write_bytes"),
            "logical_read_bytes": process_io.get("rchar"),
            "logical_write_bytes": process_io.get("wchar"),
        },
        "system": {
            "memory_total_bytes": meminfo.get("MemTotal"),
            "memory_available_bytes": meminfo.get("MemAvailable"),
            "swap_free_bytes": meminfo.get("SwapFree"),
            "temperatures": temperatures(),
            "power_rails": power_rails(),
        },
    }
    return value, (now, user_ticks, system_ticks)


def finite(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def summarize_jsonl(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("event") != "resource_sample":
                raise ValueError(f"{path}:{line_number}: invalid resource sample")
            rows.append(value)
    process_rows = [row.get("process", {}) for row in rows]
    cpu = finite([row.get("cpu_percent_one_core") for row in process_rows])
    rss = finite([row.get("rss_bytes") for row in process_rows])
    rss_peak = finite([row.get("rss_peak_bytes") for row in process_rows])
    temperatures_c = finite(
        [
            item.get("millicelsius", 0) / 1000.0
            for row in rows
            for item in row.get("system", {}).get("temperatures", [])
        ]
    )
    power_by_rail: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for item in row.get("system", {}).get("power_rails", []):
            value = item.get("microwatts")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            key = "/".join(
                str(item.get(field, "")) for field in ("device", "label", "name")
            )
            power_by_rail[key].append(float(value) / 1.0e6)

    def delta(key: str) -> int | None:
        values = finite([row.get(key) for row in process_rows])
        return int(values[-1] - values[0]) if len(values) >= 2 else None

    return {
        "schema_version": 1,
        "samples": len(rows),
        "sampling_duration_seconds": (
            (rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]) / 1.0e9
            if len(rows) >= 2
            else 0.0
        ),
        "process": {
            "rss_peak_bytes": int(max(rss_peak or rss, default=0)),
            "rss_mean_bytes": sum(rss) / len(rss) if rss else None,
            "cpu_percent_one_core_mean": sum(cpu) / len(cpu) if cpu else None,
            "cpu_percent_one_core_peak": max(cpu, default=None),
            "read_bytes_delta": delta("read_bytes"),
            "write_bytes_delta": delta("write_bytes"),
            "logical_read_bytes_delta": delta("logical_read_bytes"),
            "logical_write_bytes_delta": delta("logical_write_bytes"),
        },
        "temperature_celsius_peak": max(temperatures_c, default=None),
        "power_watts_by_rail": {
            rail: {
                "samples": len(values),
                "mean": sum(values) / len(values),
                "peak": max(values),
            }
            for rail, values in sorted(power_by_rail.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error("pid and interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    index = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        while True:
            value, previous = sample(args.pid, args.run_id, index, previous)
            if value is None:
                break
            output_file.write(json.dumps(value, separators=(",", ":")) + "\n")
            index += 1
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
