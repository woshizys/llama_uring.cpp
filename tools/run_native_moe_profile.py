#!/usr/bin/env python3
"""Measure native mmap MoE page loading and compute under a cgroup limit."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "build-expert-cache/bin/llama-completion"
DEFAULT_MODEL = Path(
    "/data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/pdcat/runs"
CGROUP_ROOT = Path("/sys/fs/cgroup")
PERF_LINE = re.compile(
    r"^(?:llama_perf_context_print|common_perf_print):\s*"
    r"(?P<label>load time|prompt eval time|eval time|total time)\s*=\s*"
    r"(?P<milliseconds>[0-9]+(?:\.[0-9]+)?) ms"
    r"(?:\s*/\s*(?P<count>[0-9]+)\s*(?P<unit>tokens|runs)"
    r"(?:\s*\(\s*(?P<ms_per>[0-9]+(?:\.[0-9]+)?) ms per token,\s*"
    r"(?P<tokens_per_second>[0-9]+(?:\.[0-9]+)?) tokens per second\))?)?\s*$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_integer(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text and text != "max" else None


def read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def process_rss_bytes(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            return int(fields[1]) * 1024
    return None


def host_memory_available_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024
    return None


def sampled_cgroup_peak_bytes(path: Path) -> int | None:
    peak: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line).get("cgroup_memory_current_bytes")
        except json.JSONDecodeError:
            continue
        if isinstance(value, int):
            peak = value if peak is None else max(peak, value)
    return peak


def parse_performance(path: Path) -> dict[str, Any]:
    labels = {
        "load time": "model_load",
        "prompt eval time": "prompt_eval",
        "eval time": "decode_eval",
        "total time": "total",
    }
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PERF_LINE.match(raw_line.strip())
        if match is None:
            continue
        row: dict[str, Any] = {"milliseconds": float(match.group("milliseconds"))}
        if match.group("count") is not None:
            row["count"] = int(match.group("count"))
            row["count_unit"] = match.group("unit")
        if match.group("ms_per") is not None:
            row["milliseconds_per_token"] = float(match.group("ms_per"))
            row["tokens_per_second"] = float(match.group("tokens_per_second"))
        result[labels[match.group("label")]] = row
    return result


def summarize_trace(path: Path, prompt_eval_ms: float | None) -> dict[str, Any]:
    load_records: list[dict[str, Any]] = []
    compute_records: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("event") == "native_moe_load_profile":
                    load_records.append(record)
                elif record.get("event") == "native_cuda_moe_profile":
                    compute_records.append(record)

    load_ms = sum(float(row["load_wall_us"]) for row in load_records) / 1000.0
    expert_compute_ms = sum(float(row["compute_wall_us"]) for row in compute_records) / 1000.0
    profiled_ms = load_ms + expert_compute_ms
    resident_pages = sum(int(row["resident_pages_before"]) for row in load_records)
    nonresident_pages = sum(int(row["nonresident_pages_before"]) for row in load_records)
    total_pages = resident_pages + nonresident_pages
    token_counts: dict[str, int] = {}
    for row in load_records:
        key = str(row["tokens"])
        token_counts[key] = token_counts.get(key, 0) + 1

    result: dict[str, Any] = {
        "record_count": len(load_records) + len(compute_records),
        "load_record_count": len(load_records),
        "compute_record_count": len(compute_records),
        "load_unique_tensor_count": len({str(row["tensor"]) for row in load_records}),
        "compute_unique_tensor_count": len({str(row["tensor"]) for row in compute_records}),
        "token_count_histogram": token_counts,
        "load_wall_ms": load_ms,
        "expert_compute_wall_ms": expert_compute_ms,
        "profiled_expert_phase_ms": profiled_ms,
        "load_fraction_of_profiled_expert_phase": load_ms / profiled_ms if profiled_ms else None,
        "resident_pages_before": resident_pages,
        "nonresident_pages_before": nonresident_pages,
        "nonresident_fraction_before": nonresident_pages / total_pages if total_pages else None,
        "required_bytes_sum": sum(int(row["required_bytes"]) for row in load_records),
        "block_read_bytes": sum(int(row["load_read_bytes"]) for row in load_records),
        "load_major_faults": sum(int(row["load_major_faults"]) for row in load_records),
        "load_minor_faults": sum(int(row["load_minor_faults"]) for row in load_records),
        "compute_block_read_bytes": sum(int(row["compute_read_bytes"]) for row in compute_records),
        "compute_major_faults": sum(int(row["compute_major_faults"]) for row in compute_records),
        "compute_minor_faults": sum(int(row["compute_minor_faults"]) for row in compute_records),
    }
    if prompt_eval_ms is not None and prompt_eval_ms > 0:
        result["load_fraction_of_prompt_eval"] = load_ms / prompt_eval_ms
        result["expert_compute_fraction_of_prompt_eval"] = expert_compute_ms / prompt_eval_ms
        result["other_prompt_time_ms"] = prompt_eval_ms - profiled_ms
        result["other_prompt_fraction"] = (prompt_eval_ms - profiled_ms) / prompt_eval_ms
    return result


def make_prompt(token_count: int) -> str:
    # DeepSeek-V2-Lite tokenizes BOS + each leading-space " hello" as one token.
    if token_count < 2:
        raise ValueError("prompt token count must be at least 2")
    return " hello" * (token_count - 1)


def evict_model_cache(model: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    descriptor = os.open(model, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def prefetch_oracle_experts(
    model: Path,
    trace_path: Path,
    prompt_tokens: int,
    workers: int = 8,
    cpu: int | None = None,
) -> dict[str, Any]:
    ranges: list[tuple[int, int]] = []
    source_records = 0
    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("event") != "native_moe_load_profile":
            continue
        if record.get("tokens") != prompt_tokens:
            raise RuntimeError(
                f"oracle token mismatch: expected {prompt_tokens}, got {record.get('tokens')}"
            )
        tensor_offset = record.get("tensor_file_offset")
        expert_size = record.get("expert_bytes")
        expert_count = record.get("expert_count")
        expert_ids = record.get("expert_ids")
        if (
            not isinstance(tensor_offset, int)
            or not isinstance(expert_size, int)
            or not isinstance(expert_count, int)
            or not isinstance(expert_ids, list)
            or not expert_ids
            or tensor_offset < 0
            or expert_size <= 0
            or expert_count <= 0
            or not all(isinstance(value, int) and 0 <= value < expert_count for value in expert_ids)
        ):
            raise RuntimeError(f"invalid oracle record: {record.get('tensor')}")
        source_records += 1
        ordered = sorted(set(expert_ids))
        first = ordered[0]
        last = first
        for expert in ordered[1:]:
            if expert == last + 1:
                last = expert
                continue
            offset = tensor_offset + first * expert_size
            length = (last - first + 1) * expert_size
            if last < expert_count - 1:
                length += min(expert_size, 512)
            ranges.append((offset, length))
            first = expert
            last = expert
        offset = tensor_offset + first * expert_size
        length = (last - first + 1) * expert_size
        if last < expert_count - 1:
            length += min(expert_size, 512)
        ranges.append((offset, length))

    if source_records == 0 or not ranges:
        raise RuntimeError(f"oracle trace contains no usable load records: {trace_path}")

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for offset, length in ranges:
        end = offset + length
        if merged and offset <= merged[-1][0] + merged[-1][1]:
            previous_offset, previous_length = merged[-1]
            merged[-1] = (
                previous_offset,
                max(previous_offset + previous_length, end) - previous_offset,
            )
        else:
            merged.append((offset, length))

    if workers < 1:
        raise ValueError("oracle prefetch workers must be positive")
    original_affinity: set[int] | None = None
    if cpu is not None:
        original_affinity = os.sched_getaffinity(0)
        if cpu not in original_affinity:
            raise RuntimeError(
                f"oracle prefetch CPU {cpu} is outside the current affinity "
                f"{sorted(original_affinity)}"
            )
        os.sched_setaffinity(0, {cpu})

    started = time.monotonic()
    descriptor = -1

    def read_ranges(worker: int) -> int:
        completed_total = 0
        for range_index in range(worker, len(merged), workers):
            offset, length = merged[range_index]
            completed = 0
            while completed < length:
                request = min(1024 * 1024, length - completed)
                block = os.pread(descriptor, request, offset + completed)
                if not block:
                    raise RuntimeError(
                        f"short oracle prefetch at offset {offset + completed}"
                    )
                completed += len(block)
                completed_total += len(block)
        return completed_total

    try:
        descriptor = os.open(model, os.O_RDONLY)
        if workers == 1:
            bytes_read = read_ranges(0)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                bytes_read = sum(executor.map(read_ranges, range(workers)))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if original_affinity is not None:
            os.sched_setaffinity(0, original_affinity)
    wall_seconds = time.monotonic() - started
    return {
        "source_trace": str(trace_path),
        "source_load_records": source_records,
        "range_count_before_merge": len(ranges),
        "range_count": len(merged),
        "bytes_requested": sum(length for _, length in merged),
        "bytes_read": bytes_read,
        "workers": workers,
        "cpu": cpu,
        "wall_seconds": wall_seconds,
        "throughput_megabytes_per_second": bytes_read / wall_seconds / 1_000_000,
    }


def enable_memory_controller() -> bool:
    controllers = set(
        (CGROUP_ROOT / "cgroup.controllers").read_text(encoding="utf-8").split()
    )
    subtree_control_path = CGROUP_ROOT / "cgroup.subtree_control"
    subtree_control = set(subtree_control_path.read_text(encoding="utf-8").split())
    if "memory" in subtree_control:
        return False
    if "memory" not in controllers:
        raise RuntimeError("cgroup v2 memory controller is unavailable")
    subtree_control_path.write_text("+memory", encoding="utf-8")
    return True


def monitor_process(
    process: subprocess.Popen[str],
    cgroup: Path,
    output: Path,
    stop: threading.Event,
    min_host_memory_available_bytes: int,
) -> None:
    started = time.monotonic()
    next_report = 30.0
    with output.open("w", encoding="utf-8") as destination:
        while not stop.is_set():
            elapsed = time.monotonic() - started
            current = read_integer(cgroup / "memory.current")
            sample = {
                "elapsed_seconds": elapsed,
                "cgroup_memory_current_bytes": current,
                "process_rss_bytes": process_rss_bytes(process.pid),
                "host_memory_available_bytes": host_memory_available_bytes(),
            }
            available = sample["host_memory_available_bytes"]
            guard_triggered = (
                min_host_memory_available_bytes > 0
                and isinstance(available, int)
                and available < min_host_memory_available_bytes
            )
            sample["host_memory_guard_triggered"] = guard_triggered
            destination.write(json.dumps(sample, sort_keys=True) + "\n")
            destination.flush()
            if guard_triggered and process.poll() is None:
                print(
                    "host-memory guard triggered: "
                    f"available={available / 1024**3:.2f}GiB "
                    f"minimum={min_host_memory_available_bytes / 1024**3:.2f}GiB; "
                    "terminating llama",
                    flush=True,
                )
                process.terminate()
                return
            if elapsed >= next_report:
                current_gib = current / (1024**3) if current is not None else float("nan")
                print(
                    f"progress pid={process.pid} elapsed={elapsed:.0f}s "
                    f"cgroup_memory={current_gib:.2f}GiB",
                    flush=True,
                )
                next_report += 30.0
            stop.wait(0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gib", type=int, choices=(4, 8), required=True)
    parser.add_argument("--prompt-tokens", type=int, choices=(128, 256), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--current-cgroup",
        action="store_true",
        help="use and verify the current container cgroup instead of creating a child",
    )
    parser.add_argument("--gdb", action="store_true", help="run llama under batch GDB")
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="run a cold native end-to-end control without profiling synchronization",
    )
    parser.add_argument(
        "--gpu-layers",
        default="all",
        help="value passed to llama --gpu-layers (default: all)",
    )
    parser.add_argument(
        "--fit",
        choices=("on", "off"),
        default="off",
        help="value passed to llama --fit (default: off)",
    )
    parser.add_argument(
        "--fit-target-mib",
        type=int,
        help="minimum free device memory requested through --fit-target",
    )
    parser.add_argument(
        "--cpu-moe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep routed MoE weights on CPU (default: enabled)",
    )
    parser.add_argument(
        "--expert-cache-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass --expert-cache-capacity 0 (disable for upstream llama.cpp)",
    )
    parser.add_argument(
        "--min-host-memory-available-gib",
        type=float,
        default=0.0,
        help="terminate llama if host MemAvailable falls below this value",
    )
    parser.add_argument(
        "--oracle-trace",
        type=Path,
        help="prefetch exact GGUF expert ranges recorded by an earlier identical input",
    )
    parser.add_argument(
        "--oracle-workers",
        type=int,
        choices=range(1, 9),
        default=8,
        help="number of oracle prefetch readers (default: 8)",
    )
    parser.add_argument(
        "--oracle-cpu",
        type=int,
        help="temporarily pin oracle prefetch to one CPU; inference affinity is restored",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    model = args.model.resolve()
    oracle_trace = args.oracle_trace.resolve() if args.oracle_trace else None
    run_dir = (args.output_root / args.run_id).resolve()
    if run_dir.exists():
        raise RuntimeError(f"output directory already exists: {run_dir}")
    if not binary.is_file() or not model.is_file():
        raise RuntimeError(f"missing binary or model: {binary}, {model}")
    if oracle_trace is not None and not oracle_trace.is_file():
        raise RuntimeError(f"missing oracle trace: {oracle_trace}")
    if args.fit_target_mib is not None and args.fit_target_mib <= 0:
        raise RuntimeError("--fit-target-mib must be positive")
    if args.min_host_memory_available_gib < 0:
        raise RuntimeError("--min-host-memory-available-gib must be non-negative")
    if os.geteuid() != 0:
        raise RuntimeError("root is required to create the experiment cgroup")

    run_dir.mkdir(parents=True)
    trace_path = run_dir / "native_moe_profile.jsonl"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    samples_path = run_dir / "cgroup_samples.jsonl"
    memory_bytes = args.memory_gib * 1024**3
    if args.current_cgroup:
        cgroup = CGROUP_ROOT
        memory_controller_enabled = False
        actual_memory_limit = read_integer(cgroup / "memory.max")
        actual_swap_limit = read_integer(cgroup / "memory.swap.max")
        if actual_memory_limit != memory_bytes or actual_swap_limit != 0:
            raise RuntimeError(
                "current cgroup limits do not match the request: "
                f"memory.max={actual_memory_limit}, memory.swap.max={actual_swap_limit}"
            )
    else:
        cgroup = CGROUP_ROOT / (
            f"pdcat-native-{os.getpid()}-{args.memory_gib}g-p{args.prompt_tokens}"
        )
        memory_controller_enabled = enable_memory_controller()
        try:
            cgroup.mkdir()
            (cgroup / "memory.max").write_text(str(memory_bytes), encoding="utf-8")
            (cgroup / "memory.swap.max").write_text("0", encoding="utf-8")
            (cgroup / "memory.oom.group").write_text("1", encoding="utf-8")
        except BaseException:
            if cgroup.exists():
                cgroup.rmdir()
            if memory_controller_enabled:
                (CGROUP_ROOT / "cgroup.subtree_control").write_text(
                    "-memory", encoding="utf-8"
                )
            raise

    prompt = make_prompt(args.prompt_tokens)
    command = [
        str(binary),
        "--model", str(model),
        "--prompt", prompt,
        "--n-predict", "1",
        "--seed", "42",
        "--temp", "0",
        "--ctx-size", "4096",
        "--batch-size", "512",
        "--ubatch-size", "512",
        "--threads", "8",
        "--no-repack",
        "--no-warmup",
        "--no-display-prompt",
        "--no-conversation",
        "--gpu-layers", args.gpu_layers,
        "--fit", args.fit,
        "--mmap",
        "--no-direct-io",
    ]
    if args.expert_cache_flag:
        command.extend(["--expert-cache-capacity", "0"])
    if args.cpu_moe:
        command.append("--cpu-moe")
    if args.fit_target_mib is not None:
        command.extend(["--fit-target", str(args.fit_target_mib)])
    if args.gdb:
        command = [
            "gdb",
            "--batch",
            "-ex", "run",
            "-ex", "thread apply all bt",
            "--args",
            *command,
        ]
    environment = os.environ.copy()
    if not args.no_trace:
        environment["LLAMA_NATIVE_MOE_PROFILE_JSONL"] = str(trace_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "status": "running",
        "memory_limit_bytes": memory_bytes,
        "swap_limit_bytes": 0,
        "requested_prompt_tokens": args.prompt_tokens,
        "prompt_construction": "BOS + (' hello' repeated prompt_tokens - 1)",
        "binary": str(binary),
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "command": command,
        "trace": str(trace_path),
        "trace_enabled": not args.no_trace,
        "cgroup": str(cgroup),
        "current_cgroup": args.current_cgroup,
        "oracle_trace": str(oracle_trace) if oracle_trace is not None else None,
        "oracle_workers": args.oracle_workers,
        "oracle_cpu": args.oracle_cpu,
        "gpu_layers": args.gpu_layers,
        "fit": args.fit,
        "fit_target_mib": args.fit_target_mib,
        "cpu_moe": args.cpu_moe,
        "expert_cache_flag": args.expert_cache_flag,
        "min_host_memory_available_gib": args.min_host_memory_available_gib,
    }
    write_json(run_dir / "run_manifest.json", manifest)

    process: subprocess.Popen[str] | None = None
    stop = threading.Event()
    monitor: threading.Thread | None = None
    started = time.monotonic()
    try:
        evict_model_cache(model)
        manifest["cache_eviction"] = "POSIX_FADV_DONTNEED completed"
        if oracle_trace is not None:
            print(f"prefetching oracle ranges from {oracle_trace}", flush=True)
            manifest["oracle_prefetch"] = prefetch_oracle_experts(
                model,
                oracle_trace,
                args.prompt_tokens,
                workers=args.oracle_workers,
                cpu=args.oracle_cpu,
            )
            write_json(run_dir / "run_manifest.json", manifest)
            print(
                "oracle prefetch completed "
                f"bytes={manifest['oracle_prefetch']['bytes_read']} "
                f"wall={manifest['oracle_prefetch']['wall_seconds']:.3f}s",
                flush=True,
            )

        def enter_cgroup() -> None:
            (cgroup / "cgroup.procs").write_text(str(os.getpid()), encoding="utf-8")

        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=environment,
                preexec_fn=None if args.current_cgroup else enter_cgroup,
            )
            print(
                f"started run_id={args.run_id} pid={process.pid} "
                f"limit={args.memory_gib}GiB prompt={args.prompt_tokens}",
                flush=True,
            )
            monitor = threading.Thread(
                target=monitor_process,
                args=(
                    process,
                    cgroup,
                    samples_path,
                    stop,
                    int(args.min_host_memory_available_gib * 1024**3),
                ),
                daemon=True,
            )
            monitor.start()
            return_code = process.wait()

        manifest["return_code"] = return_code
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["completed_at"] = utc_now()
        manifest["status"] = "completed" if return_code == 0 else "failed"
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        manifest["cgroup_memory_peak_bytes"] = read_integer(cgroup / "memory.peak")
        manifest["cgroup_memory_sample_peak_bytes"] = sampled_cgroup_peak_bytes(
            samples_path
        )
        manifest["cgroup_memory_events"] = read_key_values(cgroup / "memory.events")
        manifest["cgroup_memory_stat"] = read_key_values(cgroup / "memory.stat")
        try:
            performance = parse_performance(stderr_path)
            manifest["llama_performance"] = performance
            prompt_eval = performance.get("prompt_eval", {}).get("milliseconds")
            manifest["native_moe_profile_summary"] = summarize_trace(
                trace_path,
                float(prompt_eval) if prompt_eval is not None else None,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            manifest["summary_error"] = str(error)
        write_json(run_dir / "run_manifest.json", manifest)
        if not args.current_cgroup:
            try:
                cgroup.rmdir()
            except OSError as error:
                manifest["cgroup_cleanup_error"] = str(error)
                write_json(run_dir / "run_manifest.json", manifest)
            else:
                if memory_controller_enabled:
                    try:
                        (CGROUP_ROOT / "cgroup.subtree_control").write_text(
                            "-memory", encoding="utf-8"
                        )
                    except OSError as error:
                        manifest["controller_cleanup_error"] = str(error)
                        write_json(run_dir / "run_manifest.json", manifest)

    print(
        f"finished run_id={args.run_id} status={manifest['status']} "
        f"elapsed={manifest.get('elapsed_seconds', 0):.1f}s",
        flush=True,
    )
    return int(manifest.get("return_code", 1) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
