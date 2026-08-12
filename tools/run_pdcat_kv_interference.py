#!/usr/bin/env python3
"""Measure PDCat decode latency while a second llama-server slot is saved via P4."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pdcat_model import inspect_from_run_config
from monitor_pdcat_process import summarize_jsonl as summarize_resource_jsonl
from run_pdcat_experiment import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    compact_inspection,
    expand_environment,
    inspect_expert_pack,
    read_json,
    summarize_jsonl,
    utc_now,
    validate_config,
    write_json,
)


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[dict[str, Any], float]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    elapsed = time.monotonic() - start
    if not isinstance(result, dict):
        raise RuntimeError(f"{url}: expected a JSON object")
    return result, elapsed


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with {process.returncode}")
        try:
            request_json(base_url + "/health", timeout=2.0)
            return
        except (OSError, ValueError, urllib.error.HTTPError):
            time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {timeout} seconds")


def completion(base_url: str, prompt: str, n_predict: int, slot: int) -> dict[str, Any]:
    response, elapsed = request_json(
        base_url + "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "seed": 42,
            "id_slot": slot,
            "cache_prompt": False,
        },
    )
    return {"http_elapsed_seconds": elapsed, "response": response}


def erase_slot(base_url: str, slot: int) -> dict[str, Any]:
    response, elapsed = request_json(
        f"{base_url}/slots/{slot}?action=erase", {}, timeout=120.0
    )
    return {"http_elapsed_seconds": elapsed, "response": response}


def numeric_timing(record: dict[str, Any], key: str) -> float | None:
    response = record.get("response", {})
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    value = timings.get(key) if isinstance(timings, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def mean_timing(records: list[dict[str, Any]], key: str) -> float:
    values = [numeric_timing(record, key) for record in records]
    if any(value is None for value in values):
        raise RuntimeError(f"server response is missing numeric timing {key!r}")
    numeric_values = [value for value in values if value is not None]
    return sum(numeric_values) / len(numeric_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "build-expert-cache/bin/llama-server")
    parser.add_argument("--profile", default="cache_mixed")
    parser.add_argument("--run-id", default="deepseek-v2-lite-v03-kv-p4-interference")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments/pdcat/runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--populate-repeat", type=int, default=24)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise RuntimeError("port must be in [1, 65535]")
    if args.n_predict <= 0 or args.populate_repeat <= 0 or args.startup_timeout <= 0:
        raise RuntimeError("n-predict, populate-repeat, and startup-timeout must be positive")

    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    if args.profile not in config["profiles"]:
        raise RuntimeError(f"unknown profile {args.profile!r}")
    inspection = inspect_from_run_config(config, config_path, hash_model=False)
    pack = inspect_expert_pack(config, args.profile, verify_pack_hash=False)
    if pack is None:
        raise RuntimeError("P4 interference run requires an expert-pack cache profile")

    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-server binary is missing or not executable: {binary}")
    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    slot_dir = run_dir / "slots"
    slot_dir.mkdir()
    trace_path = run_dir / "expert_io.jsonl"
    resource_path = run_dir / "resource_samples.jsonl"
    monitor_stderr_path = run_dir / "resource_monitor.stderr.log"

    base_args = [
        argument
        for argument in config["command"].get("base_args", [])
        if argument != "--no-display-prompt"
    ]
    profile = config["profiles"][args.profile]
    command = [
        str(binary),
        "--model",
        inspection["model"]["path"],
        *base_args,
        *profile.get("args", []),
        "--expert-pack-manifest",
        pack["manifest_path"],
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--parallel",
        "2",
        "--slot-save-path",
        str(slot_dir),
        "--pdcat-kv-p4",
        "--metrics",
    ]
    environment = os.environ.copy()
    environment.update(expand_environment(config["command"].get("environment", {})))
    environment.update(expand_environment(profile.get("environment", {})))
    environment.update(
        {
            "PDCAT_RUN_ID": args.run_id,
            "PDCAT_REQUEST_ID": "server",
            "PDCAT_IO_TRACE_PATH": str(trace_path),
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "starting",
        "created_at_utc": utc_now(),
        "config_path": str(config_path),
        "profile": args.profile,
        "model": compact_inspection(inspection),
        "expert_pack": pack,
        "command": command,
        "environment": {
            key: environment[key]
            for key in sorted(
                set(config["command"].get("environment", {}))
                | set(profile.get("environment", {}))
                | {"PDCAT_RUN_ID", "PDCAT_REQUEST_ID", "PDCAT_IO_TRACE_PATH"}
            )
        },
    }
    write_json(run_dir / "config.snapshot.json", config)
    write_json(run_dir / "model_manifest.json", inspection)
    write_json(run_dir / "run_manifest.json", manifest)

    base_url = f"http://{args.host}:{args.port}"
    monitor: subprocess.Popen[bytes] | None = None
    server: subprocess.Popen[bytes] | None = None
    with (run_dir / "stdout.log").open("wb") as stdout_file, (run_dir / "stderr.log").open("wb") as stderr_file, monitor_stderr_path.open("wb") as monitor_stderr:
        try:
            server = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, env=environment)
            monitor = subprocess.Popen(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools/monitor_pdcat_process.py"),
                    "--pid",
                    str(server.pid),
                    "--run-id",
                    args.run_id,
                    "--output",
                    str(resource_path),
                    "--interval",
                    "1.0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=monitor_stderr,
            )
            wait_for_server(base_url, server, args.startup_timeout)
            manifest["status"] = "running"
            manifest["server_ready_at_utc"] = utc_now()
            write_json(run_dir / "run_manifest.json", manifest)

            populate_prompt = (
                "Explain bounded asynchronous expert I/O scheduling and cache safety. "
                * args.populate_repeat
            )
            measured_prompt = (
                "Explain why demand expert reads must outrank speculative prefetch and "
                "background KV writes on a unified-memory edge device."
            )
            populate = completion(base_url, populate_prompt, 1, 0)
            # Warm the measured prompt's expert working set before collecting
            # either baseline. Erasing the KV slot preserves expert-cache
            # residency while preventing prompt-cache reuse.
            measured_warmup = completion(base_url, measured_prompt, args.n_predict, 1)
            erase_warmup = erase_slot(base_url, 1)
            baseline_before = completion(base_url, measured_prompt, args.n_predict, 1)
            erase_before = erase_slot(base_url, 1)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                save_future = pool.submit(
                    request_json,
                    base_url + "/slots/0?action=save",
                    {"filename": "slot0-p4.bin"},
                    600.0,
                )
                time.sleep(0.01)
                interference = completion(base_url, measured_prompt, args.n_predict, 1)
                save_response, save_elapsed = save_future.result()
            erase_after = erase_slot(base_url, 1)
            baseline_after = completion(base_url, measured_prompt, args.n_predict, 1)

            baseline_seconds = (
                baseline_before["http_elapsed_seconds"]
                + baseline_after["http_elapsed_seconds"]
            ) / 2.0
            interference_seconds = interference["http_elapsed_seconds"]
            baselines = [baseline_before, baseline_after]
            baseline_prompt_ms = mean_timing(baselines, "prompt_ms")
            baseline_predicted_ms = mean_timing(baselines, "predicted_ms")
            interference_prompt_ms = numeric_timing(interference, "prompt_ms")
            interference_predicted_ms = numeric_timing(interference, "predicted_ms")
            if interference_prompt_ms is None or interference_predicted_ms is None:
                raise RuntimeError("interference response is missing server timings")
            manifest["results"] = {
                "populate": populate,
                "measured_warmup": measured_warmup,
                "erase_warmup": erase_warmup,
                "baseline_before": baseline_before,
                "erase_before": erase_before,
                "interference": interference,
                "p4_save": {
                    "http_elapsed_seconds": save_elapsed,
                    "response": save_response,
                },
                "erase_after": erase_after,
                "baseline_after": baseline_after,
                "baseline_http_seconds_mean": baseline_seconds,
                "interference_http_seconds": interference_seconds,
                "slowdown_ratio": interference_seconds / baseline_seconds,
                "slowdown_ratios": {
                    "http": interference_seconds / baseline_seconds,
                    "prompt": interference_prompt_ms / baseline_prompt_ms,
                    "decode": interference_predicted_ms / baseline_predicted_ms,
                },
                "baseline_prompt_ms_mean": baseline_prompt_ms,
                "interference_prompt_ms": interference_prompt_ms,
                "baseline_predicted_ms": [
                    numeric_timing(baseline_before, "predicted_ms"),
                    numeric_timing(baseline_after, "predicted_ms"),
                ],
                "baseline_predicted_ms_mean": baseline_predicted_ms,
                "interference_predicted_ms": interference_predicted_ms,
            }
            manifest["status"] = "completed"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if monitor is not None:
                try:
                    monitor.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait()

    if trace_path.is_file():
        try:
            manifest["expert_io_summary"] = summarize_jsonl(trace_path)
            write_json(run_dir / "expert_io_summary.json", manifest["expert_io_summary"])
        except (OSError, ValueError) as error:
            manifest["expert_io_summary_error"] = str(error)
    if resource_path.is_file():
        try:
            manifest["resource_summary"] = summarize_resource_jsonl(resource_path)
            write_json(run_dir / "resource_summary.json", manifest["resource_summary"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            manifest["resource_summary_error"] = str(error)
    manifest["finished_at_utc"] = utc_now()
    write_json(run_dir / "run_manifest.json", manifest)
    print(run_dir)
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
