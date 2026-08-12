#!/usr/bin/env python3
"""Run a versioned PDCat performance/capacity/ablation matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "experiments/pdcat/matrices/deepseek-v2-lite-v03.json"
RUNNER = REPO_ROOT / "tools/run_pdcat_experiment.py"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source_file:
        value = json.load(source_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 1:
        raise ValueError("matrix.schema_version must be 1")
    if not isinstance(matrix.get("matrix_id"), str) or not matrix["matrix_id"]:
        raise ValueError("matrix.matrix_id must be a non-empty string")
    if not isinstance(matrix.get("config"), str) or not matrix["config"]:
        raise ValueError("matrix.config must be a non-empty string")
    if not isinstance(matrix.get("prompt"), str) or not matrix["prompt"]:
        raise ValueError("matrix.prompt must be a non-empty string")
    if not isinstance(matrix.get("n_predict"), int) or matrix["n_predict"] <= 0:
        raise ValueError("matrix.n_predict must be positive")
    if not isinstance(matrix.get("repeats"), int) or matrix["repeats"] <= 0:
        raise ValueError("matrix.repeats must be positive")
    if not isinstance(matrix.get("warmups", 0), int) or matrix.get("warmups", 0) < 0:
        raise ValueError("matrix.warmups must be non-negative")
    if not isinstance(matrix.get("monitor", False), bool):
        raise ValueError("matrix.monitor must be a boolean")
    monitor_interval = matrix.get("monitor_interval", 1.0)
    if not isinstance(monitor_interval, (int, float)) or monitor_interval <= 0:
        raise ValueError("matrix.monitor_interval must be positive")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix.cases must be a non-empty array")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"matrix.cases[{index}] must be an object")
        case_id = case.get("id")
        profile = case.get("profile")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"matrix.cases[{index}].id is missing or duplicate")
        if not isinstance(profile, str) or not profile:
            raise ValueError(f"matrix.cases[{index}].profile must be a string")
        ids.add(case_id)
        extra_args = case.get("extra_args", [])
        environment = case.get("environment", {})
        if not isinstance(extra_args, list) or not all(
            isinstance(value, str) for value in extra_args
        ):
            raise ValueError(f"matrix.cases[{index}].extra_args must be strings")
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float, bool))
            for key, value in environment.items()
        ):
            raise ValueError(f"matrix.cases[{index}].environment must be scalar values")


def resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def command_for(
    matrix: dict[str, Any],
    matrix_path: Path,
    case: dict[str, Any],
    iteration: int,
    measurement: bool,
    runs_dir: Path,
    dry_run: bool,
) -> tuple[list[str], str]:
    matrix_id = matrix["matrix_id"]
    kind = "r" if measurement else "w"
    run_id = f"{matrix_id}-{case['id']}-{kind}{iteration:02d}"
    config = resolve(matrix["config"], matrix_path.parent)
    prompt = str(case.get("prompt", matrix["prompt"]))
    n_predict = int(case.get("n_predict", matrix["n_predict"]))
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config),
        "--profile",
        case["profile"],
        "--run-id",
        run_id,
        "--output-root",
        str(runs_dir),
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
    ]
    for fragment in case.get("extra_args", []):
        tokens = shlex.split(fragment)
        if not tokens:
            raise ValueError(f"matrix case {case['id']}: extra_args contains an empty fragment")
        command.extend(f"--extra-arg={token}" for token in tokens)
    for key, value in sorted(case.get("environment", {}).items()):
        command.extend(["--env", f"{key}={value}"])
    if matrix.get("monitor", False):
        command.extend(["--monitor", "--monitor-interval", str(matrix.get("monitor_interval", 1.0))])

    if dry_run:
        command.append("--dry-run")
    return command, run_id


def nested_number(value: dict[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_measurements(manifest: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, tuple[str, ...]] = {
        "elapsed_seconds": ("elapsed_seconds",),
        "load_milliseconds": ("llama_performance", "load", "milliseconds"),
        "prompt_tokens_per_second": (
            "llama_performance",
            "prompt_eval",
            "tokens_per_second",
        ),
        "decode_tokens_per_second": (
            "llama_performance",
            "decode_eval",
            "tokens_per_second",
        ),
        "total_milliseconds": ("llama_performance", "total", "milliseconds"),
        "cache_hit_rate": ("expert_io_summary", "expert_use", "cache_hit_rate"),
        "prediction_hit_rate": (
            "expert_io_summary",
            "expert_use",
            "prediction_hit_rate_over_uses",
        ),
        "ready_recall": ("expert_io_summary", "ready_recall"),
        "read_amplification": ("expert_io_summary", "read_amplification"),
        "wrong_prefetch_bytes": (
            "expert_io_summary",
            "wrong_prefetch_bytes_correlated",
        ),
        "predictor_latency_mean_us": (
            "expert_io_summary",
            "predictor_latency_us",
            "mean",
        ),
        "rss_peak_bytes": ("resource_summary", "process", "rss_peak_bytes"),
        "temperature_peak_celsius": ("resource_summary", "temperature_celsius_peak"),
    }
    case_ids = sorted(
        {
            str(record["case_id"])
            for record in manifest["runs"]
            if record.get("measurement", False)
        }
    )
    cases: dict[str, Any] = {}
    for case_id in case_ids:
        records = [
            record
            for record in manifest["runs"]
            if record.get("measurement", False) and record["case_id"] == case_id
        ]
        completed = [
            record["run_manifest"]
            for record in records
            if record.get("status") == "completed"
            and isinstance(record.get("run_manifest"), dict)
        ]
        cases[case_id] = {
            "measurement_runs": len(records),
            "completed_runs": len(completed),
            "metrics": {
                field: distribution(
                    [
                        number
                        for child in completed
                        if (number := nested_number(child, *path)) is not None
                    ]
                )
                for field, path in fields.items()
            },
        }
    return {"schema_version": 1, "matrix_id": manifest["matrix_id"], "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    try:
        matrix = read_json(matrix_path)
        validate(matrix)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"matrix error: {error}", file=sys.stderr)
        return 2

    selected = set(args.selected_cases or [])
    available = {case["id"] for case in matrix["cases"]}
    unknown = sorted(selected - available)
    if unknown:
        print(f"unknown matrix cases: {unknown}", file=sys.stderr)
        return 2
    cases = [
        case for case in matrix["cases"] if not selected or case["id"] in selected
    ]

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else REPO_ROOT / "experiments/pdcat/runs" / matrix["matrix_id"]
    )
    if args.dry_run:
        runs_dir = output_root / "runs"
        for case in cases:
            for warmup in range(1, int(case.get("warmups", matrix.get("warmups", 0))) + 1):
                command, _ = command_for(
                    matrix, matrix_path, case, warmup, False, runs_dir, True
                )
                print(shlex.join(command))
            for repeat in range(1, int(case.get("repeats", matrix["repeats"])) + 1):
                command, _ = command_for(
                    matrix, matrix_path, case, repeat, True, runs_dir, True
                )
                print(shlex.join(command))
        return 0

    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"matrix output already exists: {output_root}", file=sys.stderr)
        return 2
    runs_dir = output_root / "runs"
    runs_dir.mkdir()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "status": "running",
        "created_at_utc": utc_now(),
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "selected_cases": [case["id"] for case in cases],
        "runs": [],
    }
    write_json(output_root / "matrix-manifest.json", manifest)

    succeeded = True
    for case in cases:
        warmups = int(case.get("warmups", matrix.get("warmups", 0)))
        repeats = int(case.get("repeats", matrix["repeats"]))
        iterations = [
            (False, iteration) for iteration in range(1, warmups + 1)
        ] + [(True, iteration) for iteration in range(1, repeats + 1)]
        for measurement, iteration in iterations:
            command, run_id = command_for(
                matrix, matrix_path, case, iteration, measurement, runs_dir, False
            )
            record: dict[str, Any] = {
                "case_id": case["id"],
                "measurement": measurement,
                "iteration": iteration,
                "run_id": run_id,
                "command": command,
                "command_display": shlex.join(command),
                "started_at_utc": utc_now(),
                "status": "running",
            }
            manifest["runs"].append(record)
            write_json(output_root / "matrix-manifest.json", manifest)
            completed = subprocess.run(command, check=False)
            record["finished_at_utc"] = utc_now()
            record["returncode"] = completed.returncode
            record["status"] = "completed" if completed.returncode == 0 else "failed"
            child_manifest = runs_dir / run_id / "run_manifest.json"
            if child_manifest.is_file():
                record["run_manifest"] = read_json(child_manifest)
            write_json(output_root / "matrix-manifest.json", manifest)
            if completed.returncode != 0:
                succeeded = False
                if not args.continue_on_error:
                    break
        if not succeeded and not args.continue_on_error:
            break

    manifest["status"] = "passed" if succeeded else "failed"
    manifest["finished_at_utc"] = utc_now()
    write_json(output_root / "matrix-manifest.json", manifest)
    write_json(output_root / "matrix-summary.json", summarize_measurements(manifest))
    print(output_root)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
