#!/usr/bin/env python3
"""Run the minimal 4/8 GiB x 128/256-token PDCat prefill oracle matrix.

Each cell runs a demand-only source pass, exports that pass's exact router
transitions, then runs the oracle pass with the same prompt. Execution requires
an explicit NVMe-risk acknowledgement. Any new nvme1 timeout/reset/error stops
the matrix immediately and no retry is attempted.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .export_native_moe_oracle_trace import export_routes, load_routes
    from .summarize_pdcat_prefill_oracle import SUMMARY_SCHEMA_VERSION, summarize_run
except ImportError:
    from export_native_moe_oracle_trace import export_routes, load_routes
    from summarize_pdcat_prefill_oracle import SUMMARY_SCHEMA_VERSION, summarize_run


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "kural:v1.2"
DEFAULT_HOST_WORKSPACE = Path("/home/kural")
DEFAULT_HOST_DATA = Path("/data")
CONTAINER_REPO = Path("/workspace/llama_uring.cpp")
CONFIG = CONTAINER_REPO / "experiments/pdcat/configs/deepseek-v2-lite-q8_0.json"
RUNNER = CONTAINER_REPO / "tools/run_pdcat_experiment.py"
CELLS = ((4, 128), (8, 128), (4, 256), (8, 256))
PREFILL_WINDOW_US = {128: 50_000, 256: 72_000}
NVME1_FAILURE = re.compile(
    r"nvme\s+nvme1:.*(?:timeout|completion polled|reset|I/O.*error)", re.IGNORECASE
)
PAIR_SHARED_ENVIRONMENT = (
    "LLAMA_MOE_CUDA_DELIVERY",
    "PDCAT_IO_MAX_QD",
    "PDCAT_IO_MAX_INFLIGHT_BYTES",
    "PDCAT_IO_P0_RESERVED_BYTES",
    "PDCAT_IO_MAX_P1_QD",
    "PDCAT_IO_MAX_P1_INFLIGHT_BYTES",
    "PDCAT_IO_MAX_SPECULATIVE_BYTES",
    "PDCAT_IO_MAX_BACKGROUND_BYTES",
    "PDCAT_MODEL_EXPERT_COUNT",
    "PDCAT_MODEL_ID",
    "PDCAT_MODEL_SHA256",
    "PDCAT_PHASE",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prompt_for(token_count: int) -> str:
    return " hello" * (token_count - 1)


def parse_cell(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(4|8)g[-/:](128|256)", value.lower())
    if match is None:
        raise argparse.ArgumentTypeError("cell must be 4g-128, 8g-128, 4g-256, or 8g-256")
    return int(match.group(1)), int(match.group(2))


def kernel_nvme1_failures() -> list[str]:
    result = subprocess.run(
        ["dmesg", "--color=never"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read kernel log: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if NVME1_FAILURE.search(line)]


def new_lines(before: list[str], after: list[str]) -> list[str]:
    return list((collections.Counter(after) - collections.Counter(before)).elements())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def pair_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("configured_environment")
    if not isinstance(environment, dict):
        raise ValueError("run manifest has no configured_environment object")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("run manifest has no valid command array")
    required_types = {
        "host": dict,
        "repositories": list,
        "runtime_artifacts": list,
        "config_sha256": str,
        "model": dict,
        "model_sha256_expected": str,
        "expert_pack": dict,
    }
    for key, expected_type in required_types.items():
        if not isinstance(manifest.get(key), expected_type):
            raise ValueError(f"run manifest has no valid {key}")
    if not manifest["repositories"]:
        raise ValueError("run manifest repository snapshot is empty")
    if not manifest["runtime_artifacts"]:
        raise ValueError("run manifest runtime artifact snapshot is empty")
    for artifact in manifest["runtime_artifacts"]:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("size_bytes"), int)
            or not isinstance(artifact.get("sha256"), str)
        ):
            raise ValueError("run manifest has an invalid runtime artifact")
    return {
        "host": manifest.get("host"),
        "repositories": manifest.get("repositories"),
        "runtime_artifacts": manifest.get("runtime_artifacts"),
        "config_sha256": manifest.get("config_sha256"),
        "model": manifest.get("model"),
        "model_sha256_expected": manifest.get("model_sha256_expected"),
        "expert_pack": manifest.get("expert_pack"),
        "command": command,
        "shared_environment": {
            key: environment.get(key) for key in PAIR_SHARED_ENVIRONMENT
        },
    }


def validate_pair_artifacts(
    baseline_dir: Path, oracle_dir: Path, prompt_tokens: int
) -> dict[str, Any]:
    baseline_manifest = read_json(baseline_dir / "run_manifest.json")
    oracle_manifest = read_json(oracle_dir / "run_manifest.json")
    baseline_fingerprint = pair_fingerprint(baseline_manifest)
    oracle_fingerprint = pair_fingerprint(oracle_manifest)
    if baseline_fingerprint != oracle_fingerprint:
        differing = sorted(
            key
            for key in set(baseline_fingerprint) | set(oracle_fingerprint)
            if baseline_fingerprint.get(key) != oracle_fingerprint.get(key)
        )
        raise ValueError(f"baseline/oracle pair differs in {differing}")

    baseline_routes = load_routes(baseline_dir / "moe_gate.jsonl", prompt_tokens)
    oracle_routes = load_routes(oracle_dir / "moe_gate.jsonl", prompt_tokens)
    if baseline_routes != oracle_routes:
        differing_layers = sorted(
            layer
            for layer in set(baseline_routes) | set(oracle_routes)
            if baseline_routes.get(layer) != oracle_routes.get(layer)
        )
        raise ValueError(
            f"baseline/oracle router selections differ at layers {differing_layers}"
        )

    baseline_output = (baseline_dir / "stdout.log").read_bytes()
    oracle_output = (oracle_dir / "stdout.log").read_bytes()
    if baseline_output != oracle_output:
        raise ValueError("baseline/oracle generated output differs")

    fingerprint_sha256 = hashlib.sha256(
        json.dumps(
            baseline_fingerprint, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "pair_fingerprint_sha256": fingerprint_sha256,
        "router_match": True,
        "router_layers": len(baseline_routes),
        "router_expert_labels": sum(map(len, baseline_routes.values())),
        "generated_output_match": True,
        "generated_output_bytes": len(baseline_output),
    }


def write_pair_summary(
    output_root: Path, baseline_dir: Path, oracle_dir: Path
) -> None:
    write_json(
        output_root / "summary.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "runs": [summarize_run(baseline_dir), summarize_run(oracle_dir)],
        },
    )


def container_path(host_visible_path: Path) -> Path:
    relative = host_visible_path.resolve().relative_to(REPO_ROOT)
    return CONTAINER_REPO / relative


def docker_command(
    *,
    image: str,
    host_workspace: Path,
    host_data: Path,
    memory_gib: int,
    container_name: str,
    profile: str,
    run_id: str,
    output_root: Path,
    prompt_tokens: int,
    environment: dict[str, str],
    n_predict: int = 1,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--runtime",
        "nvidia",
        "--privileged",
        "--memory",
        f"{memory_gib}g",
        "--memory-swap",
        f"{memory_gib}g",
        "--shm-size",
        "2g",
        "-v",
        f"{host_workspace}:/workspace",
        "-v",
        f"{host_data}:/data",
        "-w",
        str(CONTAINER_REPO),
        image,
        "python3",
        "-u",
        str(RUNNER.relative_to(CONTAINER_REPO)),
        "--config",
        str(CONFIG),
        "--profile",
        profile,
        "--run-id",
        run_id,
        "--output-root",
        str(container_path(output_root)),
        "--prompt",
        prompt_for(prompt_tokens),
        "--n-predict",
        str(n_predict),
        "--extra-arg=--no-conversation",
        "--monitor",
        "--monitor-interval",
        "0.25",
    ]
    for key, value in sorted(environment.items()):
        command.extend(["--env", f"{key}={value}"])
    return command


def run_container(command: list[str], container_name: str, timeout_seconds: int) -> int:
    try:
        return subprocess.run(command, check=False, timeout=timeout_seconds).returncode
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "stop", "--time", "5", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return 124


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--cell", action="append", type=parse_cell)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--host-workspace", type=Path, default=DEFAULT_HOST_WORKSPACE)
    parser.add_argument("--host-data", type=Path, default=DEFAULT_HOST_DATA)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acknowledge-nvme-risk",
        action="store_true",
        help="required for execution: nvme1 may timeout or freeze the Jetson",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    selected = args.cell or list(CELLS)
    if len(set(selected)) != len(selected):
        raise SystemExit("duplicate --cell")
    if not args.dry_run and not args.acknowledge_nvme_risk:
        raise SystemExit(
            "execution refused: pass --acknowledge-nvme-risk only after the user "
            "explicitly accepts possible nvme1 timeout/system freeze"
        )

    output_root = REPO_ROOT / "experiments/pdcat/runs" / args.matrix_id
    runs_root = output_root / "runs"
    oracle_root = output_root / "oracles"

    planned: list[tuple[tuple[int, int], str, list[str]]] = []
    for memory_gib, prompt_tokens in selected:
        cell_id = f"{memory_gib}g-p{prompt_tokens}"
        baseline_id = f"{args.matrix_id}-{cell_id}-baseline"
        baseline_name = f"pdcat-{cell_id}-baseline"
        baseline = docker_command(
            image=args.image,
            host_workspace=args.host_workspace,
            host_data=args.host_data,
            memory_gib=memory_gib,
            container_name=baseline_name,
            profile="cache_mixed",
            run_id=baseline_id,
            output_root=runs_root,
            prompt_tokens=prompt_tokens,
            environment={"PDCAT_IO_MAX_QD": "2"},
        )
        planned.append(((memory_gib, prompt_tokens), "baseline", baseline))

        oracle_id = f"{args.matrix_id}-{cell_id}-oracle"
        oracle_name = f"pdcat-{cell_id}-oracle"
        oracle_path = oracle_root / f"{cell_id}.jsonl"
        oracle = docker_command(
            image=args.image,
            host_workspace=args.host_workspace,
            host_data=args.host_data,
            memory_gib=memory_gib,
            container_name=oracle_name,
            profile="cache_mixed_prefill_oracle",
            run_id=oracle_id,
            output_root=runs_root,
            prompt_tokens=prompt_tokens,
            environment={
                "LLAMA_MOE_PREFILL_PREFETCH_WINDOW_US": str(
                    PREFILL_WINDOW_US[prompt_tokens]
                ),
                "PDCAT_PREDICTOR_TRACE": str(container_path(oracle_path)),
            },
        )
        planned.append(((memory_gib, prompt_tokens), "oracle", oracle))

    if args.dry_run:
        for cell, phase, command in planned:
            print(f"# {cell[0]} GiB / {cell[1]} tokens / {phase}")
            print(shlex.join(command))
        return 0

    if output_root.exists():
        raise SystemExit(f"matrix output already exists: {output_root}")
    runs_root.mkdir(parents=True)
    oracle_root.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "matrix_id": args.matrix_id,
        "status": "running",
        "created_at_utc": utc_now(),
        "cells": [f"{memory}g-p{tokens}" for memory, tokens in selected],
        "risk_acknowledged": True,
        "stop_policy": "stop after first nonzero exit or new nvme1 failure; never retry",
        "runs": [],
    }
    manifest_path = output_root / "matrix_manifest.json"
    write_json(manifest_path, manifest)

    for (memory_gib, prompt_tokens), phase, command in planned:
        cell_id = f"{memory_gib}g-p{prompt_tokens}"
        run_id = command[command.index("--run-id") + 1]
        container_name = command[command.index("--name") + 1]
        before = kernel_nvme1_failures()
        started = utc_now()
        returncode = run_container(command, container_name, args.timeout_seconds)
        after = kernel_nvme1_failures()
        new_failures = new_lines(before, after)
        record: dict[str, Any] = {
            "cell": cell_id,
            "phase": phase,
            "run_id": run_id,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "returncode": returncode,
            "new_nvme1_failures": new_failures,
            "command": command,
        }
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)

        hardware_contaminated = bool(new_failures)
        if returncode != 0:
            manifest["status"] = "contaminated" if new_failures else "failed"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return returncode

        run_manifest_path = runs_root / run_id / "run_manifest.json"
        try:
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            actual_prompt_tokens = run_manifest["llama_performance"]["prompt_eval"]["count"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            record["validation_error"] = f"cannot read prompt token count: {error}"
            manifest["status"] = "contaminated" if hardware_contaminated else "failed"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return 3 if hardware_contaminated else 4
        record["actual_prompt_tokens"] = actual_prompt_tokens
        if actual_prompt_tokens != prompt_tokens:
            record["validation_error"] = (
                f"expected {prompt_tokens} prompt tokens, got {actual_prompt_tokens}"
            )
            manifest["status"] = "contaminated" if hardware_contaminated else "failed"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return 3 if hardware_contaminated else 4

        if phase == "baseline":
            gate_trace = runs_root / run_id / "moe_gate.jsonl"
            oracle_path = oracle_root / f"{cell_id}.jsonl"
            try:
                routes = load_routes(gate_trace, prompt_tokens)
                export_routes(routes, oracle_path, run_id, "runtime")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                record["validation_error"] = f"cannot export strict oracle: {error}"
                manifest["status"] = (
                    "contaminated" if hardware_contaminated else "failed"
                )
                manifest["finished_at_utc"] = utc_now()
                write_json(manifest_path, manifest)
                return 3 if hardware_contaminated else 4
            record["oracle_trace"] = str(oracle_path)
            record["oracle_layers"] = len(routes)
            record["oracle_expert_labels"] = sum(map(len, routes.values()))
            write_json(manifest_path, manifest)
        else:
            baseline_id = f"{args.matrix_id}-{cell_id}-baseline"
            baseline_dir = runs_root / baseline_id
            oracle_dir = runs_root / run_id
            try:
                record["pair_validation"] = validate_pair_artifacts(
                    baseline_dir, oracle_dir, prompt_tokens
                )
                write_pair_summary(output_root, baseline_dir, oracle_dir)
                record["summary"] = str(output_root / "summary.json")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                record["validation_error"] = f"pair validation failed: {error}"
                manifest["status"] = (
                    "contaminated" if hardware_contaminated else "failed"
                )
                manifest["finished_at_utc"] = utc_now()
                write_json(manifest_path, manifest)
                return 3 if hardware_contaminated else 4
            write_json(manifest_path, manifest)

        # A completed process may still have triggered an NVMe timeout that the
        # kernel recovered via polling. Finish all read-only validation for that
        # run, then stop before another container can be launched.
        if hardware_contaminated:
            manifest["status"] = "contaminated"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return 3

    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
