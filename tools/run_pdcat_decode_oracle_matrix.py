#!/usr/bin/env python3
"""Run a token-indexed strict decode-oracle matrix with stop-on-NVMe-failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

try:
    from .export_native_moe_oracle_trace import export_token_routes, load_token_router_routes
    from .run_pdcat_prefill_oracle_matrix import (
        CELLS,
        DEFAULT_HOST_DATA,
        DEFAULT_HOST_WORKSPACE,
        DEFAULT_IMAGE,
        REPO_ROOT,
        container_path,
        docker_command,
        kernel_nvme1_failures,
        new_lines,
        pair_fingerprint,
        parse_cell,
        read_json,
        run_container,
        utc_now,
        write_json,
    )
    from .summarize_pdcat_decode_oracle import (
        SUMMARY_SCHEMA_VERSION,
        summarize_run,
    )
except ImportError:
    from export_native_moe_oracle_trace import export_token_routes, load_token_router_routes
    from run_pdcat_prefill_oracle_matrix import (
        CELLS,
        DEFAULT_HOST_DATA,
        DEFAULT_HOST_WORKSPACE,
        DEFAULT_IMAGE,
        REPO_ROOT,
        container_path,
        docker_command,
        kernel_nvme1_failures,
        new_lines,
        pair_fingerprint,
        parse_cell,
        read_json,
        run_container,
        utc_now,
        write_json,
    )
    from summarize_pdcat_decode_oracle import SUMMARY_SCHEMA_VERSION, summarize_run


DECODE_WINDOW_US = 12_000


def add_ignore_eos(command: list[str]) -> list[str]:
    command.extend(["--extra-arg=--ignore-eos"])
    return command


def validate_route_grid(
    routes: list[tuple[int, int, list[int]]],
    expected_token_count: int,
    expected_layers: list[int],
) -> None:
    expected_tokens = set(range(expected_token_count))
    layers_by_token: dict[int, set[int]] = {}
    for token_id, layer, _ in routes:
        layers_by_token.setdefault(token_id, set()).add(layer)
    if set(layers_by_token) != expected_tokens:
        raise ValueError(
            f"expected router token ids 0..{expected_token_count - 1}, got "
            f"{sorted(layers_by_token)}"
        )
    expected_layer_set = set(expected_layers)
    if not expected_layer_set:
        raise ValueError("model manifest has no MoE layers")
    incomplete = {
        token_id: sorted(expected_layer_set - layers)
        for token_id, layers in layers_by_token.items()
        if layers != expected_layer_set
    }
    extras = {
        token_id: sorted(layers - expected_layer_set)
        for token_id, layers in layers_by_token.items()
        if layers - expected_layer_set
    }
    if incomplete or extras:
        raise ValueError(
            f"incomplete token/layer route grid: missing={incomplete}, extras={extras}"
        )


def validate_decode_pair(
    baseline_dir: Path,
    oracle_dir: Path,
    prompt_tokens: int,
    decode_tokens: int,
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

    for name, manifest in (("baseline", baseline_manifest), ("oracle", oracle_manifest)):
        performance = manifest.get("llama_performance", {})
        actual_prompt = performance.get("prompt_eval", {}).get("count")
        actual_decode = performance.get("decode_eval", {}).get("count")
        if actual_prompt != prompt_tokens:
            raise ValueError(
                f"{name}: expected {prompt_tokens} prompt tokens, got {actual_prompt}"
            )
        if actual_decode != decode_tokens - 1:
            raise ValueError(
                f"{name}: expected {decode_tokens - 1} decode eval runs, got {actual_decode}"
            )

    baseline_routes = load_token_router_routes(baseline_dir / "moe_gate.jsonl")
    oracle_routes = load_token_router_routes(oracle_dir / "moe_gate.jsonl")
    if baseline_routes != oracle_routes:
        raise ValueError("baseline/oracle token-indexed router selections differ")
    expected_layers = baseline_manifest["model"]["topology"]["moe_layers"]
    if not isinstance(expected_layers, list) or not all(
        isinstance(layer, int) and not isinstance(layer, bool) for layer in expected_layers
    ):
        raise ValueError("baseline model manifest has invalid moe_layers")
    validate_route_grid(baseline_routes, decode_tokens, expected_layers)
    observed_tokens = {token_id for token_id, _, _ in baseline_routes}

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
        "router_tokens": len(observed_tokens),
        "router_positions": len(baseline_routes),
        "router_expert_labels": sum(len(experts) for _, _, experts in baseline_routes),
        "generated_output_match": True,
        "generated_output_bytes": len(baseline_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--cell", action="append", type=parse_cell)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--host-workspace", type=Path, default=DEFAULT_HOST_WORKSPACE)
    parser.add_argument("--host-data", type=Path, default=DEFAULT_HOST_DATA)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
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
    if args.decode_tokens < 2:
        raise SystemExit("--decode-tokens must be at least 2")
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
        cell_id = f"{memory_gib}g-p{prompt_tokens}-d{args.decode_tokens}"
        baseline_id = f"{args.matrix_id}-{cell_id}-baseline"
        baseline = add_ignore_eos(
            docker_command(
                image=args.image,
                host_workspace=args.host_workspace,
                host_data=args.host_data,
                memory_gib=memory_gib,
                container_name=f"pdcat-{cell_id}-baseline",
                profile="cache_mixed",
                run_id=baseline_id,
                output_root=runs_root,
                prompt_tokens=prompt_tokens,
                n_predict=args.decode_tokens,
                environment={"PDCAT_IO_MAX_QD": "2"},
            )
        )
        planned.append(((memory_gib, prompt_tokens), "baseline", baseline))

        oracle_path = oracle_root / f"{cell_id}.jsonl"
        oracle_id = f"{args.matrix_id}-{cell_id}-oracle"
        oracle = add_ignore_eos(
            docker_command(
                image=args.image,
                host_workspace=args.host_workspace,
                host_data=args.host_data,
                memory_gib=memory_gib,
                container_name=f"pdcat-{cell_id}-oracle",
                profile="cache_mixed_decode_oracle",
                run_id=oracle_id,
                output_root=runs_root,
                prompt_tokens=prompt_tokens,
                n_predict=args.decode_tokens,
                environment={
                    "LLAMA_MOE_DECODE_PREFETCH_WINDOW_US": str(DECODE_WINDOW_US),
                    "PDCAT_PREDICTOR_TRACE": str(container_path(oracle_path)),
                },
            )
        )
        planned.append(((memory_gib, prompt_tokens), "oracle", oracle))

    if args.dry_run:
        for cell, phase, command in planned:
            print(
                f"# {cell[0]} GiB / {cell[1]} prompt tokens / "
                f"{args.decode_tokens} generated tokens / {phase}"
            )
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
        "decode_tokens": args.decode_tokens,
        "risk_acknowledged": True,
        "stop_policy": "stop after first nonzero exit or new nvme1 failure; never retry",
        "runs": [],
    }
    manifest_path = output_root / "matrix_manifest.json"
    write_json(manifest_path, manifest)

    for (memory_gib, prompt_tokens), phase, command in planned:
        cell_id = f"{memory_gib}g-p{prompt_tokens}-d{args.decode_tokens}"
        run_id = command[command.index("--run-id") + 1]
        container_name = command[command.index("--name") + 1]
        before = kernel_nvme1_failures()
        started = utc_now()
        returncode = run_container(command, container_name, args.timeout_seconds)
        after = kernel_nvme1_failures()
        failures = new_lines(before, after)
        record: dict[str, Any] = {
            "cell": cell_id,
            "phase": phase,
            "run_id": run_id,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "returncode": returncode,
            "new_nvme1_failures": failures,
            "command": command,
        }
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)
        contaminated = bool(failures)
        if returncode != 0:
            manifest["status"] = "contaminated" if contaminated else "failed"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return returncode

        run_dir = runs_root / run_id
        try:
            run_manifest = read_json(run_dir / "run_manifest.json")
            performance = run_manifest.get("llama_performance", {})
            actual_prompt = performance.get("prompt_eval", {}).get("count")
            actual_decode = performance.get("decode_eval", {}).get("count")
            if actual_prompt != prompt_tokens or actual_decode != args.decode_tokens - 1:
                raise ValueError(
                    f"expected prompt/decode {prompt_tokens}/{args.decode_tokens - 1}, "
                    f"got {actual_prompt}/{actual_decode}"
                )
            routes = load_token_router_routes(run_dir / "moe_gate.jsonl")
            expected_layers = run_manifest["model"]["topology"]["moe_layers"]
            if not isinstance(expected_layers, list) or not all(
                isinstance(layer, int) and not isinstance(layer, bool)
                for layer in expected_layers
            ):
                raise ValueError("run manifest has invalid moe_layers")
            validate_route_grid(routes, args.decode_tokens, expected_layers)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            record["validation_error"] = str(error)
            manifest["status"] = "contaminated" if contaminated else "failed"
            manifest["finished_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            return 3 if contaminated else 4

        if phase == "baseline":
            oracle_path = oracle_root / f"{cell_id}.jsonl"
            export_token_routes(routes, oracle_path, run_id, "runtime")
            record["oracle_trace"] = str(oracle_path)
            record["oracle_positions"] = len(routes)
            record["oracle_expert_labels"] = sum(
                len(experts) for _, _, experts in routes
            )
        else:
            baseline_id = f"{args.matrix_id}-{cell_id}-baseline"
            baseline_dir = runs_root / baseline_id
            try:
                record["pair_validation"] = validate_decode_pair(
                    baseline_dir, run_dir, prompt_tokens, args.decode_tokens
                )
                write_json(
                    output_root / "decode_summary.json",
                    {
                        "schema_version": SUMMARY_SCHEMA_VERSION,
                        "runs": [summarize_run(baseline_dir), summarize_run(run_dir)],
                    },
                )
                record["summary"] = str(output_root / "decode_summary.json")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                record["validation_error"] = f"pair validation failed: {error}"
                manifest["status"] = "contaminated" if contaminated else "failed"
                manifest["finished_at_utc"] = utc_now()
                write_json(manifest_path, manifest)
                return 3 if contaminated else 4
        write_json(manifest_path, manifest)

        if contaminated:
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
