#!/usr/bin/env python3
"""Run and compare the fixed PDCat multi-profile Gate A correctness suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pdcat_model import ModelInspectionError, inspect_from_run_config, sha256_file
from run_pdcat_experiment import (
    DEFAULT_CONFIG,
    ConfigError,
    command_for,
    compact_inspection,
    expand_environment,
    git_snapshot,
    inspect_expert_pack,
    read_json,
    resolve_from_repo,
    validate_config,
    write_json,
)
from summarize_pdcat_io import TraceError, summarize_jsonl
# The shared experiment configuration targets llama-completion.  Gate A uses
# llama-debug's batch mode, whose parser intentionally omits this display flag.
DEBUG_INCOMPATIBLE_ARGS = frozenset({"--no-display-prompt"})


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "experiments/pdcat/datasets/gate_a_100.jsonl"
DEFAULT_BINARY = REPO_ROOT / "build-expert-cache/bin/llama-debug"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/pdcat/runs"
DEFAULT_PROFILES = ("cache_mixed_promote", "cache_mixed")
DEFAULT_STRICT_PAIR = "cache_mixed_promote:cache_mixed"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfigError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ConfigError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def inspect_dataset(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != expected_rows:
        raise ConfigError(
            f"{path}: expected exactly {expected_rows} prompts, got {len(rows)}"
        )
    ids: set[str] = set()
    hashes: set[str] = set()
    split_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        prompt_id = row.get("prompt_id")
        prompt = row.get("prompt")
        expected_hash = row.get("prompt_sha256")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ConfigError(f"{path}:{index + 1}: invalid prompt_id")
        if not isinstance(prompt, str) or not prompt:
            raise ConfigError(f"{path}:{index + 1}: invalid prompt")
        actual_hash = __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest()
        if expected_hash != actual_hash:
            raise ConfigError(
                f"{path}:{index + 1}: prompt SHA-256 mismatch for {prompt_id}"
            )
        if prompt_id in ids or actual_hash in hashes:
            raise ConfigError(f"{path}:{index + 1}: duplicate prompt ID or content")
        ids.add(prompt_id)
        hashes.add(actual_hash)
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "split_counts": split_counts,
        "prompt_ids_sha256": __import__("hashlib")
        .sha256("\n".join(sorted(ids)).encode("utf-8"))
        .hexdigest(),
    }


def load_result_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in indexed:
            raise ConfigError(f"{path}: missing or duplicate prompt_id")
        indexed[prompt_id] = row
    return indexed


def load_route_index(
    path: Path,
) -> tuple[dict[tuple[str, str, int | None, int], dict[str, Any]], list[str]]:
    indexed: dict[tuple[str, str, int | None, int], dict[str, Any]] = {}
    errors: list[str] = []
    for line_number, row in enumerate(load_jsonl(path), 1):
        if row.get("event") != "router_selection":
            continue
        try:
            key = (
                str(row["request_id"]),
                str(row["phase"]),
                row.get("token_id"),
                int(row["layer_id"]),
            )
            experts = row["experts"]
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{path}:{line_number}: malformed route: {error}")
            continue
        if (
            not isinstance(experts, list)
            or not experts
            or not all(isinstance(value, int) for value in experts)
        ):
            errors.append(f"{path}:{line_number}: invalid experts")
            continue
        if key in indexed:
            errors.append(f"{path}:{line_number}: duplicate route context {key}")
            continue
        indexed[key] = row
    if not indexed:
        errors.append(f"{path}: no router_selection events")
    return indexed, errors


def timing_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def profile_summary(profile_dir: Path) -> dict[str, Any]:
    rows = load_result_index(profile_dir / "gate_a_results.jsonl")
    prefill_ms = [float(row["prefill_ms"]) for row in rows.values()]
    decode_ms = [
        float(value)
        for row in rows.values()
        for value in row.get("decode_input_ms", [])
    ]
    return {
        "prompts": len(rows),
        "generated_tokens": sum(
            len(row.get("generated_tokens", [])) for row in rows.values()
        ),
        "prompt_tokens": sum(int(row["prompt_token_count"]) for row in rows.values()),
        "prefill_ms": timing_distribution(prefill_ms),
        "decode_input_ms": timing_distribution(decode_ms),
        "logits_sha256": sha256_file(profile_dir / "gate_a_logits.f32"),
        "tokens_sha256": sha256_file(profile_dir / "gate_a_tokens.i32"),
        "route_trace_sha256": sha256_file(profile_dir / "moe_gate.jsonl"),
    }


def compare_pair(
    run_dir: Path,
    reference_profile: str,
    candidate_profile: str,
    expected_prompts: int,
    criterion: str,
) -> dict[str, Any]:
    reference_dir = run_dir / reference_profile
    candidate_dir = run_dir / candidate_profile
    reference_rows = load_result_index(reference_dir / "gate_a_results.jsonl")
    candidate_rows = load_result_index(candidate_dir / "gate_a_results.jsonl")
    prompt_ids = sorted(set(reference_rows) | set(candidate_rows))
    missing_reference = sorted(set(candidate_rows) - set(reference_rows))
    missing_candidate = sorted(set(reference_rows) - set(candidate_rows))

    reference_logits_path = reference_dir / "gate_a_logits.f32"
    candidate_logits_path = candidate_dir / "gate_a_logits.f32"
    reference_logits = np.memmap(reference_logits_path, mode="r", dtype="<f4")
    candidate_logits = np.memmap(candidate_logits_path, mode="r", dtype="<f4")

    token_mismatches: list[dict[str, Any]] = []
    logits_layout_mismatches: list[str] = []
    top1_mismatches: list[dict[str, Any]] = []
    exact_values = 0
    compared_values = 0
    finite_values = 0
    nonfinite_mismatches = 0
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    max_abs_error = 0.0
    max_abs_context: dict[str, Any] | None = None

    for prompt_id in sorted(set(reference_rows) & set(candidate_rows)):
        left = reference_rows[prompt_id]
        right = candidate_rows[prompt_id]
        left_tokens = left.get("generated_tokens")
        right_tokens = right.get("generated_tokens")
        if left_tokens != right_tokens:
            token_mismatches.append(
                {
                    "prompt_id": prompt_id,
                    "reference": left_tokens,
                    "candidate": right_tokens,
                }
            )

        layout = (
            int(left["logits_step_count"]),
            int(left["n_vocab"]),
            int(left["logits_byte_count"]),
        )
        candidate_layout = (
            int(right["logits_step_count"]),
            int(right["n_vocab"]),
            int(right["logits_byte_count"]),
        )
        if layout != candidate_layout:
            logits_layout_mismatches.append(prompt_id)
            continue

        count = layout[0] * layout[1]
        left_begin = int(left["logits_byte_offset"]) // 4
        right_begin = int(right["logits_byte_offset"]) // 4
        left_values = np.asarray(reference_logits[left_begin : left_begin + count])
        right_values = np.asarray(candidate_logits[right_begin : right_begin + count])
        if left_values.size != count or right_values.size != count:
            logits_layout_mismatches.append(prompt_id)
            continue

        equal = left_values == right_values
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        exact_values += int(np.count_nonzero(equal | both_nan))
        compared_values += count

        finite = np.isfinite(left_values) & np.isfinite(right_values)
        finite_count = int(np.count_nonzero(finite))
        finite_values += finite_count
        nonfinite_equal = (~finite) & (equal | both_nan)
        nonfinite_mismatches += int(np.count_nonzero((~finite) & (~nonfinite_equal)))
        if finite_count:
            differences = np.abs(
                left_values[finite].astype(np.float64)
                - right_values[finite].astype(np.float64)
            )
            abs_error_sum += float(np.sum(differences))
            squared_error_sum += float(np.dot(differences, differences))
            local_index = int(np.argmax(differences))
            local_max = float(differences[local_index])
            if local_max > max_abs_error:
                max_abs_error = local_max
                flattened_finite = np.flatnonzero(finite)
                flat_index = int(flattened_finite[local_index])
                max_abs_context = {
                    "prompt_id": prompt_id,
                    "step": flat_index // layout[1],
                    "vocab_index": flat_index % layout[1],
                }

        left_steps = left_values.reshape(layout[0], layout[1])
        right_steps = right_values.reshape(layout[0], layout[1])
        left_top1 = np.argmax(left_steps, axis=1)
        right_top1 = np.argmax(right_steps, axis=1)
        for step in np.flatnonzero(left_top1 != right_top1):
            top1_mismatches.append(
                {
                    "prompt_id": prompt_id,
                    "step": int(step),
                    "reference": int(left_top1[step]),
                    "candidate": int(right_top1[step]),
                }
            )

    reference_routes, reference_route_errors = load_route_index(
        reference_dir / "moe_gate.jsonl"
    )
    candidate_routes, candidate_route_errors = load_route_index(
        candidate_dir / "moe_gate.jsonl"
    )
    route_keys = sorted(set(reference_routes) | set(candidate_routes))
    route_mismatches: list[dict[str, Any]] = []
    router_score_max_abs = 0.0
    router_score_pairs = 0
    for key in route_keys:
        left = reference_routes.get(key)
        right = candidate_routes.get(key)
        if left is None or right is None or left["experts"] != right["experts"]:
            route_mismatches.append(
                {
                    "context": list(key),
                    "reference": None if left is None else left["experts"],
                    "candidate": None if right is None else right["experts"],
                }
            )
            continue
        left_scores = left.get("router_scores")
        right_scores = right.get("router_scores")
        if (
            isinstance(left_scores, list)
            and isinstance(right_scores, list)
            and len(left_scores) == len(right_scores)
        ):
            router_score_pairs += len(left_scores)
            router_score_max_abs = max(
                router_score_max_abs,
                max(
                    (
                        abs(float(left_value) - float(right_value))
                        for left_value, right_value in zip(left_scores, right_scores)
                    ),
                    default=0.0,
                ),
            )

    failures: list[str] = []
    if len(reference_rows) != expected_prompts or len(candidate_rows) != expected_prompts:
        failures.append("profile result count differs from expected prompt count")
    if missing_reference or missing_candidate:
        failures.append("profile prompt IDs differ")
    if token_mismatches:
        failures.append(f"{len(token_mismatches)} prompts have greedy token mismatches")
    if logits_layout_mismatches:
        failures.append(f"{len(logits_layout_mismatches)} prompts have logits layout mismatches")
    if top1_mismatches:
        failures.append(f"{len(top1_mismatches)} per-step logits top-1 values differ")
    if nonfinite_mismatches:
        failures.append(f"{nonfinite_mismatches} non-finite logits differ")
    if reference_route_errors or candidate_route_errors:
        failures.append("one or both router traces are malformed")
    if criterion == "strict" and route_mismatches:
        failures.append(f"{len(route_mismatches)} router contexts differ")

    return {
        "schema_version": 1,
        "criterion": criterion,
        "reference_profile": reference_profile,
        "candidate_profile": candidate_profile,
        "valid": not failures,
        "failures": failures,
        "prompts_compared": len(set(reference_rows) & set(candidate_rows)),
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "greedy_token_mismatches": token_mismatches[:20],
        "greedy_token_mismatch_count": len(token_mismatches),
        "logits": {
            "values": compared_values,
            "exact_values": exact_values,
            "exact_fraction": exact_values / compared_values if compared_values else None,
            "finite_values": finite_values,
            "nonfinite_mismatches": nonfinite_mismatches,
            "max_abs_error": max_abs_error,
            "max_abs_context": max_abs_context,
            "mean_abs_error": abs_error_sum / finite_values if finite_values else None,
            "rmse": (
                float(np.sqrt(squared_error_sum / finite_values))
                if finite_values
                else None
            ),
            "top1_mismatch_count": len(top1_mismatches),
            "top1_mismatches": top1_mismatches[:20],
            "layout_mismatches": logits_layout_mismatches,
        },
        "router": {
            "reference_events": len(reference_routes),
            "candidate_events": len(candidate_routes),
            "mismatch_count": len(route_mismatches),
            "mismatches": route_mismatches[:20],
            "reference_errors": reference_route_errors,
            "candidate_errors": candidate_route_errors,
            "score_pairs": router_score_pairs,
            "score_max_abs_error": router_score_max_abs,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument(
        "--pair",
        action="append",
        help="strict reference:candidate pair; defaults to cache_mixed_promote:cache_mixed",
    )
    parser.add_argument(
        "--semantic-pair",
        action="append",
        help="cross-backend pair requiring tokens/logits top-1 while reporting route deltas",
    )
    parser.add_argument("--n-predict", type=int, default=2)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--expected-prompts", type=int, default=100)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--verify-model-hash", action="store_true")
    parser.add_argument("--verify-expert-pack-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_predict <= 0 or args.expected_prompts <= 0:
        print("--n-predict and --expected-prompts must be positive", file=sys.stderr)
        return 2
    if args.max_prompts is not None and args.max_prompts <= 0:
        print("--max-prompts must be positive", file=sys.stderr)
        return 2
    if args.no_compare and (args.pair or args.semantic_pair):
        print("--no-compare cannot be combined with pair options", file=sys.stderr)
        return 2


    config_path = args.config.resolve()
    dataset_path = args.dataset.resolve()
    binary = args.binary.resolve()
    profiles = args.profiles or list(DEFAULT_PROFILES)
    pair_specs: list[tuple[str, str, str]] = []
    for criterion, values in (
        ("strict", [] if args.no_compare else (args.pair or [DEFAULT_STRICT_PAIR])),
        ("semantic", args.semantic_pair or []),
    ):
        for value in values:
            if value.count(":") != 1:
                print(
                    f"invalid {criterion} pair {value!r}; expected reference:candidate",
                    file=sys.stderr,
                )
                return 2
            reference, candidate = value.split(":", 1)
            pair_specs.append((criterion, reference, candidate))

    try:
        config = read_json(config_path)
        validate_config(config)
        inspection = inspect_from_run_config(
            config, config_path, hash_model=args.verify_model_hash
        )
        dataset = inspect_dataset(dataset_path, args.expected_prompts)
        for profile in profiles:
            if profile not in config["profiles"]:
                raise ConfigError(f"profile {profile!r} is not defined")
        for _, reference, candidate in pair_specs:
            if reference not in profiles or candidate not in profiles:
                raise ConfigError(
                    f"comparison {reference}:{candidate} requires both profiles"
                )
    except (ConfigError, ModelInspectionError, OSError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    if not binary.is_file() or not os.access(binary, os.X_OK):
        print(f"Gate A binary is missing or not executable: {binary}", file=sys.stderr)
        return 2

    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-gate-a-"
        + "-".join(profiles)
    )
    output_root = args.output_root.resolve()
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"run directory already exists: {run_dir}", file=sys.stderr)
        return 2

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "created_at_utc": utc_now(),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "dataset": dataset,
        "model": compact_inspection(inspection),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "profiles": profiles,
        "pairs": [
            {
                "criterion": criterion,
                "reference": reference,
                "candidate": candidate,
            }
            for criterion, reference, candidate in pair_specs
        ],
        "n_predict": args.n_predict,
        "expected_prompts": args.expected_prompts,
        "repositories": [
            git_snapshot(REPO_ROOT),
            git_snapshot(REPO_ROOT.parent / "InterfaceIO"),
            git_snapshot(REPO_ROOT.parent / "EK-Edge"),
        ],
        "profile_runs": {},
    }
    write_json(run_dir / "manifest.json", manifest)

    expected_model_hash = config["model"]["sha256"].lower()
    actual_hash = inspection["model"].get("sha256")
    if actual_hash is not None and actual_hash.lower() != expected_model_hash:
        manifest["status"] = "failed"
        manifest["error"] = "model SHA-256 mismatch"
        write_json(run_dir / "manifest.json", manifest)
        return 2

    all_succeeded = True
    for profile in profiles:
        profile_dir = run_dir / profile
        profile_dir.mkdir()
        try:
            expert_pack = inspect_expert_pack(
                config, profile, args.verify_expert_pack_hash
            )
            command, configured_environment, _ = command_for(
                config,
                inspection,
                profile,
                expert_pack,
                binary,
                "pdcat-gate-a-bootstrap",
                args.n_predict,
                [
                    "--save-logits",
                    "--logits-output-dir",
                    str(profile_dir),
                ],
            )
            command = [
                argument for argument in command if argument not in DEBUG_INCOMPATIBLE_ARGS
            ]
        except (ConfigError, OSError) as error:
            manifest["profile_runs"][profile] = {
                "status": "failed",
                "error": str(error),
            }
            all_succeeded = False
            break

        configured_environment["PDCAT_GATE_A_DATASET"] = str(dataset_path)
        configured_environment["PDCAT_RUN_ID"] = f"{run_id}-{profile}"
        configured_environment["PDCAT_REQUEST_ID"] = "bootstrap"
        configured_environment["LLAMA_MOE_GATE_TRACE_JSONL"] = str(
            profile_dir / "moe_gate.jsonl"
        )
        configured_environment["PDCAT_IO_TRACE_PATH"] = str(
            profile_dir / "expert_io.jsonl"
        )
        if args.max_prompts is not None:
            configured_environment["PDCAT_GATE_A_MAX_PROMPTS"] = str(args.max_prompts)

        run_record = {
            "status": "running",
            "started_at_utc": utc_now(),
            "command": command,
            "command_display": shlex.join(command),
            "configured_environment": configured_environment,
        }
        manifest["profile_runs"][profile] = run_record
        write_json(run_dir / "manifest.json", manifest)
        print(f"[PDCAT][gate-suite] starting profile={profile}", flush=True)

        environment = os.environ.copy()
        environment.update(expand_environment(configured_environment))
        start = time.monotonic()
        with (profile_dir / "stdout.log").open("wb") as stdout_file, (
            profile_dir / "stderr.log"
        ).open("wb") as stderr_file:
            completed = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        run_record.update(
            {
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - start,
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
            }
        )
        if completed.returncode != 0:
            all_succeeded = False
            write_json(run_dir / "manifest.json", manifest)
            print(
                f"[PDCAT][gate-suite] failed profile={profile} "
                f"returncode={completed.returncode}",
                flush=True,
            )
            break

        io_trace = profile_dir / "expert_io.jsonl"
        if io_trace.is_file() and io_trace.stat().st_size:
            try:
                io_summary = summarize_jsonl(io_trace)
                write_json(profile_dir / "expert_io_summary.json", io_summary)
                run_record["expert_io_summary"] = io_summary
            except (OSError, TraceError, ValueError) as error:
                run_record["expert_io_summary_error"] = str(error)
        run_record["summary"] = profile_summary(profile_dir)
        write_json(run_dir / "manifest.json", manifest)
        print(
            f"[PDCAT][gate-suite] completed profile={profile} "
            f"seconds={run_record['elapsed_seconds']:.3f}",
            flush=True,
        )

    comparisons: dict[str, Any] = {}
    if all_succeeded:
        compare_expected = args.max_prompts or args.expected_prompts
        for criterion, reference, candidate in pair_specs:
            key = f"{criterion}-{reference}__vs__{candidate}"
            comparison = compare_pair(
                run_dir,
                reference,
                candidate,
                compare_expected,
                criterion,
            )
            comparisons[key] = comparison
            write_json(run_dir / f"comparison-{key}.json", comparison)
            if not comparison["valid"]:
                all_succeeded = False

    manifest["comparisons"] = comparisons
    manifest["finished_at_utc"] = utc_now()
    manifest["status"] = "passed" if all_succeeded else "failed"
    write_json(run_dir / "manifest.json", manifest)
    print(run_dir)
    return 0 if all_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
