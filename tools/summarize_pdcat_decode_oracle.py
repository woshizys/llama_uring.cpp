#!/usr/bin/env python3
"""Summarize token-indexed strict decode-oracle runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .summarize_pdcat_prefill_oracle import aggregate_bandwidth, distribution
except ImportError:
    from summarize_pdcat_prefill_oracle import aggregate_bandwidth, distribution


SUMMARY_SCHEMA_VERSION = 1


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def route_position(event: dict[str, Any]) -> tuple[int, int] | None:
    token_id = event.get("token_id")
    layer = event.get("layer_id")
    if (
        event.get("event") != "router_selection"
        or not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        or not isinstance(layer, int)
        or isinstance(layer, bool)
        or layer < 0
        or not isinstance(event.get("experts"), list)
    ):
        return None
    return token_id, layer


def target_position(event: dict[str, Any]) -> tuple[int, int] | None:
    token_id = event.get("token_id")
    source_layer = event.get("layer_id")
    target_layer = event.get("target_layer_id")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (token_id, source_layer, target_layer)
    ):
        return None
    return token_id + int(target_layer <= source_layer), target_layer


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    io_events = read_jsonl(run_dir / "expert_io.jsonl")
    gate_events = read_jsonl(run_dir / "moe_gate.jsonl")

    routes: dict[tuple[int, int], dict[str, Any]] = {}
    request_ids: set[str] = set()
    for event in gate_events:
        position = route_position(event)
        if position is None:
            continue
        request_id = event.get("request_id")
        if isinstance(request_id, str):
            request_ids.add(request_id)
        previous = routes.setdefault(position, event)
        if set(previous["experts"]) != set(event["experts"]):
            raise ValueError(f"{run_dir}: conflicting route at {position}")
    if not routes:
        raise ValueError(f"{run_dir}: no token-aware router records")
    if len(request_ids) != 1:
        raise ValueError(f"{run_dir}: router records span multiple requests")

    submissions = [
        event
        for event in gate_events
        if event.get("event") == "predictor_submission"
        and event.get("phase") == "decode"
        and target_position(event) is not None
        and isinstance(event.get("experts"), list)
    ]
    predictions: dict[tuple[int, int], set[int]] = {}
    prediction_expert_bytes: dict[tuple[int, int], int] = {}
    requested_labels = 0
    for event in submissions:
        position = target_position(event)
        assert position is not None
        candidates = {
            expert
            for expert in event["experts"]
            if isinstance(expert, int) and not isinstance(expert, bool)
        }
        predictions.setdefault(position, set()).update(candidates)
        expert_bytes = event.get("expert_bytes")
        if isinstance(expert_bytes, int) and expert_bytes > 0:
            prediction_expert_bytes[position] = expert_bytes
        requested = event.get("requested_predictions")
        if isinstance(requested, int):
            requested_labels += requested

    decode_io = [
        event
        for event in io_events
        if event.get("phase") == "decode"
        and isinstance(event.get("token_id"), int)
        and isinstance(event.get("layer_id"), int)
    ]
    use_events = [event for event in decode_io if event.get("event") == "expert_use"]
    read_events = [
        event
        for event in decode_io
        if event.get("event") == "expert_io" and event.get("operation") == "read"
    ]
    prediction_reads = [
        event
        for event in read_events
        if event.get("predicted_probability") is not None
        or event.get("request_class") == "p2-speculative"
    ]
    p0_reads = [
        event
        for event in read_events
        if event.get("request_class") == "p0-demand"
        and event.get("predicted_probability") is None
    ]
    prediction_hits = [event for event in use_events if event.get("prediction_hit") is True]
    on_time_hits = [
        event
        for event in prediction_hits
        if event.get("cache_hit") is True
        and isinstance(event.get("ready_ns"), int)
        and isinstance(event.get("use_ns"), int)
        and event["ready_ns"] <= event["use_ns"]
    ]

    candidate_labels = 0
    correct_labels = 0
    target_labels = 0
    candidate_bytes = 0
    correct_bytes = 0
    target_bytes = 0
    transition_rows: list[dict[str, Any]] = []
    for position, candidates in sorted(predictions.items()):
        actual = set(routes.get(position, {}).get("experts", []))
        correct = candidates & actual
        expert_bytes = prediction_expert_bytes.get(position, 0)
        candidate_labels += len(candidates)
        correct_labels += len(correct)
        target_labels += len(actual)
        candidate_bytes += len(candidates) * expert_bytes
        correct_bytes += len(correct) * expert_bytes
        target_bytes += len(actual) * expert_bytes
        transition_rows.append(
            {
                "token_id": position[0],
                "layer": position[1],
                "predicted": len(candidates),
                "actual": len(actual),
                "correct": len(correct),
            }
        )

    router_time = {
        position: event.get("timestamp_ns") for position, event in routes.items()
    }
    ready_before_router = 0
    for event in prediction_reads:
        position = (event["token_id"], event["layer_id"])
        timestamp_ns = router_time.get(position)
        if (
            isinstance(timestamp_ns, int)
            and isinstance(event.get("ready_ns"), int)
            and event["ready_ns"] <= timestamp_ns
        ):
            ready_before_router += 1

    blocked_ms = [
        event["blocked_ns"] / 1.0e6
        for event in use_events
        if isinstance(event.get("blocked_ns"), int)
    ]
    blocked_by_token: dict[int, float] = {}
    for event in use_events:
        blocked = event.get("blocked_ns")
        if isinstance(blocked, int):
            token_id = event["token_id"]
            blocked_by_token[token_id] = blocked_by_token.get(token_id, 0.0) + blocked / 1.0e6

    performance = manifest.get("llama_performance", {})
    decode_eval = performance.get("decode_eval", {})
    prompt_eval = performance.get("prompt_eval", {})
    actual_keys = {
        (event.get("token_id"), event.get("layer_id"), event.get("logical_expert_id"))
        for event in use_events
    }
    wrong_prediction_reads = [
        event
        for event in prediction_reads
        if (event.get("token_id"), event.get("layer_id"), event.get("logical_expert_id"))
        not in actual_keys
    ]
    budget_profiles = sorted(
        {
            (
                event.get("window_us"),
                event.get("bandwidth_mib_s"),
                event.get("utilization_pct"),
                event.get("budget_bytes"),
                event.get("expert_bytes"),
            )
            for event in submissions
        },
        key=lambda values: tuple(-1 if value is None else value for value in values),
    )
    return {
        "run_id": manifest.get("run_id"),
        "profile": manifest.get("profile"),
        "status": manifest.get("status"),
        "prompt_eval": prompt_eval,
        "decode_eval": decode_eval,
        "ttft_proxy_prompt_eval_ms": prompt_eval.get("milliseconds"),
        "decode_tpot_ms": decode_eval.get("milliseconds_per_token"),
        "decode_tokens_per_second": decode_eval.get("tokens_per_second"),
        "rss_peak_bytes": manifest.get("resource_summary", {})
        .get("process", {})
        .get("rss_peak_bytes"),
        "router_tokens": len({token for token, _ in routes}),
        "router_positions": len(routes),
        "predictor_submissions": len(submissions),
        "predictor_requested_labels": requested_labels,
        "predictor_candidate_labels": candidate_labels,
        "predictor_correct_labels": correct_labels,
        "predictor_target_labels": target_labels,
        "predictor_precision": correct_labels / candidate_labels if candidate_labels else None,
        "predictor_recall": correct_labels / target_labels if target_labels else None,
        "predictor_candidate_bytes": candidate_bytes,
        "predictor_correct_bytes": correct_bytes,
        "predictor_target_bytes": target_bytes,
        "predictor_byte_precision": correct_bytes / candidate_bytes if candidate_bytes else None,
        "predictor_byte_recall": correct_bytes / target_bytes if target_bytes else None,
        "expert_uses": len(use_events),
        "prediction_hit_uses": len(prediction_hits),
        "on_time_prediction_uses": len(on_time_hits),
        "ready_recall": len(on_time_hits) / len(use_events) if use_events else None,
        "late_prefetch_rate": (
            (len(prediction_hits) - len(on_time_hits)) / len(prediction_hits)
            if prediction_hits
            else None
        ),
        "prediction_reads_ready_before_router": ready_before_router,
        "p0_blocked_ms": distribution(blocked_ms),
        "per_token_total_blocked_ms": distribution(blocked_by_token.values()),
        "p0_bandwidth": aggregate_bandwidth(p0_reads),
        "prediction_bandwidth": aggregate_bandwidth(prediction_reads),
        "total_read_bandwidth": aggregate_bandwidth(read_events),
        "read_amplification_over_actual_uses": (
            sum(event.get("issued_bytes", 0) for event in read_events)
            / sum(event.get("useful_bytes", 0) for event in use_events)
            if sum(event.get("useful_bytes", 0) for event in use_events) > 0
            else None
        ),
        "wrong_prefetch_read_events": len(wrong_prediction_reads),
        "wrong_prefetch_bytes": sum(
            event.get("issued_bytes", 0) for event in wrong_prediction_reads
        ),
        "predictor_budget_profiles": [
            {
                "window_us": values[0],
                "bandwidth_mib_s": values[1],
                "utilization_pct": values[2],
                "budget_bytes": values[3],
                "expert_bytes": values[4],
                "candidate_limit": (
                    values[3] // values[4]
                    if isinstance(values[3], int)
                    and isinstance(values[4], int)
                    and values[4] > 0
                    else None
                ),
            }
            for values in budget_profiles
        ],
        "transitions": transition_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "runs": [summarize_run(path.resolve()) for path in args.run_dirs],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
