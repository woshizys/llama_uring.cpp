#!/usr/bin/env python3
"""Summarize TTFT, P0 stalls, P2 bandwidth, and per-layer oracle readiness."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


SUMMARY_SCHEMA_VERSION = 2


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


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized) if materialized else None,
        "median": statistics.median(materialized) if materialized else None,
        "p95": percentile(materialized, 0.95),
        "max": max(materialized) if materialized else None,
    }


def aggregate_bandwidth(events: list[dict[str, Any]]) -> dict[str, float | int | None]:
    successful = [
        event
        for event in events
        if event.get("event") == "expert_io"
        and event.get("operation") == "read"
        and event.get("success") is True
        and isinstance(event.get("cqe_ns"), int)
        and isinstance(event.get("submit_ns"), int)
        and isinstance(event.get("blocked_ns"), int)
        and isinstance(event.get("issued_bytes"), int)
    ]
    if not successful:
        return {
            "events": 0,
            "bytes": 0,
            "active_span_ms": None,
            "mib_per_second": None,
            "service_ms": distribution([]),
            "clean_under_1s_events": 0,
            "clean_union_active_ms": None,
            "clean_union_mib_per_second": None,
        }
    first_start = min(event["submit_ns"] + event["blocked_ns"] for event in successful)
    last_cqe = max(event["cqe_ns"] for event in successful)
    span_ns = max(0, last_cqe - first_start)
    byte_count = sum(event["issued_bytes"] for event in successful)
    service_ms = [
        (event["cqe_ns"] - event["submit_ns"] - event["blocked_ns"]) / 1.0e6
        for event in successful
    ]
    clean = [
        event
        for event, milliseconds in zip(successful, service_ms)
        if milliseconds < 1000.0
    ]
    intervals = sorted(
        (event["submit_ns"] + event["blocked_ns"], event["cqe_ns"]) for event in clean
    )
    union_ns = 0
    if intervals:
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                union_ns += current_end - current_start
                current_start, current_end = start, end
        union_ns += current_end - current_start
    clean_bytes = sum(event["issued_bytes"] for event in clean)
    return {
        "events": len(successful),
        "bytes": byte_count,
        "active_span_ms": span_ns / 1.0e6,
        "mib_per_second": (
            byte_count / (1024.0 * 1024.0) / (span_ns / 1.0e9)
            if span_ns > 0
            else None
        ),
        "service_ms": distribution(service_ms),
        "clean_under_1s_events": len(clean),
        "clean_union_active_ms": union_ns / 1.0e6 if union_ns else None,
        "clean_union_mib_per_second": (
            clean_bytes / (1024.0 * 1024.0) / (union_ns / 1.0e9)
            if union_ns > 0
            else None
        ),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    io_events = read_jsonl(run_dir / "expert_io.jsonl")
    gate_events = read_jsonl(run_dir / "moe_gate.jsonl")

    prompt_tokens = (
        manifest.get("llama_performance", {}).get("prompt_eval", {}).get("count")
    )
    router_events = [
        event
        for event in gate_events
        if event.get("event") == "router_selection"
        and isinstance(event.get("layer_id"), int)
        and isinstance(event.get("experts"), list)
    ]
    full_router_events = [
        event for event in router_events if event.get("token_count") == prompt_tokens
    ]
    if not full_router_events:
        raise ValueError(f"{run_dir}: no router records for {prompt_tokens} prompt tokens")
    request_ids = {event.get("request_id") for event in full_router_events}
    if len(request_ids) != 1:
        raise ValueError(f"{run_dir}: prompt router records span multiple requests")
    request_id = next(iter(request_ids))
    router_by_layer = {event["layer_id"]: event for event in full_router_events}
    max_layer = max(router_by_layer)
    last_full_index = max(gate_events.index(event) for event in full_router_events)
    for event in gate_events[last_full_index + 1 :]:
        if event.get("event") != "router_selection":
            continue
        layer = event.get("layer_id")
        if (
            event.get("request_id") != request_id
            or event.get("token_count") != 1
            or not isinstance(layer, int)
            or layer <= max_layer
        ):
            break
        router_by_layer[layer] = event
        max_layer = layer

    predictor_submissions = [
        event
        for event in gate_events
        if event.get("event") == "predictor_submission"
        and event.get("phase") == "prefill"
        and event.get("token_count") == prompt_tokens
        and isinstance(event.get("target_layer_id"), int)
        and isinstance(event.get("experts"), list)
    ]
    prediction_candidates_by_layer: dict[int, set[int]] = {}
    prediction_expert_bytes_by_layer: dict[int, int] = {}
    requested_prediction_labels = 0
    for event in predictor_submissions:
        target_layer = event["target_layer_id"]
        candidates = {
            expert for expert in event["experts"] if isinstance(expert, int)
        }
        prediction_candidates_by_layer.setdefault(target_layer, set()).update(candidates)
        expert_bytes = event.get("expert_bytes")
        if isinstance(expert_bytes, int) and expert_bytes > 0:
            prediction_expert_bytes_by_layer[target_layer] = expert_bytes
        requested = event.get("requested_predictions")
        if isinstance(requested, int):
            requested_prediction_labels += requested

    numeric_token_ids = sorted(
        {
            event["token_id"]
            for event in io_events
            if isinstance(event.get("token_id"), int)
        }
    )
    prompt_token_id = numeric_token_ids[0] if numeric_token_ids else None
    io_events = [
        event
        for event in io_events
        if event.get("token_id") is None or event.get("token_id") == prompt_token_id
    ]
    uses_by_layer: dict[int, list[dict[str, Any]]] = {}
    reads_by_layer: dict[int, list[dict[str, Any]]] = {}
    for event in io_events:
        layer = event.get("layer_id")
        if not isinstance(layer, int):
            continue
        if event.get("event") == "expert_use":
            uses_by_layer.setdefault(layer, []).append(event)
        elif event.get("event") == "expert_io" and event.get("operation") == "read":
            reads_by_layer.setdefault(layer, []).append(event)

    layer_rows: list[dict[str, Any]] = []
    for layer, router in sorted(router_by_layer.items()):
        uses = uses_by_layer.get(layer, [])
        reads = reads_by_layer.get(layer, [])
        prediction_reads = [
            event
            for event in reads
            if event.get("predicted_probability") is not None
            or event.get("request_class") == "p2-speculative"
        ]
        router_ns = router.get("timestamp_ns")
        ready_before_router = [
            event
            for event in prediction_reads
            if isinstance(router_ns, int)
            and isinstance(event.get("ready_ns"), int)
            and event["ready_ns"] <= router_ns
        ]
        predicted_uses = [event for event in uses if event.get("prediction_hit") is True]
        on_time_uses = [
            event
            for event in predicted_uses
            if event.get("cache_hit") is True
            and isinstance(event.get("ready_ns"), int)
            and isinstance(event.get("use_ns"), int)
            and event["ready_ns"] <= event["use_ns"]
        ]
        blocked_ms = [
            event["blocked_ns"] / 1.0e6
            for event in uses
            if isinstance(event.get("blocked_ns"), int)
        ]
        actual = set(router["experts"])
        predicted_candidates = prediction_candidates_by_layer.get(layer, set())
        layer_rows.append(
            {
                "layer": layer,
                "router_token_count": router.get("token_count"),
                "actual_experts": len(actual),
                "actual_use_events": len(uses),
                "predicted_candidates": len(predicted_candidates),
                "correct_prediction_candidates": len(actual & predicted_candidates),
                "wrong_prediction_candidates": len(predicted_candidates - actual),
                "prediction_hit_uses": len(predicted_uses),
                "on_time_prediction_uses": len(on_time_uses),
                "prediction_reads": len(prediction_reads),
                "prediction_read_bytes": sum(
                    event.get("issued_bytes", 0) for event in prediction_reads
                ),
                "prediction_reads_ready_before_router": len(ready_before_router),
                "p0_read_events": sum(
                    event.get("request_class") == "p0-demand"
                    and event.get("predicted_probability") is None
                    for event in reads
                ),
                "promoted_prediction_read_events": sum(
                    event.get("request_class") == "p0-demand"
                    and event.get("predicted_probability") is not None
                    for event in reads
                ),
                "max_use_blocked_ms": max(blocked_ms) if blocked_ms else None,
            }
        )

    use_events = [event for values in uses_by_layer.values() for event in values]
    prediction_hits = [event for event in use_events if event.get("prediction_hit") is True]
    on_time_hits = [
        event
        for event in prediction_hits
        if event.get("cache_hit") is True
        and isinstance(event.get("ready_ns"), int)
        and isinstance(event.get("use_ns"), int)
        and event["ready_ns"] <= event["use_ns"]
    ]
    p0_reads = [
        event
        for values in reads_by_layer.values()
        for event in values
        if event.get("request_class") == "p0-demand"
        and event.get("predicted_probability") is None
    ]
    prediction_reads = [
        event
        for values in reads_by_layer.values()
        for event in values
        if event.get("predicted_probability") is not None
        or event.get("request_class") == "p2-speculative"
    ]
    promoted_prediction_reads = [
        event
        for event in prediction_reads
        if event.get("request_class") == "p0-demand"
    ]
    all_reads = [event for values in reads_by_layer.values() for event in values]
    actual_keys = {
        (event.get("layer_id"), event.get("logical_expert_id"))
        for event in use_events
    }
    wrong_prediction_reads = [
        event
        for event in prediction_reads
        if (event.get("layer_id"), event.get("logical_expert_id")) not in actual_keys
    ]
    ready_before_router_count = sum(
        row["prediction_reads_ready_before_router"] for row in layer_rows
    )
    predicted_candidate_count = sum(
        len(candidates) for candidates in prediction_candidates_by_layer.values()
    )
    correct_candidate_count = sum(
        len(prediction_candidates_by_layer.get(layer, set()) & set(router["experts"]))
        for layer, router in router_by_layer.items()
    )
    target_actual_count = sum(
        len(set(router_by_layer[layer]["experts"]))
        for layer in prediction_candidates_by_layer
        if layer in router_by_layer
    )
    token_weighted_target = 0
    token_weighted_covered = 0
    token_weighted_available = True
    for layer, candidates in prediction_candidates_by_layer.items():
        router = router_by_layer.get(layer)
        if router is None:
            continue
        experts = router.get("experts")
        counts = router.get("expert_token_counts")
        if (
            not isinstance(experts, list)
            or not isinstance(counts, list)
            or len(experts) != len(counts)
            or not all(isinstance(count, int) and count >= 0 for count in counts)
        ):
            token_weighted_available = False
            break
        per_expert = dict(zip(experts, counts))
        token_weighted_target += sum(counts)
        token_weighted_covered += sum(per_expert.get(expert, 0) for expert in candidates)
    predictor_candidate_bytes = sum(
        len(candidates) * prediction_expert_bytes_by_layer.get(layer, 0)
        for layer, candidates in prediction_candidates_by_layer.items()
    )
    predictor_correct_bytes = sum(
        len(candidates & set(router_by_layer[layer]["experts"]))
        * prediction_expert_bytes_by_layer.get(layer, 0)
        for layer, candidates in prediction_candidates_by_layer.items()
        if layer in router_by_layer
    )
    predictor_target_bytes = sum(
        len(set(router_by_layer[layer]["experts"]))
        * prediction_expert_bytes_by_layer.get(layer, 0)
        for layer in prediction_candidates_by_layer
        if layer in router_by_layer
    )
    budget_profiles = sorted(
        {
            (
                event.get("window_us"),
                event.get("bandwidth_mib_s"),
                event.get("utilization_pct"),
                event.get("budget_bytes"),
                event.get("expert_bytes"),
            )
            for event in predictor_submissions
        },
        key=lambda item: tuple(-1 if value is None else value for value in item),
    )
    full_demand_ready_layers = sum(
        row["actual_experts"] > 0
        and row["on_time_prediction_uses"] == row["actual_experts"]
        for row in layer_rows
    )
    performance = manifest.get("llama_performance", {})
    prompt_eval = performance.get("prompt_eval", {})
    return {
        "run_id": manifest.get("run_id"),
        "profile": manifest.get("profile"),
        "status": manifest.get("status"),
        "prompt_eval": prompt_eval,
        "load_ms": performance.get("load", {}).get("milliseconds"),
        "rss_peak_bytes": manifest.get("resource_summary", {})
        .get("process", {})
        .get("rss_peak_bytes"),
        "max_device_queue_depth": manifest.get("expert_io_summary", {}).get(
            "max_device_queue_depth"
        ),
        "p0_bandwidth": aggregate_bandwidth(p0_reads),
        "prediction_bandwidth": aggregate_bandwidth(prediction_reads),
        "promoted_prediction_bandwidth": aggregate_bandwidth(promoted_prediction_reads),
        "total_read_bandwidth": aggregate_bandwidth(all_reads),
        "read_amplification_over_actual_uses": (
            sum(event.get("issued_bytes", 0) for event in all_reads)
            / sum(event.get("useful_bytes", 0) for event in use_events)
            if use_events and sum(event.get("useful_bytes", 0) for event in use_events) > 0
            else None
        ),
        "expert_uses": len(use_events),
        "prediction_hit_uses": len(prediction_hits),
        "on_time_prediction_uses": len(on_time_hits),
        "prediction_hit_rate": len(prediction_hits) / len(use_events) if use_events else None,
        "on_time_prediction_hit_rate": (
            len(on_time_hits) / len(use_events) if use_events else None
        ),
        "ready_recall": len(on_time_hits) / len(use_events) if use_events else None,
        "late_prefetch_rate": (
            (len(prediction_hits) - len(on_time_hits)) / len(prediction_hits)
            if prediction_hits
            else None
        ),
        "full_demand_ready_layers": full_demand_ready_layers,
        "full_demand_ready_rate": (
            full_demand_ready_layers / len(layer_rows) if layer_rows else None
        ),
        "predictor_submissions": len(predictor_submissions),
        "predictor_requested_labels": requested_prediction_labels,
        "predictor_candidate_labels": predicted_candidate_count,
        "predictor_correct_labels": correct_candidate_count,
        "predictor_wrong_labels": predicted_candidate_count - correct_candidate_count,
        "predictor_precision": (
            correct_candidate_count / predicted_candidate_count
            if predicted_candidate_count
            else None
        ),
        "predictor_recall": (
            correct_candidate_count / target_actual_count if target_actual_count else None
        ),
        "predictor_token_weighted_covered": (
            token_weighted_covered if token_weighted_available else None
        ),
        "predictor_token_weighted_target": (
            token_weighted_target if token_weighted_available else None
        ),
        "predictor_token_weighted_coverage": (
            token_weighted_covered / token_weighted_target
            if token_weighted_available and token_weighted_target
            else None
        ),
        "predictor_candidate_bytes": predictor_candidate_bytes,
        "predictor_correct_bytes": predictor_correct_bytes,
        "predictor_target_bytes": predictor_target_bytes,
        "predictor_byte_precision": (
            predictor_correct_bytes / predictor_candidate_bytes
            if predictor_candidate_bytes
            else None
        ),
        "predictor_byte_recall": (
            predictor_correct_bytes / predictor_target_bytes
            if predictor_target_bytes
            else None
        ),
        "predictor_budget_profiles": [
            {
                "window_us": profile[0],
                "bandwidth_mib_s": profile[1],
                "utilization_pct": profile[2],
                "budget_bytes": profile[3],
                "expert_bytes": profile[4],
                "candidate_limit": (
                    profile[3] // profile[4]
                    if isinstance(profile[3], int)
                    and isinstance(profile[4], int)
                    and profile[4] > 0
                    else None
                ),
            }
            for profile in budget_profiles
        ],
        "wrong_prefetch_read_events": len(wrong_prediction_reads),
        "wrong_prefetch_bytes": sum(
            event.get("issued_bytes", 0) for event in wrong_prediction_reads
        ),
        "prediction_reads_ready_before_router": ready_before_router_count,
        "ready_before_router_but_not_on_time": max(
            0, ready_before_router_count - len(on_time_hits)
        ),
        "per_layer_max_use_blocked_ms": distribution(
            row["max_use_blocked_ms"]
            for row in layer_rows
            if row["max_use_blocked_ms"] is not None
        ),
        "layers": layer_rows,
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
