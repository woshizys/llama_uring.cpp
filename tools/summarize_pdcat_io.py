#!/usr/bin/env python3
"""Summarize unified PDCat expert/KV I/O JSONL without changing the trace."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class TraceError(ValueError):
    """Raised when a PDCat I/O trace is malformed."""


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceError(f"{path}:{line_number}: {error}") from error
            if not isinstance(event, dict):
                raise TraceError(f"{path}:{line_number}: event must be an object")
            for field in (
                "event",
                "operation",
                "request_class",
                "submit_ns",
                "ready_ns",
                "blocked_ns",
                "useful_bytes",
                "issued_bytes",
                "queue_depth",
                "device_queue_depth",
                "in_flight_bytes",
                "success",
            ):
                if field not in event:
                    raise TraceError(f"{path}:{line_number}: missing {field}")
            events.append(event)
    return events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_class[str(event["request_class"])].append(event)
        by_operation[str(event["operation"])].append(event)

    class_summaries: dict[str, Any] = {}
    for request_class, rows in sorted(by_class.items()):
        latency_us = [
            max(0, int(row["ready_ns"]) - int(row["submit_ns"])) / 1000.0
            for row in rows
        ]
        queue_us = [max(0, int(row["blocked_ns"])) / 1000.0 for row in rows]
        service_us = [
            max(
                0,
                int(row["ready_ns"])
                - int(row["submit_ns"])
                - int(row["blocked_ns"]),
            )
            / 1000.0
            for row in rows
        ]
        useful_bytes = sum(int(row["useful_bytes"]) for row in rows)
        issued_bytes = sum(int(row["issued_bytes"]) for row in rows)
        class_summaries[request_class] = {
            "events": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "useful_bytes": useful_bytes,
            "issued_bytes": issued_bytes,
            "io_amplification": issued_bytes / useful_bytes if useful_bytes else None,
            "latency_us": distribution(latency_us),
            "queue_delay_us": distribution(queue_us),
            "service_us": distribution(service_us),
        }

    operation_summaries: dict[str, Any] = {}
    for operation, rows in sorted(by_operation.items()):
        latency_us = [
            max(0, int(row["ready_ns"]) - int(row["submit_ns"])) / 1000.0
            for row in rows
        ]
        operation_summaries[operation] = {
            "events": len(rows),
            "successes": sum(bool(row["success"]) for row in rows),
            "useful_bytes": sum(int(row["useful_bytes"]) for row in rows),
            "issued_bytes": sum(int(row["issued_bytes"]) for row in rows),
            "latency_us": distribution(latency_us),
        }

    expert_events = [event for event in events if event["event"] == "expert_io"]
    expert_use_events = [event for event in events if event["event"] == "expert_use"]
    kv_events = [event for event in events if event["event"] == "kv_io"]
    io_events = expert_events + kv_events
    deadline_events = [
        event
        for event in expert_events
        if event.get("predicted_deadline_ns") is not None
    ]
    ready_events = [event for event in deadline_events if event.get("ready_before_deadline")]
    predicted_events = [
        event
        for event in expert_events
        if event["request_class"] == "p2-speculative"
        or event.get("predicted_probability") is not None
    ]
    predictor_latency_by_invocation: dict[tuple[Any, ...], float] = {}
    for event in predicted_events:
        raw_latency = event.get("predictor_us")
        if raw_latency is None:
            continue
        latency = float(raw_latency)
        if not math.isfinite(latency) or latency < 0:
            raise TraceError("predictor_us must be finite and non-negative")
        key = (
            event.get("run_id"),
            event.get("request_id"),
            event.get("token_id"),
            event.get("layer_id"),
        )
        previous = predictor_latency_by_invocation.setdefault(key, latency)
        if not math.isclose(previous, latency, rel_tol=1.0e-9, abs_tol=1.0e-6):
            raise TraceError(
                "one predictor invocation has inconsistent predictor_us values"
            )
    predictor_latency_us = list(predictor_latency_by_invocation.values())
    prediction_hits = [
        event for event in expert_use_events if event.get("prediction_hit")
    ]
    prediction_hit_keys = {
        (
            event.get("request_id"),
            event.get("token_id"),
            event.get("object_id"),
        )
        for event in prediction_hits
    }
    wrong_prefetch_events = [
        event
        for event in predicted_events
        if (
            event.get("request_id"),
            event.get("token_id"),
            event.get("object_id"),
        )
        not in prediction_hit_keys
    ]
    ready_to_use_us = [
        max(0, int(event["use_ns"]) - int(event["ready_ns"])) / 1000.0
        for event in expert_use_events
        if event.get("use_ns") is not None
    ]
    useful_bytes = sum(int(event["useful_bytes"]) for event in io_events)
    issued_bytes = sum(int(event["issued_bytes"]) for event in io_events)
    expert_useful_bytes = sum(int(event["useful_bytes"]) for event in expert_events)
    expert_issued_bytes = sum(int(event["issued_bytes"]) for event in expert_events)
    kv_writes = [event for event in kv_events if event["operation"] == "write"]
    kv_syncs = [
        event for event in kv_events if event["operation"] in ("fdatasync", "fsync")
    ]

    return {
        "schema_version": 1,
        "event_count": len(events),
        "event_counts": {
            event_type: sum(event["event"] == event_type for event in events)
            for event_type in sorted({str(event["event"]) for event in events})
        },
        "operation_counts": {
            operation: len(rows) for operation, rows in sorted(by_operation.items())
        },
        "run_ids": sorted({str(event.get("run_id", "")) for event in events}),
        "request_ids": sorted(
            {str(event.get("request_id", "")) for event in events}
        ),
        "successes": sum(bool(event["success"]) for event in events),
        "failures": sum(not bool(event["success"]) for event in events),
        "useful_bytes": useful_bytes,
        "issued_bytes": issued_bytes,
        "io_amplification": issued_bytes / useful_bytes if useful_bytes else None,
        "read_amplification": (
            expert_issued_bytes / expert_useful_bytes
            if expert_useful_bytes
            else None
        ),
        "expert_io": {
            "events": len(expert_events),
            "useful_bytes": expert_useful_bytes,
            "issued_bytes": expert_issued_bytes,
        },
        "expert_use": {
            "events": len(expert_use_events),
            "cache_hits": sum(bool(event.get("cache_hit")) for event in expert_use_events),
            "cache_hit_rate": (
                sum(bool(event.get("cache_hit")) for event in expert_use_events)
                / len(expert_use_events)
                if expert_use_events
                else None
            ),
            "prediction_hits": len(prediction_hits),
            "prediction_hit_rate_over_uses": (
                len(prediction_hits) / len(expert_use_events)
                if expert_use_events
                else None
            ),
            "direct_uses": sum(
                bool(event.get("direct_use")) for event in expert_use_events
            ),
            "promoted_uses": sum(
                bool(event.get("promoted")) for event in expert_use_events
            ),
            "ready_to_use_us": distribution(ready_to_use_us),
        },
        "kv_io": {
            "events": len(kv_events),
            "write_events": len(kv_writes),
            "write_bytes": sum(int(event["issued_bytes"]) for event in kv_writes),
            "sync_events": len(kv_syncs),
            "durable_successes": sum(
                bool(event.get("durable")) and bool(event["success"])
                for event in kv_syncs
            ),
        },
        "max_queue_depth": max(
            (int(event["queue_depth"]) for event in events), default=0
        ),
        "max_device_queue_depth": max(
            (int(event["device_queue_depth"]) for event in events), default=0
        ),
        "max_in_flight_bytes": max(
            (int(event["in_flight_bytes"]) for event in events), default=0
        ),
        "deadline_events": len(deadline_events),
        "ready_before_deadline": len(ready_events),
        "ready_recall": len(ready_events) / len(deadline_events)
        if deadline_events
        else None,
        "late_prefetch_events": len(deadline_events) - len(ready_events),
        "prediction_io_events": len(predicted_events),
        "predictor_invocations_traced": len(predictor_latency_us),
        "predictor_latency_us": distribution(predictor_latency_us),
        "prediction_hits_visible_at_actual_use": len(prediction_hits),
        "wrong_prefetch_io_events_correlated": len(wrong_prefetch_events),
        "wrong_prefetch_bytes_correlated": sum(
            int(event["issued_bytes"]) for event in wrong_prefetch_events
        ),
        "by_class": class_summaries,
        "by_operation": operation_summaries,
    }


def summarize_jsonl(path: Path) -> dict[str, Any]:
    return summarize_events(read_events(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = summarize_jsonl(args.trace)
    except (OSError, TraceError) as error:
        print(f"trace error: {error}", file=__import__("sys").stderr)
        return 2
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
