#!/usr/bin/env python3
"""Validate llama MoE router-to-slot Gate A JSONL traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be a JSON object")
            event["_line"] = line_number
            events.append(event)
    events.sort(key=lambda event: (event.get("sequence", -1), event["_line"]))
    return events


def validate(
    events: list[dict[str, Any]],
    min_router_events: int,
    min_slot_reuses: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    current_route: dict[int, set[int]] = {}
    pending: dict[int, set[int]] = {}
    slot_owner: dict[tuple[int, int], tuple[int, int]] = {}
    latest_generation: dict[int, int] = {}
    router_events = 0
    slot_events = 0
    slot_reuses = 0
    seen_sequences: set[int] = set()
    experts_by_layer: dict[int, set[int]] = defaultdict(set)

    def error(event: dict[str, Any], message: str) -> None:
        if len(errors) < 100:
            errors.append(f"line {event.get('_line', '?')}: {message}")

    for event in events:
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            error(event, "missing or invalid sequence")
        elif sequence in seen_sequences:
            error(event, f"duplicate sequence {sequence}")
        else:
            seen_sequences.add(sequence)

        kind = event.get("event")
        layer = event.get("layer_id")
        if not isinstance(layer, int) or layer < 0:
            error(event, "missing or invalid layer_id")
            continue

        if kind == "router_selection":
            router_events += 1
            experts = event.get("experts")
            if not isinstance(experts, list) or not experts:
                error(event, "router_selection requires a non-empty experts array")
                continue
            if not all(isinstance(expert, int) and expert >= 0 for expert in experts):
                error(event, "experts must contain non-negative integers")
                continue
            if len(set(experts)) != len(experts):
                error(event, "experts contains duplicates")
            if pending.get(layer):
                error(
                    event,
                    f"new route arrived before all prior experts were ready: {sorted(pending[layer])}",
                )
            current_route[layer] = set(experts)
            pending[layer] = set(experts)
            continue

        if kind != "expert_slot_ready":
            error(event, f"unsupported event kind {kind!r}")
            continue

        slot_events += 1
        expert = event.get("logical_expert_id")
        slot = event.get("physical_slot")
        generation = event.get("slot_generation")
        if not isinstance(expert, int) or expert < 0:
            error(event, "missing or invalid logical_expert_id")
            continue
        if not isinstance(slot, int) or slot < 0:
            error(event, "missing or invalid physical_slot")
            continue
        if not isinstance(generation, int) or generation <= 0:
            error(event, "missing or invalid slot_generation")
            continue

        if expert not in current_route.get(layer, set()):
            error(event, f"expert {expert} was not selected by the current layer route")
        pending.setdefault(layer, set()).discard(expert)
        experts_by_layer[layer].add(expert)

        owner_key = (slot, generation)
        owner = (layer, expert)
        prior_owner = slot_owner.get(owner_key)
        if prior_owner is not None and prior_owner != owner:
            error(
                event,
                f"slot {slot} generation {generation} changed owner from {prior_owner} to {owner}",
            )
        slot_owner[owner_key] = owner

        previous_generation = latest_generation.get(slot)
        if previous_generation is not None:
            if generation < previous_generation:
                error(
                    event,
                    f"slot {slot} generation regressed from {previous_generation} to {generation}",
                )
            elif generation > previous_generation:
                slot_reuses += 1
        latest_generation[slot] = max(generation, previous_generation or 0)

    for layer, missing in sorted(pending.items()):
        if missing:
            errors.append(f"end of trace: layer {layer} never produced ready slots for {sorted(missing)}")

    if router_events < min_router_events:
        errors.append(f"router event count {router_events} is below required {min_router_events}")
    if slot_reuses < min_slot_reuses:
        errors.append(f"slot reuse count {slot_reuses} is below required {min_slot_reuses}")

    summary = {
        "events": len(events),
        "router_events": router_events,
        "slot_ready_events": slot_events,
        "physical_slots": len(latest_generation),
        "slot_reuses": slot_reuses,
        "layers": len(experts_by_layer),
        "distinct_logical_experts": sum(len(experts) for experts in experts_by_layer.values()),
        "valid": not errors,
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--min-router-events", type=int, default=1)
    parser.add_argument("--min-slot-reuses", type=int, default=0)
    args = parser.parse_args()

    try:
        events = load_events(args.trace)
        summary, errors = validate(events, args.min_router_events, args.min_slot_reuses)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    for message in errors:
        print(f"ERROR: {message}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
