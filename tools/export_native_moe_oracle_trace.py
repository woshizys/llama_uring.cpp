#!/usr/bin/env python3
"""Convert a deterministic native/PDCat MoE trace into a route oracle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LAYER = re.compile(r"(?:^|#)blk\.(?P<layer>[0-9]+)\.")
EXPECTED_PARTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if raw_line.strip():
            records.append((line_number, json.loads(raw_line)))
    return records


def load_native_routes(
    path: Path,
    records: list[tuple[int, dict[str, Any]]],
    prompt_tokens: int | None,
) -> dict[int, list[int]]:
    routes: dict[int, list[int]] = {}
    parts: dict[int, set[str]] = {}
    for line_number, record in records:
        if record.get("event") != "native_moe_load_profile":
            continue
        if prompt_tokens is not None and record.get("tokens") != prompt_tokens:
            raise ValueError(
                f"line {line_number}: expected {prompt_tokens} tokens, "
                f"got {record.get('tokens')}"
            )
        tensor = record.get("tensor")
        expert_ids = record.get("expert_ids")
        match = LAYER.search(tensor) if isinstance(tensor, str) else None
        if match is None or not isinstance(expert_ids, list) or not expert_ids:
            raise ValueError(f"line {line_number}: invalid native MoE load record")
        if not all(isinstance(expert, int) and expert >= 0 for expert in expert_ids):
            raise ValueError(f"line {line_number}: invalid expert_ids")

        layer = int(match.group("layer"))
        ordered = sorted(set(expert_ids))
        previous = routes.setdefault(layer, ordered)
        if previous != ordered:
            raise ValueError(
                f"layer {layer}: gate/up/down records disagree on expert IDs"
            )
        part = next((name for name in EXPECTED_PARTS if name in tensor), None)
        if part is None:
            raise ValueError(f"line {line_number}: unsupported expert tensor {tensor}")
        parts.setdefault(layer, set()).add(part)

    if not routes:
        raise ValueError(f"{path}: no native_moe_load_profile records")
    for layer, seen_parts in parts.items():
        missing = set(EXPECTED_PARTS) - seen_parts
        if missing:
            raise ValueError(
                f"layer {layer}: missing expert tensor parts {sorted(missing)}"
            )
    return routes


def load_router_routes(
    path: Path,
    records: list[tuple[int, dict[str, Any]]],
    prompt_tokens: int | None,
) -> dict[int, list[int]]:
    routes: dict[int, list[int]] = {}
    request_ids: set[str] = set()
    last_full_record_index: int | None = None
    for line_number, record in records:
        if record.get("event") != "router_selection":
            continue
        token_count = record.get("token_count")
        if prompt_tokens is not None and token_count != prompt_tokens:
            continue
        if prompt_tokens is None and record.get("phase") != "prefill":
            continue

        layer = record.get("layer_id")
        expert_ids = record.get("experts")
        request_id = record.get("request_id")
        if (
            not isinstance(layer, int)
            or layer < 0
            or not isinstance(expert_ids, list)
            or not expert_ids
            or not all(isinstance(expert, int) and expert >= 0 for expert in expert_ids)
            or not isinstance(request_id, str)
            or not request_id
        ):
            raise ValueError(f"line {line_number}: invalid router_selection record")
        request_ids.add(request_id)
        last_full_record_index = line_number
        ordered = sorted(set(expert_ids))
        previous = routes.setdefault(layer, ordered)
        if previous != ordered:
            raise ValueError(f"layer {layer}: duplicate router records disagree")

    if not routes:
        raise ValueError(f"{path}: no prefill router_selection records")
    if len(request_ids) != 1:
        raise ValueError(
            f"{path}: oracle source must contain exactly one request, got {sorted(request_ids)}"
        )

    # llama.cpp may prune the final blocks to the single token whose logits are
    # requested and label those callbacks as decode. Append only the contiguous
    # higher-numbered suffix immediately after the full-prompt records; stop at
    # the first layer wrap so generated-token routes cannot enter the oracle.
    if prompt_tokens is not None and last_full_record_index is not None:
        request_id = next(iter(request_ids))
        max_layer = max(routes)
        for line_number, record in records:
            if line_number <= last_full_record_index:
                continue
            if record.get("event") != "router_selection":
                continue
            if record.get("request_id") != request_id or record.get("token_count") != 1:
                break
            layer = record.get("layer_id")
            expert_ids = record.get("experts")
            if not isinstance(layer, int) or layer <= max_layer:
                break
            if not isinstance(expert_ids, list) or not expert_ids or not all(
                isinstance(expert, int) and expert >= 0 for expert in expert_ids
            ):
                raise ValueError(f"line {line_number}: invalid router_selection record")
            routes[layer] = sorted(set(expert_ids))
            max_layer = layer
    return routes


def load_routes(path: Path, prompt_tokens: int | None) -> dict[int, list[int]]:
    records = read_records(path)
    events = {record.get("event") for _, record in records}
    if "native_moe_load_profile" in events:
        return load_native_routes(path, records, prompt_tokens)
    if "router_selection" in events:
        return load_router_routes(path, records, prompt_tokens)
    raise ValueError(
        f"{path}: expected native_moe_load_profile or router_selection records"
    )


def export_routes(
    routes: dict[int, list[int]],
    output: Path,
    request_id: str,
    model_prefix: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for layer, experts in sorted(routes.items()):
            record = {
                "request_id": request_id,
                "layer": layer,
                "experts": [
                    f"{model_prefix}/l{layer}-e{expert}" for expert in experts
                ],
            }
            destination.write(json.dumps(record, sort_keys=True) + "\n")


def load_token_router_routes(path: Path) -> list[tuple[int, int, list[int]]]:
    """Load an ordered, token-aware router trace for strict decode replay.

    The result includes the prefill generation (normally token 0), the
    single-token tail of that generation, and all subsequent decode forwards.
    A missing token_id or a layer wrap without a token increment is rejected:
    either would collapse distinct decode states into a non-strict oracle.
    """

    routes: list[tuple[int, int, list[int]]] = []
    request_ids: set[str] = set()
    last_position: tuple[int, int] | None = None
    last_experts: list[int] | None = None
    for line_number, record in read_records(path):
        if record.get("event") != "router_selection":
            continue
        request_id = record.get("request_id")
        token_id = record.get("token_id")
        layer = record.get("layer_id")
        expert_ids = record.get("experts")
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or not isinstance(layer, int)
            or isinstance(layer, bool)
            or layer < 0
            or not isinstance(expert_ids, list)
            or not expert_ids
            or not all(
                isinstance(expert, int)
                and not isinstance(expert, bool)
                and expert >= 0
                for expert in expert_ids
            )
        ):
            raise ValueError(
                f"line {line_number}: decode oracle requires valid request_id, "
                "token_id, layer_id, and experts"
            )
        request_ids.add(request_id)
        position = (token_id, layer)
        experts = sorted(set(expert_ids))
        if last_position is not None and position < last_position:
            raise ValueError(
                f"line {line_number}: router position {position} follows "
                f"{last_position}; token/layer order is not monotonic"
            )
        if position == last_position:
            if experts != last_experts:
                raise ValueError(
                    f"line {line_number}: duplicate router position {position} disagrees"
                )
            continue
        routes.append((token_id, layer, experts))
        last_position = position
        last_experts = experts

    if not routes:
        raise ValueError(f"{path}: no token-aware router_selection records")
    if len(request_ids) != 1:
        raise ValueError(
            f"{path}: oracle source must contain exactly one request, got {sorted(request_ids)}"
        )
    return routes


def export_token_routes(
    routes: list[tuple[int, int, list[int]]],
    output: Path,
    request_id: str,
    model_prefix: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for token_id, layer, experts in routes:
            record = {
                "request_id": request_id,
                "token_id": token_id,
                "layer": layer,
                "experts": [
                    f"{model_prefix}/l{layer}-e{expert}" for expert in experts
                ],
            }
            destination.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--request-id", default="native-oracle-source")
    parser.add_argument("--model-prefix", default="runtime")
    parser.add_argument(
        "--token-aware",
        action="store_true",
        help="export the complete ordered prefill+decode trace with token_id",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.token_aware:
        routes = load_token_router_routes(args.input.resolve())
        export_token_routes(
            routes,
            args.output.resolve(),
            args.request_id,
            args.model_prefix,
        )
        print(
            f"exported {len(routes)} token/layer routes to {args.output} "
            f"with {sum(len(experts) for _, _, experts in routes)} expert labels"
        )
    else:
        routes = load_routes(args.input.resolve(), args.prompt_tokens)
        export_routes(
            routes,
            args.output.resolve(),
            args.request_id,
            args.model_prefix,
        )
        print(
            f"exported {len(routes)} layers to {args.output} "
            f"with {sum(len(experts) for experts in routes.values())} expert labels"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
