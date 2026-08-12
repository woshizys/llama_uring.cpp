#!/usr/bin/env python3
"""Estimate a phase-aware prefill prefetch budget from an exact route trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .export_native_moe_oracle_trace import load_routes
except ImportError:
    from export_native_moe_oracle_trace import load_routes


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER = re.compile(r"(?:^|#)blk\.(?P<layer>[0-9]+)\.")
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/pdcat/configs/deepseek-v2-lite-q8_0.json"
)


def load_route_counts(path: Path, prompt_tokens: int) -> dict[int, list[int]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(record.get("event") == "router_selection" for record in records):
        return load_routes(path, prompt_tokens)

    counts: dict[int, int] = {}
    for record in records:
        if record.get("event") != "native_moe_load_profile":
            continue
        if record.get("tokens") != prompt_tokens:
            raise ValueError(
                f"expected {prompt_tokens} tokens, got {record.get('tokens')}"
            )
        tensor = record.get("tensor")
        active_experts = record.get("active_experts")
        match = LAYER.search(tensor) if isinstance(tensor, str) else None
        if match is None or not isinstance(active_experts, int) or active_experts <= 0:
            raise ValueError("invalid native MoE load count record")
        layer = int(match.group("layer"))
        previous = counts.setdefault(layer, active_experts)
        if previous != active_experts:
            raise ValueError(f"layer {layer}: native tensor parts disagree on count")
    if not counts:
        raise ValueError(f"{path}: no route or native MoE load counts")
    return {layer: list(range(count)) for layer, count in counts.items()}


def estimate_budget(
    routes: dict[int, list[int]],
    *,
    expert_bytes: int,
    window_us: int,
    bandwidth_mib_s: int,
    utilization_pct: int,
    requested_top_n: int,
) -> dict[str, Any]:
    if not routes:
        raise ValueError("route trace is empty")
    for name, value in (
        ("expert_bytes", expert_bytes),
        ("window_us", window_us),
        ("bandwidth_mib_s", bandwidth_mib_s),
        ("requested_top_n", requested_top_n),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if utilization_pct <= 0 or utilization_pct > 100:
        raise ValueError("utilization_pct must be in 1..100")

    budget_bytes = int(
        bandwidth_mib_s
        * 1024
        * 1024
        * window_us
        / 1_000_000
        * utilization_pct
        / 100
    )
    candidate_limit = min(requested_top_n, budget_bytes // expert_bytes)
    ordered_layers = sorted(routes)
    target_layers = ordered_layers[1:]
    layer_rows = []
    for layer in target_layers:
        actual = len(set(routes[layer]))
        candidates = min(actual, candidate_limit)
        layer_rows.append(
            {
                "layer": layer,
                "actual_experts": actual,
                "candidate_limit": candidate_limit,
                "budgeted_candidates": candidates,
                "upper_bound_ready_recall": candidates / actual if actual else None,
                "prefetch_bytes": candidates * expert_bytes,
                "residual_demand_bytes": (actual - candidates) * expert_bytes,
            }
        )

    actual_labels = sum(len(set(experts)) for experts in routes.values())
    predictable_labels = sum(row["actual_experts"] for row in layer_rows)
    candidate_labels = sum(row["budgeted_candidates"] for row in layer_rows)
    prefetch_bytes = candidate_labels * expert_bytes
    total_demand_bytes = actual_labels * expert_bytes
    return {
        "schema_version": 1,
        "route_layers": len(routes),
        "actual_expert_labels": actual_labels,
        "predictable_expert_labels": predictable_labels,
        "requested_top_n": requested_top_n,
        "window_us": window_us,
        "bandwidth_mib_s": bandwidth_mib_s,
        "utilization_pct": utilization_pct,
        "budget_bytes": budget_bytes,
        "expert_bytes": expert_bytes,
        "candidate_limit": candidate_limit,
        "budgeted_candidate_labels": candidate_labels,
        "upper_bound_recall_over_predictable": (
            candidate_labels / predictable_labels if predictable_labels else None
        ),
        "upper_bound_recall_over_all_uses": candidate_labels / actual_labels,
        "prefetch_bytes": prefetch_bytes,
        "total_demand_bytes": total_demand_bytes,
        "residual_demand_bytes": total_demand_bytes - prefetch_bytes,
        "ideal_read_amplification": 1.0,
        "layers": layer_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="cache_mixed_prefill_oracle")
    parser.add_argument("--window-us", type=int)
    parser.add_argument("--bandwidth-mib-s", type=int)
    parser.add_argument("--utilization-pct", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    topology = config["model"]["expected_topology"]
    environment = config["profiles"][args.profile]["environment"]
    result = estimate_budget(
        load_route_counts(args.routes.resolve(), args.prompt_tokens),
        expert_bytes=int(topology["expert_object_bytes"]),
        window_us=(
            args.window_us
            if args.window_us is not None
            else int(environment["LLAMA_MOE_PREFILL_PREFETCH_WINDOW_US"])
        ),
        bandwidth_mib_s=(
            args.bandwidth_mib_s
            if args.bandwidth_mib_s is not None
            else int(environment["LLAMA_MOE_PREFETCH_BANDWIDTH_MIB_S"])
        ),
        utilization_pct=(
            args.utilization_pct
            if args.utilization_pct is not None
            else int(environment["LLAMA_MOE_PREFETCH_BANDWIDTH_UTILIZATION_PCT"])
        ),
        requested_top_n=int(environment["LLAMA_MOE_PREDICTOR_TOP_N"]),
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
