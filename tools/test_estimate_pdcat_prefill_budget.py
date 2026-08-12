#!/usr/bin/env python3
"""Unit tests for model-independent prefill bandwidth budgeting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.estimate_pdcat_prefill_budget import estimate_budget, load_route_counts


class PrefillBudgetEstimateTest(unittest.TestCase):
    def test_budget_caps_each_predictable_layer(self) -> None:
        result = estimate_budget(
            {1: list(range(3)), 2: list(range(12)), 3: list(range(6))},
            expert_bytes=100,
            window_us=1_000,
            bandwidth_mib_s=1,
            utilization_pct=100,
            requested_top_n=64,
        )
        self.assertEqual(result["candidate_limit"], 10)
        self.assertEqual(result["budgeted_candidate_labels"], 16)
        self.assertEqual(result["actual_expert_labels"], 21)
        self.assertEqual(result["predictable_expert_labels"], 18)
        self.assertEqual(result["prefetch_bytes"], 1_600)
        self.assertEqual(result["residual_demand_bytes"], 500)

    def test_rejects_zero_sized_expert(self) -> None:
        with self.assertRaisesRegex(ValueError, "expert_bytes"):
            estimate_budget(
                {1: [0], 2: [1]},
                expert_bytes=0,
                window_us=1,
                bandwidth_mib_s=1,
                utilization_pct=80,
                requested_top_n=1,
            )

    def test_loads_legacy_native_active_expert_counts(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trace = directory / "native.jsonl"
        records = [
            {
                "event": "native_moe_load_profile",
                "tokens": 256,
                "tensor": f"blk.1.ffn_{part}_exps.weight",
                "active_experts": 17,
            }
            for part in ("gate", "up", "down")
        ]
        trace.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        self.assertEqual(len(load_route_counts(trace, 256)[1]), 17)


if __name__ == "__main__":
    unittest.main()
