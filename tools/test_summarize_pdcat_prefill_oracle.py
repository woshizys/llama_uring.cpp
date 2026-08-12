#!/usr/bin/env python3
"""Unit tests for prefill oracle prediction and readiness metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_pdcat_prefill_oracle import summarize_run


class PrefillOracleSummaryTest(unittest.TestCase):
    def test_prediction_quality_budget_and_wrong_bytes(self) -> None:
        run_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        manifest = {
            "run_id": "synthetic",
            "profile": "cache_mixed_prefill_oracle",
            "status": "completed",
            "llama_performance": {
                "prompt_eval": {"count": 128, "milliseconds": 10.0},
                "load": {"milliseconds": 1.0},
            },
            "resource_summary": {"process": {"rss_peak_bytes": 100}},
            "expert_io_summary": {"max_device_queue_depth": 2},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        gate_events = [
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_count": 128,
                "layer_id": 1,
                "experts": [1, 2],
                "expert_token_counts": [50, 50],
            },
            {
                "event": "predictor_submission",
                "request_id": "r0",
                "phase": "prefill",
                "token_count": 128,
                "layer_id": 1,
                "target_layer_id": 2,
                "experts": [3, 9],
                "requested_predictions": 64,
                "window_us": 50_000,
                "bandwidth_mib_s": 2_048,
                "utilization_pct": 80,
                "budget_bytes": 900,
                "expert_bytes": 100,
            },
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_count": 128,
                "layer_id": 2,
                "experts": [3, 4],
                "expert_token_counts": [70, 30],
            },
        ]
        (run_dir / "moe_gate.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in gate_events),
            encoding="utf-8",
        )
        io_events = []
        for layer, experts in ((1, (1, 2)), (2, (3, 4))):
            for expert in experts:
                predicted = layer == 2 and expert == 3
                io_events.append(
                    {
                        "event": "expert_use",
                        "token_id": 0,
                        "layer_id": layer,
                        "logical_expert_id": expert,
                        "blocked_ns": 1,
                        "useful_bytes": 100,
                        "prediction_hit": predicted,
                        "cache_hit": predicted,
                        "ready_ns": 5,
                        "use_ns": 10,
                    }
                )
        for expert in (3, 9):
            io_events.append(
                {
                    "event": "expert_io",
                    "operation": "read",
                    "success": True,
                    "token_id": 0,
                    "layer_id": 2,
                    "logical_expert_id": expert,
                    "request_class": "p2-speculative",
                    "predicted_probability": 0.5,
                    "submit_ns": 0,
                    "blocked_ns": 0,
                    "cqe_ns": 5,
                    "ready_ns": 5,
                    "issued_bytes": 100,
                }
            )
        (run_dir / "expert_io.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in io_events),
            encoding="utf-8",
        )

        summary = summarize_run(run_dir)
        self.assertEqual(summary["predictor_submissions"], 1)
        self.assertEqual(summary["predictor_precision"], 0.5)
        self.assertEqual(summary["predictor_recall"], 0.5)
        self.assertEqual(summary["predictor_byte_precision"], 0.5)
        self.assertEqual(summary["predictor_byte_recall"], 0.5)
        self.assertEqual(summary["predictor_token_weighted_coverage"], 0.7)
        self.assertEqual(summary["wrong_prefetch_bytes"], 100)
        self.assertEqual(summary["ready_recall"], 0.25)
        self.assertEqual(
            summary["predictor_budget_profiles"][0]["candidate_limit"], 9
        )


if __name__ == "__main__":
    unittest.main()
