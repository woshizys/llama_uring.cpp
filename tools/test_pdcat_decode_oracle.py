#!/usr/bin/env python3
"""Unit tests for token-indexed decode oracle tooling."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.run_pdcat_decode_oracle_matrix import validate_decode_pair
from tools.summarize_pdcat_decode_oracle import summarize_run


class DecodeOracleToolingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )

    @staticmethod
    def manifest() -> dict[str, object]:
        return {
            "run_id": "decode",
            "profile": "cache_mixed",
            "status": "completed",
            "host": {"hostname": "jetson", "architecture": "aarch64"},
            "repositories": [{"path": "/repo", "commit": "abc", "dirty": False}],
            "runtime_artifacts": [
                {"path": "/repo/llama", "size_bytes": 1, "sha256": "a" * 64}
            ],
            "config_sha256": "b" * 64,
            "model": {"model": {"size_bytes": 10}, "topology": {"moe_layers": [1, 26]}},
            "model_sha256_expected": "c" * 64,
            "expert_pack": {"manifest_sha256": "d" * 64},
            "command": ["/repo/llama", "--n-predict", "2"],
            "configured_environment": {
                "LLAMA_MOE_CUDA_DELIVERY": "mapped",
                "PDCAT_IO_MAX_QD": "2",
            },
            "llama_performance": {
                "prompt_eval": {"count": 128, "milliseconds": 1000.0},
                "decode_eval": {
                    "count": 1,
                    "milliseconds": 100.0,
                    "milliseconds_per_token": 100.0,
                    "tokens_per_second": 10.0,
                },
            },
        }

    @staticmethod
    def routes() -> list[dict[str, object]]:
        return [
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_id": 0,
                "token_count": 128,
                "layer_id": 1,
                "experts": [3],
                "timestamp_ns": 5,
            },
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_id": 0,
                "token_count": 1,
                "layer_id": 26,
                "experts": [1],
                "timestamp_ns": 10,
            },
            {
                "event": "predictor_submission",
                "request_id": "r0",
                "phase": "decode",
                "token_id": 0,
                "token_count": 1,
                "layer_id": 26,
                "target_layer_id": 1,
                "experts": [2],
                "requested_predictions": 2,
                "expert_bytes": 100,
                "budget_bytes": 200,
                "window_us": 12000,
                "bandwidth_mib_s": 2048,
                "utilization_pct": 80,
            },
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_id": 1,
                "token_count": 1,
                "layer_id": 1,
                "experts": [2],
                "timestamp_ns": 20,
            },
            {
                "event": "router_selection",
                "request_id": "r0",
                "token_id": 1,
                "token_count": 1,
                "layer_id": 26,
                "experts": [4],
                "timestamp_ns": 30,
            },
        ]

    @staticmethod
    def io_events() -> list[dict[str, object]]:
        return [
            {
                "event": "expert_io",
                "operation": "read",
                "phase": "decode",
                "token_id": 1,
                "layer_id": 1,
                "logical_expert_id": 2,
                "request_class": "p2-speculative",
                "predicted_probability": 1.0,
                "submit_ns": 1,
                "blocked_ns": 0,
                "cqe_ns": 15,
                "ready_ns": 15,
                "issued_bytes": 100,
                "success": True,
            },
            {
                "event": "expert_use",
                "operation": "use",
                "phase": "decode",
                "token_id": 1,
                "layer_id": 1,
                "logical_expert_id": 2,
                "prediction_hit": True,
                "cache_hit": True,
                "ready_ns": 15,
                "use_ns": 25,
                "blocked_ns": 0,
                "useful_bytes": 100,
            },
        ]

    def make_run(self, name: str, manifest: dict[str, object] | None = None) -> Path:
        directory = self.root / name
        directory.mkdir()
        self.write_json(directory / "run_manifest.json", manifest or self.manifest())
        self.write_jsonl(directory / "moe_gate.jsonl", self.routes())
        self.write_jsonl(directory / "expert_io.jsonl", self.io_events())
        (directory / "stdout.log").write_bytes(b"deterministic\n")
        return directory

    def test_summary_reports_token_aware_oracle_metrics(self) -> None:
        run = self.make_run("run")
        summary = summarize_run(run)
        self.assertEqual(summary["router_tokens"], 2)
        self.assertEqual(summary["predictor_recall"], 1.0)
        self.assertEqual(summary["ready_recall"], 1.0)
        self.assertEqual(summary["prediction_reads_ready_before_router"], 1)
        self.assertEqual(summary["decode_tpot_ms"], 100.0)

    def test_pair_validation_requires_exact_token_routes_and_output(self) -> None:
        baseline = self.make_run("baseline")
        oracle_manifest = copy.deepcopy(self.manifest())
        oracle_manifest["profile"] = "cache_mixed_decode_oracle"
        oracle = self.make_run("oracle", oracle_manifest)
        result = validate_decode_pair(baseline, oracle, 128, 2)
        self.assertTrue(result["router_match"])
        self.assertEqual(result["router_tokens"], 2)
        self.assertEqual(result["router_positions"], 4)

        (oracle / "stdout.log").write_bytes(b"different\n")
        with self.assertRaisesRegex(ValueError, "generated output differs"):
            validate_decode_pair(baseline, oracle, 128, 2)


if __name__ == "__main__":
    unittest.main()
