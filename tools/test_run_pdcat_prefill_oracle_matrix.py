#!/usr/bin/env python3
"""Unit tests for reproducibility gates in the prefill oracle matrix."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.run_pdcat_experiment import runtime_artifact_snapshot
from tools.run_pdcat_prefill_oracle_matrix import PREFILL_WINDOW_US, validate_pair_artifacts


class PrefillOracleMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.baseline = self.root / "baseline"
        self.oracle = self.root / "oracle"
        self.baseline.mkdir()
        self.oracle.mkdir()
        self.manifest = {
            "host": {
                "hostname": "jetson",
                "architecture": "aarch64",
                "kernel": "test",
            },
            "repositories": [
                {"path": "/repo", "commit": "abc", "dirty": False}
            ],
            "runtime_artifacts": [
                {
                    "path": "/repo/bin/llama",
                    "size_bytes": 123,
                    "sha256": "a" * 64,
                }
            ],
            "config_sha256": "b" * 64,
            "model": {
                "model": {"size_bytes": 1000},
                "topology": {"moe_layers": [1, 2]},
            },
            "model_sha256_expected": "c" * 64,
            "expert_pack": {"manifest_sha256": "d" * 64},
            "command": ["/repo/bin/llama", "--prompt", "hello"],
            "configured_environment": {
                "LLAMA_MOE_CUDA_DELIVERY": "mapped",
                "PDCAT_IO_MAX_QD": "2",
                "PDCAT_PHASE": "prefill",
            },
        }
        self.write_pair(self.manifest, self.manifest)

    @staticmethod
    def router_records(experts: list[int] | None = None) -> str:
        records = [
            {
                "event": "router_selection",
                "request_id": "request-0",
                "phase": "prefill",
                "token_count": 128,
                "layer_id": layer,
                "experts": (
                    experts
                    if experts is not None and layer == 2
                    else [layer, layer + 1]
                ),
            }
            for layer in (1, 2)
        ]
        return "".join(json.dumps(record) + "\n" for record in records)

    def write_pair(
        self,
        baseline_manifest: dict[str, object],
        oracle_manifest: dict[str, object],
    ) -> None:
        for directory, manifest in (
            (self.baseline, baseline_manifest),
            (self.oracle, oracle_manifest),
        ):
            (directory / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (directory / "moe_gate.jsonl").write_text(
                self.router_records(), encoding="utf-8"
            )
            (directory / "stdout.log").write_bytes(b" token\n")

    def test_accepts_identical_runtime_routes_and_output(self) -> None:
        result = validate_pair_artifacts(self.baseline, self.oracle, 128)
        self.assertTrue(result["router_match"])
        self.assertTrue(result["generated_output_match"])
        self.assertEqual(result["router_layers"], 2)

    def test_rejects_runtime_artifact_mismatch(self) -> None:
        oracle_manifest = copy.deepcopy(self.manifest)
        oracle_manifest["runtime_artifacts"][0]["sha256"] = "e" * 64
        self.write_pair(self.manifest, oracle_manifest)
        with self.assertRaisesRegex(ValueError, "runtime_artifacts"):
            validate_pair_artifacts(self.baseline, self.oracle, 128)

    def test_rejects_router_mismatch(self) -> None:
        (self.oracle / "moe_gate.jsonl").write_text(
            self.router_records([9, 10]), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "router selections differ"):
            validate_pair_artifacts(self.baseline, self.oracle, 128)

    def test_rejects_generated_output_mismatch(self) -> None:
        (self.oracle / "stdout.log").write_bytes(b" different\n")
        with self.assertRaisesRegex(ValueError, "generated output differs"):
            validate_pair_artifacts(self.baseline, self.oracle, 128)

    def test_runtime_snapshot_hashes_the_selected_binary(self) -> None:
        binary = Path(shutil.which("true") or "/bin/true")
        artifacts = runtime_artifact_snapshot(binary)
        self.assertTrue(artifacts)
        self.assertEqual(artifacts[0]["path"], str(binary.resolve()))
        self.assertEqual(len(artifacts[0]["sha256"]), 64)

    def test_prompt_lengths_use_measured_prefill_windows(self) -> None:
        self.assertEqual(PREFILL_WINDOW_US, {128: 50_000, 256: 72_000})


if __name__ == "__main__":
    unittest.main()
