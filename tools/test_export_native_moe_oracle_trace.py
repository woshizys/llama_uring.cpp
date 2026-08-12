#!/usr/bin/env python3
"""Unit tests for export_native_moe_oracle_trace.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.export_native_moe_oracle_trace import (
    export_token_routes,
    load_routes,
    load_token_router_routes,
)


class OracleTraceExportTest(unittest.TestCase):
    def write_jsonl(self, records: list[dict[str, object]]) -> Path:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_router_trace_keeps_single_token_final_layer(self) -> None:
        path = self.write_jsonl(
            [
                {
                    "event": "router_selection",
                    "phase": "decode",
                    "request_id": "r0",
                    "token_count": 128,
                    "layer_id": 1,
                    "experts": [7, 2, 7],
                },
                {
                    "event": "router_selection",
                    "phase": "prefill",
                    "request_id": "r0",
                    "token_count": 1,
                    "layer_id": 2,
                    "experts": [9],
                },
                {
                    "event": "router_selection",
                    "phase": "decode",
                    "request_id": "r0",
                    "token_count": 1,
                    "layer_id": 1,
                    "experts": [63],
                },
            ]
        )
        self.assertEqual(load_routes(path, 128), {1: [2, 7], 2: [9]})

    def test_router_trace_rejects_multiple_requests(self) -> None:
        path = self.write_jsonl(
            [
                {
                    "event": "router_selection",
                    "phase": "prefill",
                    "request_id": request_id,
                    "token_count": 128,
                    "layer_id": layer,
                    "experts": [layer],
                }
                for layer, request_id in [(1, "r0"), (2, "r1")]
            ]
        )
        with self.assertRaisesRegex(ValueError, "exactly one request"):
            load_routes(path, 128)

    def test_token_router_trace_preserves_decode_occurrences_and_wrap(self) -> None:
        path = self.write_jsonl(
            [
                {
                    "event": "router_selection",
                    "request_id": "r0",
                    "token_id": token_id,
                    "token_count": 1,
                    "layer_id": layer,
                    "experts": experts,
                }
                for token_id, layer, experts in (
                    (0, 1, [7, 2, 7]),
                    (0, 2, [9]),
                    (1, 1, [7, 2]),
                    (1, 2, [8]),
                )
            ]
        )
        routes = load_token_router_routes(path)
        self.assertEqual(
            routes,
            [(0, 1, [2, 7]), (0, 2, [9]), (1, 1, [2, 7]), (1, 2, [8])],
        )

        output = path.parent / "oracle.jsonl"
        export_token_routes(routes, output, "source", "runtime")
        exported = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(exported[2]["token_id"], 1)
        self.assertEqual(exported[2]["experts"], ["runtime/l1-e2", "runtime/l1-e7"])

    def test_token_router_trace_rejects_unlabelled_decode(self) -> None:
        path = self.write_jsonl(
            [
                {
                    "event": "router_selection",
                    "request_id": "r0",
                    "token_id": None,
                    "layer_id": 1,
                    "experts": [2],
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires valid"):
            load_token_router_routes(path)

    def test_token_router_trace_rejects_layer_wrap_without_token_increment(self) -> None:
        path = self.write_jsonl(
            [
                {
                    "event": "router_selection",
                    "request_id": "r0",
                    "token_id": 0,
                    "layer_id": layer,
                    "experts": [layer],
                }
                for layer in (2, 1)
            ]
        )
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            load_token_router_routes(path)


if __name__ == "__main__":
    unittest.main()
