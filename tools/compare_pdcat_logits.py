#!/usr/bin/env python3
"""Compare two raw float32 logits vectors and emit a Gate A JSON summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_f32(path: Path) -> array:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if len(payload) % 4 != 0:
        raise ValueError(f"{path}: byte length {len(payload)} is not divisible by 4")
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def top_indices(values: array, count: int) -> list[int]:
    return sorted(
        range(len(values)),
        key=lambda index: (
            -math.inf if math.isnan(values[index]) else values[index],
            -index,
        ),
        reverse=True,
    )[:count]


def compare(reference: array, candidate: array, top_k: int) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError(
            f"logits length mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )
    if not reference:
        raise ValueError("logits vectors are empty")
    top_k = min(top_k, len(reference))

    abs_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    max_abs_index = -1
    exact = 0
    finite_pairs = 0
    nonfinite_equal = 0
    nonfinite_mismatch = 0
    dot = 0.0
    norm_reference = 0.0
    norm_candidate = 0.0

    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left == right:
            exact += 1
            if math.isfinite(left):
                error = 0.0
                finite_pairs += 1
                dot += left * right
                norm_reference += left * left
                norm_candidate += right * right
            else:
                nonfinite_equal += 1
            continue
        if not math.isfinite(left) or not math.isfinite(right):
            nonfinite_mismatch += 1
            continue

        error = abs(left - right)
        finite_pairs += 1
        abs_sum += error
        squared_sum += error * error
        dot += left * right
        norm_reference += left * left
        norm_candidate += right * right
        if error > max_abs:
            max_abs = error
            max_abs_index = index

    reference_top = top_indices(reference, top_k)
    candidate_top = top_indices(candidate, top_k)
    reference_top1 = reference_top[0]
    candidate_top1 = candidate_top[0]
    reference_margin = (
        float(reference[reference_top[0]] - reference[reference_top[1]])
        if len(reference_top) > 1
        else math.inf
    )
    candidate_margin = (
        float(candidate[candidate_top[0]] - candidate[candidate_top[1]])
        if len(candidate_top) > 1
        else math.inf
    )
    denominator = math.sqrt(norm_reference) * math.sqrt(norm_candidate)

    return {
        "count": len(reference),
        "exact_values": exact,
        "exact_fraction": exact / len(reference),
        "finite_pairs": finite_pairs,
        "nonfinite_equal": nonfinite_equal,
        "nonfinite_mismatch": nonfinite_mismatch,
        "max_abs_error": max_abs if nonfinite_mismatch == 0 else math.inf,
        "max_abs_error_index": max_abs_index,
        "mean_abs_error": abs_sum / finite_pairs if finite_pairs else 0.0,
        "rmse": math.sqrt(squared_sum / finite_pairs) if finite_pairs else 0.0,
        "cosine_similarity": dot / denominator if denominator else None,
        "top_k": top_k,
        "reference_top_k": reference_top,
        "candidate_top_k": candidate_top,
        "top_k_exact_order": reference_top == candidate_top,
        "top_k_set_overlap": len(set(reference_top) & set(candidate_top)) / top_k,
        "top1_equal": reference_top1 == candidate_top1,
        "reference_top1": {
            "index": reference_top1,
            "value": float(reference[reference_top1]),
            "margin": reference_margin,
        },
        "candidate_top1": {
            "index": candidate_top1,
            "value": float(candidate[candidate_top1]),
            "margin": candidate_margin,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--mean-abs", type=float)
    parser.add_argument("--require-top1", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    try:
        reference = load_f32(args.reference)
        candidate = load_f32(args.candidate)
        summary = compare(reference, candidate, args.top_k)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    summary["reference"] = {
        "path": str(args.reference.resolve()),
        "sha256": sha256_file(args.reference),
    }
    summary["candidate"] = {
        "path": str(args.candidate.resolve()),
        "sha256": sha256_file(args.candidate),
    }

    failures: list[str] = []
    if args.expected_count is not None and summary["count"] != args.expected_count:
        failures.append(
            f"expected {args.expected_count} logits, got {summary['count']}"
        )
    if args.max_abs is not None and summary["max_abs_error"] > args.max_abs:
        failures.append(
            f"max_abs_error {summary['max_abs_error']} exceeds {args.max_abs}"
        )
    if args.mean_abs is not None and summary["mean_abs_error"] > args.mean_abs:
        failures.append(
            f"mean_abs_error {summary['mean_abs_error']} exceeds {args.mean_abs}"
        )
    if args.require_top1 and not summary["top1_equal"]:
        failures.append("top-1 token differs")
    if args.require_exact and summary["exact_values"] != summary["count"]:
        failures.append("logits are not byte-for-byte numerically equal")
    if summary["nonfinite_mismatch"]:
        failures.append(
            f"{summary['nonfinite_mismatch']} non-finite logits differ"
        )

    summary["valid"] = not failures
    summary["failures"] = failures
    output = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
