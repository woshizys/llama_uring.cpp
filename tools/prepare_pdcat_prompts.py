#!/usr/bin/env python3
"""Build the fixed, prompt-isolated PDCat train/validation/Gate-A datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path(
    "/data/dataset/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
DEFAULT_SOURCE_SHA256 = (
    "014bcc3352fd62df5bbb7fb8af9b4fd12f87bb8a2b48a147789f245176ac8e4f"
)
DATASET_ID = "pdcat-deepseek-v2-lite-prompts-v1"
WHITESPACE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def first_human_prompt(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    for turn in conversations:
        if (
            isinstance(turn, dict)
            and turn.get("from") in ("human", "user")
            and isinstance(turn.get("value"), str)
        ):
            prompt = WHITESPACE.sub(" ", turn["value"]).strip()
            return prompt or None
    return None


def select_real_prompts(source: Path, seed: str) -> list[dict[str, Any]]:
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load ShareGPT source {source}: {error}") from error
    if not isinstance(records, list):
        raise RuntimeError(f"{source}: expected a top-level JSON array")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, record in enumerate(records):
        prompt = first_human_prompt(record)
        if prompt is None or not 8 <= len(prompt) <= 1024:
            continue
        prompt_hash = sha256_text(prompt)
        if prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        source_id = str(record.get("id", source_index))
        rank = sha256_text(f"{seed}\0{source_id}\0{prompt_hash}")
        candidates.append(
            {
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "source_id": source_id,
                "source_index": source_index,
                "selection_rank": rank,
                "category": "real-sharegpt",
            }
        )

    candidates.sort(key=lambda row: (row["selection_rank"], row["prompt_sha256"]))
    test_rows = [row for row in candidates if len(row["prompt"]) <= 96][:80]
    test_hashes = {row["prompt_sha256"] for row in test_rows}
    train_validation_rows = [
        row
        for row in candidates
        if 32 <= len(row["prompt"]) <= 1024
        and row["prompt_sha256"] not in test_hashes
    ][:100]
    if len(test_rows) != 80 or len(train_validation_rows) != 100:
        raise RuntimeError(
            "not enough unique prompts for 100 train/validation and 80 short test rows"
        )
    return train_validation_rows + test_rows


def synthetic_prompts() -> list[dict[str, str]]:
    topics = [
        (
            "systems",
            "Explain the trade-off between latency and throughput in a single-request "
            "edge inference service. Include one concrete measurement pitfall.",
        ),
        (
            "storage",
            "Design a bounded priority queue for asynchronous NVMe reads and background "
            "writes. State the invariant that prevents background traffic from starving "
            "critical reads.",
        ),
        (
            "moe",
            "Describe how a mixture-of-experts router selects experts while keeping "
            "prediction-based prefetch from changing model semantics.",
        ),
        (
            "testing",
            "Write a concise test plan for detecting stale cache-slot generations during "
            "repeated asynchronous expert replacement.",
        ),
        (
            "performance",
            "Given a fixed byte budget, explain how cache hit rate, read amplification, "
            "and effective SSD bandwidth jointly constrain token latency.",
        ),
        (
            "math",
            "A device reads 54 MiB per token at 1.8 GiB/s. Compute the ideal I/O lower "
            "bound in milliseconds and list two reasons measured stall may be higher.",
        ),
        (
            "math",
            "Compare the arithmetic mean, median, p95, and p99 for a latency distribution. "
            "Explain which values should be reported for an interactive system.",
        ),
        (
            "code",
            "Write pseudocode for an atomic file replacement protocol using a temporary "
            "file, fdatasync, rename, and parent-directory fsync.",
        ),
        (
            "code",
            "Review a generation counter design where each cache slot increments its "
            "version before reuse. Identify the checks required at I/O completion and "
            "kernel launch.",
        ),
        (
            "code",
            "Provide a small algorithm that compares two greedy token streams and their "
            "per-step logits, reporting the first mismatch and maximum absolute error.",
        ),
        (
            "chinese",
            "请解释为什么混合专家模型的预取只能决定数据搬运，而不能修改原生路由器最终选择的专家。",
        ),
        (
            "chinese",
            "请设计一个实验，比较预填充阶段按层流式加载与解码阶段持久专家缓存的差异，并列出关键指标。",
        ),
        (
            "chinese",
            "在统一内存设备上，映射锁页内存直接访问和复制到显存后访问各有什么优缺点？请给出交叉点测量方法。",
        ),
        (
            "chinese",
            "如何验证后台 KV 快照写入不会破坏前台专家读取的尾延迟？请说明对照组、负载和统计方法。",
        ),
        (
            "chinese",
            "请用简洁步骤说明如何保证训练提示词、验证提示词和最终评测提示词完全隔离且可复现。",
        ),
        (
            "reasoning",
            "An optimization improves average token latency but worsens p99 and doubles "
            "SSD bytes. Decide whether to keep it for an edge assistant and justify the "
            "decision using explicit assumptions.",
        ),
        (
            "reasoning",
            "A prefetch arrives after its expert is demanded but before the blocking read "
            "finishes. Classify the event for prediction recall, ready recall, and wasted "
            "bytes, explaining each choice.",
        ),
        (
            "reasoning",
            "Two configurations use different cache capacities and one appears faster. "
            "Explain why the comparison is invalid and propose a budget-normalized rerun.",
        ),
        (
            "reasoning",
            "A full-layer prefetch cannot fit two staging buffers. Propose a tiled fallback "
            "and name the evidence needed before claiming compute and I/O overlap.",
        ),
        (
            "reasoning",
            "Router IDs match the baseline but final logits differ. Give a prioritized "
            "debugging sequence covering expert bytes, slot mapping, synchronization, "
            "kernel addressing, and floating-point accumulation.",
        ),
    ]
    rows = []
    for index, (category, prompt) in enumerate(topics):
        rows.append(
            {
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "source_id": f"pdcat-synthetic-{index:03d}",
                "source_index": -1,
                "selection_rank": sha256_text(f"synthetic\0{index}\0{prompt}"),
                "category": f"synthetic-{category}",
            }
        )
    return rows


def assign_splits(real_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_rows: list[dict[str, Any]] = []
    boundaries = (("train", 0, 80), ("validation", 80, 100), ("test", 100, 180))
    for split, begin, end in boundaries:
        for index, selected in enumerate(real_rows[begin:end]):
            row = dict(selected)
            row.update(
                {
                    "schema_version": 1,
                    "dataset_id": DATASET_ID,
                    "split": split,
                    "prompt_id": f"{split}-real-{index:03d}",
                }
            )
            split_rows.append(row)

    for index, selected in enumerate(synthetic_prompts()):
        row = dict(selected)
        row.update(
            {
                "schema_version": 1,
                "dataset_id": DATASET_ID,
                "split": "test",
                "prompt_id": f"test-synthetic-{index:03d}",
            }
        )
        split_rows.append(row)
    return split_rows


def validate(rows: list[dict[str, Any]]) -> None:
    expected = {"train": 80, "validation": 20, "test": 100}
    counts = {
        split: sum(row["split"] == split for row in rows) for split in expected
    }
    if counts != expected:
        raise RuntimeError(f"unexpected split counts: {counts}")
    ids = [row["prompt_id"] for row in rows]
    hashes = [row["prompt_sha256"] for row in rows]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("prompt IDs and hashes must be globally unique")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/pdcat/datasets"),
    )
    parser.add_argument("--seed", default="pdcat-v0.3-prompt-split-2026-08-11")
    args = parser.parse_args()

    source = args.source.resolve()
    actual_source_hash = sha256_file(source)
    if actual_source_hash.lower() != args.source_sha256.lower():
        raise RuntimeError(
            f"ShareGPT SHA-256 mismatch: expected {args.source_sha256}, "
            f"got {actual_source_hash}"
        )

    rows = assign_splits(select_real_prompts(source, args.seed))
    validate(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "prompt_pool_200.jsonl"
    predictor_path = output_dir / "predictor_train_validation_100.jsonl"
    gate_path = output_dir / "gate_a_100.jsonl"
    write_jsonl(pool_path, rows)
    write_jsonl(predictor_path, [row for row in rows if row["split"] != "test"])
    write_jsonl(gate_path, [row for row in rows if row["split"] == "test"])

    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source": {
            "path": str(source),
            "sha256": actual_source_hash,
            "selection_field": "first human turn",
            "normalization": "collapse whitespace and strip",
            "train_validation_character_length": [32, 1024],
            "gate_a_real_character_length": [8, 96],
        },
        "selection": {
            "algorithm": "ascending sha256(seed\\0source_id\\0prompt_sha256)",
            "seed": args.seed,
            "real_prompts": 180,
            "synthetic_prompts": 20,
        },
        "split_counts": {"train": 80, "validation": 20, "test": 100},
        "prompt_hash_overlap_across_splits": 0,
        "files": {
            pool_path.name: {
                "sha256": sha256_file(pool_path),
                "rows": len(rows),
            },
            predictor_path.name: {
                "sha256": sha256_file(predictor_path),
                "rows": 100,
            },
            gate_path.name: {
                "sha256": sha256_file(gate_path),
                "rows": 100,
            },
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
