#!/usr/bin/env python3
"""Train a prompt-isolated PDCat route predictor from native router JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/pdcat/configs/deepseek-v2-lite-q8_0.json"
DEFAULT_DATASET = (
    REPO_ROOT / "experiments/pdcat/datasets/predictor_train_validation_100.jsonl"
)
DEFAULT_TEST_DATASET = REPO_ROOT / "experiments/pdcat/datasets/gate_a_100.jsonl"



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source_file:
        value = json.load(source_file)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def read_prompt_splits(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    splits: dict[str, str] = {}
    hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("prompt_id")
            split = row.get("split")
            prompt = row.get("prompt")
            prompt_hash = row.get("prompt_sha256")
            if not isinstance(prompt_id, str) or split not in ("train", "validation"):
                raise RuntimeError(
                    f"{path}:{line_number}: predictor data must be train/validation only"
                )
            if not isinstance(prompt, str) or not isinstance(prompt_hash, str):
                raise RuntimeError(f"{path}:{line_number}: missing prompt/hash")
            actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if actual_hash != prompt_hash:
                raise RuntimeError(f"{path}:{line_number}: prompt SHA-256 mismatch")
            if prompt_id in splits or prompt_hash in hashes.values():
                raise RuntimeError(f"{path}:{line_number}: duplicate prompt")
            splits[prompt_id] = split
            hashes[prompt_id] = prompt_hash
    counts = {split: list(splits.values()).count(split) for split in ("train", "validation")}
    if counts != {"train": 80, "validation": 20}:
        raise RuntimeError(f"expected 80/20 prompt split, got {counts}")
    return splits, hashes
def read_test_prompts(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    splits: dict[str, str] = {}
    hashes: set[str] = set()
    prompt_hashes: dict[str, str] = {}
    rows = 0
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            prompt_hash = row.get("prompt_sha256")
            if row.get("split") != "test":
                raise RuntimeError(f"{path}:{line_number}: expected test-only prompts")
            if not isinstance(prompt, str) or not isinstance(prompt_hash, str):
                raise RuntimeError(f"{path}:{line_number}: missing prompt/hash")
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str):
                raise RuntimeError(f"{path}:{line_number}: missing prompt_id")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_hash:
                raise RuntimeError(f"{path}:{line_number}: prompt SHA-256 mismatch")
            if prompt_id in splits or prompt_hash in hashes:
                raise RuntimeError(f"{path}:{line_number}: duplicate test prompt")
            splits[prompt_id] = "test"
            hashes.add(prompt_hash)
            prompt_hashes[prompt_id] = prompt_hash
            rows += 1
    if rows != 100:
        raise RuntimeError(f"expected 100 held-out test prompts, got {rows}")
    return splits, prompt_hashes


def read_result_coverage(
    path: Path,
    prompt_splits: dict[str, str],
    prompt_hashes: dict[str, str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("prompt_id")
            if prompt_id not in prompt_splits or prompt_id in results:
                raise RuntimeError(f"{path}:{line_number}: unknown or duplicate prompt_id")
            if row.get("prompt_sha256") != prompt_hashes[prompt_id]:
                raise RuntimeError(f"{path}:{line_number}: prompt SHA-256 mismatch")
            generated = row.get("generated_tokens")
            decode_times = row.get("decode_input_ms")
            if not isinstance(generated, list) or not all(
                isinstance(token, int) for token in generated
            ):
                raise RuntimeError(f"{path}:{line_number}: invalid generated_tokens")
            if not isinstance(decode_times, list) or len(decode_times) != max(
                0, len(generated) - 1
            ):
                raise RuntimeError(
                    f"{path}:{line_number}: decode timing count does not match generated tokens"
                )
            results[prompt_id] = row
    missing_results = sorted(set(prompt_splits) - set(results))
    if missing_results:
        raise RuntimeError(
            f"{path}: missing {len(missing_results)} result prompts, first={missing_results[:5]}"
        )
    no_decode = {
        prompt_id
        for prompt_id, row in results.items()
        if len(row["generated_tokens"]) == 1 and not row["decode_input_ms"]
    }
    return no_decode, results



def load_routes(
    path: Path,
    prompt_splits: dict[str, str],
    num_experts: int,
    top_k: int,
    allowed_missing: set[str] | None = None,
) -> tuple[
    dict[str, dict[int, dict[int, tuple[list[int], list[float]]]]],
    set[str],
]:
    routes: dict[str, dict[int, dict[int, tuple[list[int], list[float]]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    run_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "router_selection" or row.get("phase") != "decode":
                continue
            request_id = row.get("request_id")
            token_id = row.get("token_id")
            layer_id = row.get("layer_id")
            experts = row.get("experts")
            scores = row.get("router_scores")
            if request_id not in prompt_splits:
                raise RuntimeError(
                    f"{path}:{line_number}: route request {request_id!r} is outside train/validation"
                )
            if not isinstance(token_id, int) or token_id < 0:
                raise RuntimeError(f"{path}:{line_number}: decode token_id must be an integer")
            if not isinstance(layer_id, int) or layer_id < 0:
                raise RuntimeError(f"{path}:{line_number}: invalid layer_id")
            if (
                not isinstance(experts, list)
                or len(experts) != top_k
                or len(set(experts)) != top_k
                or any(not isinstance(value, int) or not 0 <= value < num_experts for value in experts)
            ):
                raise RuntimeError(f"{path}:{line_number}: invalid expert top-k")
            if (
                not isinstance(scores, list)
                or len(scores) != len(experts)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores)
            ):
                raise RuntimeError(f"{path}:{line_number}: invalid router scores")
            if layer_id in routes[request_id][token_id]:
                raise RuntimeError(
                    f"{path}:{line_number}: duplicate route context "
                    f"{request_id}/{token_id}/{layer_id}"
                )
            routes[request_id][token_id][layer_id] = (
                list(experts),
                [float(value) for value in scores],
            )
            run_ids.add(str(row.get("run_id", "")))
    missing = set(prompt_splits) - set(routes)
    unexpected_missing = sorted(missing - (allowed_missing or set()))
    if unexpected_missing:
        raise RuntimeError(
            f"trace is missing {len(unexpected_missing)} prompts with expected decode routes, "
            f"first={unexpected_missing[:5]}"
        )
    return routes, run_ids


def encode_route(
    route: tuple[list[int], list[float]],
    num_experts: int,
) -> torch.Tensor:
    experts, scores = route
    features = torch.zeros(num_experts * 2, dtype=torch.float32)
    features[experts] = 1.0
    nonnegative = [max(0.0, value) for value in scores]
    score_sum = sum(nonnegative)
    normalized = (
        [value / score_sum for value in nonnegative]
        if score_sum > 0.0
        else [1.0 / len(experts)] * len(experts)
    )
    for expert, score in zip(experts, normalized):
        features[num_experts + expert] = max(
            float(features[num_experts + expert]), score
        )
    return features


def target_route(route: tuple[list[int], list[float]], num_experts: int) -> torch.Tensor:
    target = torch.zeros(num_experts, dtype=torch.float32)
    for expert in route[0]:
        target[expert] = 1.0
    return target


def build_samples(
    routes: dict[str, dict[int, dict[int, tuple[list[int], list[float]]]]],
    prompt_splits: dict[str, str],
    expected_layer_count: int,
    num_experts: int,
    allow_empty_splits: bool = False,
) -> tuple[
    dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]],
    list[int],
]:
    layer_ids = sorted(
        {
            layer_id
            for request_routes in routes.values()
            for token_routes in request_routes.values()
            for layer_id in token_routes
        }
    )
    if len(layer_ids) != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} MoE layers, trace has {layer_ids}"
        )

    split_names = sorted(set(prompt_splits.values()))
    if not split_names:
        raise RuntimeError("prompt split map is empty")
    raw: dict[int, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = {
        layer_id: {split: [] for split in split_names} for layer_id in layer_ids
    }
    for prompt_id, token_map in routes.items():
        split = prompt_splits[prompt_id]
        token_ids = sorted(token_map)
        if not token_ids or token_ids != list(range(token_ids[-1] + 1)):
            raise RuntimeError(f"{prompt_id}: decode token IDs are not contiguous from zero")
        for token_id in token_ids:
            current = token_map[token_id]
            if sorted(current) != layer_ids:
                raise RuntimeError(f"{prompt_id}/{token_id}: incomplete MoE layer trace")
            for index, source_layer in enumerate(layer_ids):
                if index + 1 < len(layer_ids):
                    target = current[layer_ids[index + 1]]
                else:
                    next_token = token_map.get(token_id + 1)
                    if next_token is None:
                        continue
                    target = next_token[layer_ids[0]]
                raw[source_layer][split].append(
                    (
                        encode_route(current[source_layer], num_experts),
                        target_route(target, num_experts),
                    )
                )

    samples: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for source_layer, split_rows in raw.items():
        samples[source_layer] = {}
        for split, rows in split_rows.items():
            if not rows:
                if allow_empty_splits:
                    continue
                raise RuntimeError(f"layer {source_layer}: no {split} samples")
            samples[source_layer][split] = (
                torch.stack([row[0] for row in rows]),
                torch.stack([row[1] for row in rows]),
            )
    return samples, layer_ids


class RouteMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(inputs)))


def evaluate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_values: tuple[int, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "loss": float(F.binary_cross_entropy_with_logits(logits, targets))
    }
    target_bool = targets.bool()
    target_count = target_bool.sum(dim=1).clamp_min(1)
    for top_m in top_values:
        indices = torch.topk(logits, k=top_m, dim=1).indices
        predicted = torch.zeros_like(target_bool)
        predicted.scatter_(1, indices, True)
        hits = (predicted & target_bool).sum(dim=1)
        metrics[f"recall@{top_m}"] = float((hits / target_count).float().mean())
        metrics[f"precision@{top_m}"] = float((hits / top_m).float().mean())
        metrics[f"waste@{top_m}"] = float(((top_m - hits) / top_m).float().mean())
        metrics[f"any_hit@{top_m}"] = float((hits > 0).float().mean())
        metrics[f"exact@{top_m}"] = float((predicted == target_bool).all(dim=1).float().mean())
    return metrics


def train_layer(
    source_layer: int,
    train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    top_values: tuple[int, ...],
) -> tuple[RouteMlp, dict[str, float]]:
    inputs, targets = train
    validation_inputs, validation_targets = validation
    torch.manual_seed(seed + source_layer)
    model = RouteMlp(inputs.shape[1], hidden_dim, targets.shape[1])
    positive = targets.sum(dim=0)
    negative = targets.shape[0] - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1000 + source_layer)

    model.train()
    for _ in range(epochs):
        for indices in torch.randperm(inputs.shape[0], generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs[indices])
            loss = F.binary_cross_entropy_with_logits(
                logits, targets[indices], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        metrics = evaluate(model(validation_inputs), validation_targets, top_values)
    metrics["train_samples"] = float(inputs.shape[0])
    metrics["validation_samples"] = float(validation_inputs.shape[0])
    return model, metrics


def aggregate_layer_metrics(
    layers: list[dict[str, Any]],
    field: str,
    count_key: str,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    total_samples = 0.0
    available = [layer for layer in layers if isinstance(layer.get(field), dict)]
    for layer in available:
        metrics = layer[field]
        samples = float(metrics[count_key])
        total_samples += samples
        for key, value in metrics.items():
            if not key.endswith("_samples"):
                totals[key] += float(value) * samples
    if total_samples <= 0:
        raise RuntimeError(f"cannot aggregate empty {field} metrics")
    return {
        count_key: total_samples,
        **{key: value / total_samples for key, value in sorted(totals.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--test-trace", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if min(args.hidden_dim, args.epochs, args.batch_size, args.threads) <= 0:
        raise RuntimeError("hidden-dim, epochs, batch-size, and threads must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    config_path = args.config.resolve()
    dataset_path = args.dataset.resolve()
    test_dataset_path = args.test_dataset.resolve()
    trace_path = args.trace.resolve()
    test_trace_path = args.test_trace.resolve()
    results_path = (
        args.results.resolve()
        if args.results is not None
        else trace_path.with_name("gate_a_results.jsonl")
    )
    test_results_path = (
        args.test_results.resolve()
        if args.test_results is not None
        else test_trace_path.with_name("gate_a_results.jsonl")
    )
    output_dir = args.output_dir.resolve()
    config = read_json(config_path)
    model_config = config["model"]
    topology = model_config["expected_topology"]
    num_experts = int(topology["expert_count"])
    top_k = int(topology["expert_used_count"])
    expected_layer_count = int(topology["moe_layer_count"])

    prompt_splits, prompt_hashes = read_prompt_splits(dataset_path)
    test_prompt_splits, test_prompt_hashes = read_test_prompts(test_dataset_path)
    prompt_hash_overlap = set(prompt_hashes.values()) & set(test_prompt_hashes.values())
    if prompt_hash_overlap:
        raise RuntimeError(
            f"predictor train/validation overlaps {len(prompt_hash_overlap)} held-out prompts"
        )
    no_decode_prompts, results = read_result_coverage(
        results_path, prompt_splits, prompt_hashes
    )
    test_no_decode_prompts, test_results = read_result_coverage(
        test_results_path, test_prompt_splits, test_prompt_hashes
    )
    routes, run_ids = load_routes(
        trace_path,
        prompt_splits,
        num_experts,
        top_k,
        allowed_missing=no_decode_prompts,
    )
    samples, layer_ids = build_samples(
        routes, prompt_splits, expected_layer_count, num_experts
    )
    test_routes, test_run_ids = load_routes(
        test_trace_path,
        test_prompt_splits,
        num_experts,
        top_k,
        allowed_missing=test_no_decode_prompts,
    )
    test_samples, test_layer_ids = build_samples(
        test_routes,
        test_prompt_splits,
        expected_layer_count,
        num_experts,
        allow_empty_splits=True,
    )
    if test_layer_ids != layer_ids:
        raise RuntimeError(
            f"held-out trace layer IDs {test_layer_ids} do not match training {layer_ids}"
        )
    if min(samples[layer]["train"][0].shape[0] for layer in layer_ids) < args.min_train_samples:
        raise RuntimeError("one or more transitions have too few training samples")

    weights: dict[str, torch.Tensor] = {}
    layer_manifest: list[dict[str, Any]] = []
    top_values = tuple(sorted({1, 2, 4, top_k}))
    for index, source_layer in enumerate(layer_ids):
        target_layer = layer_ids[index + 1] if index + 1 < len(layer_ids) else layer_ids[0]
        model, metrics = train_layer(
            source_layer,
            samples[source_layer]["train"],
            samples[source_layer]["validation"],
            args.hidden_dim,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.seed,
            top_values,
        )
        test_metrics = None
        if "test" in test_samples[source_layer]:
            test_inputs, test_targets = test_samples[source_layer]["test"]
            with torch.inference_mode():
                test_metrics = evaluate(model(test_inputs), test_targets, top_values)
            test_metrics["test_samples"] = float(test_inputs.shape[0])
        prefix = f"layer.{source_layer}"
        weights[f"{prefix}.w1"] = model.w1.weight.detach().contiguous()
        weights[f"{prefix}.b1"] = model.w1.bias.detach().contiguous()
        weights[f"{prefix}.w2"] = model.w2.weight.detach().contiguous()
        weights[f"{prefix}.b2"] = model.w2.bias.detach().contiguous()
        layer_manifest.append(
            {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "input_dim": num_experts * 2,
                "hidden_dim": args.hidden_dim,
                "output_dim": num_experts,
                "tensor_prefix": prefix,
                "metrics": metrics,
                "test_metrics": test_metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "predictor.safetensors"
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite and (weights_path.exists() or manifest_path.exists()):
        raise RuntimeError("output exists; pass --overwrite to replace it")
    save_file(
        weights,
        str(weights_path),
        metadata={
            "format": "pdcat-route-predictor-v1",
            "model_id": str(config["experiment_id"]),
            "feature_schema": "expert_presence_and_router_score",
        },
    )
    restored = load_file(str(weights_path), device="cpu")
    if any(name not in restored or not torch.equal(tensor, restored[name]) for name, tensor in weights.items()):
        raise RuntimeError("safetensors round-trip mismatch")

    split_prompt_hashes = {
        split: hashlib.sha256(
            "\n".join(
                sorted(
                    prompt_hashes[prompt_id]
                    for prompt_id, value in prompt_splits.items()
                    if value == split
                )
            ).encode("utf-8")
        ).hexdigest()
        for split in ("train", "validation")
    }
    split_prompt_hashes["test"] = hashlib.sha256(
        "\n".join(sorted(test_prompt_hashes.values())).encode("utf-8")
    ).hexdigest()

    manifest = {
        "schema_version": 1,
        "predictor_type": "dense_mlp",
        "model_id": config["experiment_id"],
        "model_sha256": model_config["sha256"],
        "num_experts": num_experts,
        "feature_schema": {
            "kind": "expert_presence_and_router_score",
            "input_dim": num_experts * 2,
            "router_scores": "selected_normalized",
        },
        "activation": "silu",
        "output_activation": "sigmoid",
        "weights_file": weights_path.name,
        "weights_sha256": sha256_file(weights_path),
        "layers": layer_manifest,
        "aggregate_metrics": {
            "validation": aggregate_layer_metrics(
                layer_manifest, "metrics", "validation_samples"
            ),
            "held_out_test": aggregate_layer_metrics(
                layer_manifest, "test_metrics", "test_samples"
            ),
        },
        "provenance": {
            "status": "trained_prompt_isolated",
            "publication_valid": True,
            "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "trace_path": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "trace_run_ids": sorted(run_ids),
            "results_path": str(results_path),
            "results_sha256": sha256_file(results_path),
            "test_trace_path": str(test_trace_path),
            "test_trace_sha256": sha256_file(test_trace_path),
            "test_trace_run_ids": sorted(test_run_ids),
            "test_results_path": str(test_results_path),
            "test_results_sha256": sha256_file(test_results_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "test_dataset_path": str(test_dataset_path),
            "test_dataset_sha256": sha256_file(test_dataset_path),
            "prompt_split_counts": {"train": 80, "validation": 20, "test": 100},
            "trace_prompt_split_counts": {
                "train": sum(prompt_splits[prompt_id] == "train" for prompt_id in routes),
                "validation": sum(
                    prompt_splits[prompt_id] == "validation" for prompt_id in routes
                ),
                "test": 0,
            },
            "test_trace_prompt_split_counts": {
                "train": 0,
                "validation": 0,
                "test": len(test_routes),
            },
            "held_out_test_transition_coverage": {
                "tested_source_layers": [
                    layer for layer in layer_ids if "test" in test_samples[layer]
                ],
                "missing_source_layers": [
                    layer for layer in layer_ids if "test" not in test_samples[layer]
                ],
                "tested_count": sum(
                    "test" in test_samples[layer] for layer in layer_ids
                ),
                "total_count": len(layer_ids),
                "reason": (
                    "the held-out Gate run generated two tokens and therefore did not "
                    "execute the next-token first-layer route needed to score the final "
                    "MoE-layer cyclic transition"
                ),
            },
            "no_decode_transition_prompts": {
                "train_validation": [
                    {
                        "prompt_id": prompt_id,
                        "prompt_sha256": prompt_hashes[prompt_id],
                        "split": prompt_splits[prompt_id],
                        "generated_tokens": results[prompt_id]["generated_tokens"],
                    }
                    for prompt_id in sorted(no_decode_prompts)
                ],
                "held_out_test": [
                    {
                        "prompt_id": prompt_id,
                        "prompt_sha256": test_prompt_hashes[prompt_id],
                        "split": "test",
                        "generated_tokens": test_results[prompt_id]["generated_tokens"],
                    }
                    for prompt_id in sorted(test_no_decode_prompts)
                ],
                "policy": (
                    "exclude only prompts whose recorded generation has one token and zero "
                    "decode steps; all prompts with an expected decode step require a complete trace"
                ),
            },
            "prompt_hash_overlap": len(prompt_hash_overlap),
            "split_prompt_hashes": split_prompt_hashes,
            "decode_tokens_per_prompt": {
                prompt_id: len(token_map) for prompt_id, token_map in sorted(routes.items())
            },
            "test_decode_tokens_per_prompt": {
                prompt_id: len(token_map)
                for prompt_id, token_map in sorted(test_routes.items())
            },
            "last_layer_target": "first_moe_layer_of_next_decode_token",
            "prediction_target": "all_native_router_selected_experts",
            "train_config": {
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "threads": args.threads,
                "top_m_eval": list(top_values),
            },
            "torch_version": torch.__version__,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "weights": str(weights_path),
                "weights_sha256": manifest["weights_sha256"],
                "layers": len(layer_manifest),
                "train_samples_min": min(
                    int(layer["metrics"]["train_samples"]) for layer in layer_manifest
                ),
                "validation_samples_min": min(
                    int(layer["metrics"]["validation_samples"]) for layer in layer_manifest
                ),
                "publication_valid": True,
                "validation_recall_at_2": manifest["aggregate_metrics"]["validation"]["recall@2"],
                "held_out_test_recall_at_2": manifest["aggregate_metrics"]["held_out_test"]["recall@2"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
