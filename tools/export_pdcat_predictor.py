#!/usr/bin/env python3
"""Export legacy PDCat MLP state dicts as a generic safetensors bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

LAYER_RE = re.compile(r"^predictor_layer_(\d+)\.pth$")
KEYS = ("net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def checkpoints(source_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in source_dir.iterdir():
        match = LAYER_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    if not found or len({layer for layer, _ in found}) != len(found):
        raise RuntimeError(f"missing or duplicate predictor checkpoints in {source_dir}")
    return found


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    # Never load a pickle-capable checkpoint in this conversion path.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("state_dict"), dict):
        raise RuntimeError(f"{path}: expected a checkpoint with a state_dict")
    state_raw = raw["state_dict"]
    if set(state_raw) != set(KEYS):
        raise RuntimeError(f"{path}: state_dict must contain exactly {KEYS}")
    required_metadata = (
        "layer", "predicts_layer", "num_experts", "input_dim", "hidden",
        "dropout", "model_type",
    )
    if any(key not in raw for key in required_metadata):
        raise RuntimeError(f"{path}: missing transition or topology metadata")
    metadata = {key: raw[key] for key in required_metadata}
    if metadata["model_type"] != "TraceMLP" or metadata["dropout"] != 0.0:
        raise RuntimeError(f"{path}: unsupported model type or nonzero inference dropout")
    state = {
        key: state_raw[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key in KEYS
    }
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError(f"{path}: non-finite predictor weight")
    return state, metadata


def dimensions(
    path: Path,
    state: dict[str, torch.Tensor],
    expected_experts: int | None,
) -> tuple[int, int, int]:
    w1, b1 = state["net.0.weight"], state["net.0.bias"]
    w2, b2 = state["net.3.weight"], state["net.3.bias"]
    if w1.ndim != 2 or w2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise RuntimeError(f"{path}: invalid MLP tensor rank")
    hidden, input_dim = map(int, w1.shape)
    experts, hidden2 = map(int, w2.shape)
    if b1.shape != (hidden,) or b2.shape != (experts,) or hidden2 != hidden:
        raise RuntimeError(f"{path}: inconsistent MLP dimensions")
    if input_dim != 2 * experts:
        raise RuntimeError(f"{path}: input must be presence[{experts}]+router[{experts}]")
    if expected_experts is not None and experts != expected_experts:
        raise RuntimeError(f"{path}: expected {expected_experts} experts, got {experts}")
    return input_dim, hidden, experts


def torchscript_error(
    source_dir: Path,
    layer: int,
    state: dict[str, torch.Tensor],
    input_dim: int,
) -> float:
    module = torch.jit.load(
        str(source_dir / f"predictor_layer_{layer}.pt"), map_location="cpu"
    )
    module.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x50444341 + layer)
    sample = torch.randn((8, input_dim), generator=generator)
    with torch.inference_mode():
        reference = module(sample)
        manual = F.linear(
            F.silu(F.linear(sample, state["net.0.weight"], state["net.0.bias"])),
            state["net.3.weight"],
            state["net.3.bias"],
        )
    error = float((reference - manual).abs().max())
    if error > 1.0e-6:
        raise RuntimeError(f"layer {layer}: TorchScript parity error {error:.9g}")
    return error


def optional_json(path: Path) -> Any:
    return load_json(path) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--verify-torchscript", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    found = checkpoints(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "predictor.safetensors"
    manifest_path = output_dir / "manifest.json"
    if not args.overwrite and (weights_path.exists() or manifest_path.exists()):
        raise RuntimeError("output exists; pass --overwrite to replace it")

    tensors: dict[str, torch.Tensor] = {}
    layers: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    parity: dict[str, float] = {}
    num_experts = args.num_experts

    for source_layer, path in found:
        state, checkpoint_meta = load_state(path)
        input_dim, hidden_dim, output_dim = dimensions(path, state, num_experts)
        num_experts = output_dim if num_experts is None else num_experts
        if checkpoint_meta["layer"] != source_layer:
            raise RuntimeError(f"{path}: filename/source layer metadata mismatch")
        target_layer = int(checkpoint_meta["predicts_layer"])
        prefix = f"layer.{source_layer}"
        tensors[f"{prefix}.w1"] = state["net.0.weight"]
        tensors[f"{prefix}.b1"] = state["net.0.bias"]
        tensors[f"{prefix}.w2"] = state["net.3.weight"]
        tensors[f"{prefix}.b2"] = state["net.3.bias"]
        hashes[path.name] = file_sha256(path)
        if args.verify_torchscript:
            parity[str(source_layer)] = torchscript_error(
                source_dir, source_layer, state, input_dim
            )
        layers.append(
            {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "tensor_prefix": prefix,
                "metrics": optional_json(
                    source_dir / f"metrics_layer_{source_layer}.json"
                ),
            }
        )

    assert num_experts is not None
    save_file(
        tensors,
        str(weights_path),
        metadata={
            "format": "pdcat-route-predictor-v1",
            "model_id": args.model_id,
            "feature_schema": "expert_presence_and_router_score",
        },
    )
    restored = load_file(str(weights_path), device="cpu")
    if any(name not in restored or not torch.equal(value, restored[name])
           for name, value in tensors.items()):
        raise RuntimeError("safetensors round-trip mismatch")

    manifest = {
        "schema_version": 1,
        "predictor_type": "dense_mlp",
        "model_id": args.model_id,
        "model_sha256": args.model_sha256,
        "num_experts": num_experts,
        "feature_schema": {
            "kind": "expert_presence_and_router_score",
            "input_dim": num_experts * 2,
            "router_scores": "selected_normalized",
        },
        "activation": "silu",
        "output_activation": "sigmoid",
        "weights_file": weights_path.name,
        "weights_sha256": file_sha256(weights_path),
        "layers": layers,
        "provenance": {
            "status": "provisional",
            "publication_valid": False,
            "warning": (
                "Legacy validation used a random sample split; prompt-level train/eval "
                "isolation is not evidenced and must be re-run for paper results."
            ),
            "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_checkpoint_sha256": hashes,
            "source_train_config": optional_json(source_dir / "train_config.json"),
            "source_dataset_meta": optional_json(
                source_dir / "dataset_cache" / "meta.json"
            ),
            "torch_version": torch.__version__,
            "torchscript_manual_max_abs_error": parity,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "weights_sha256": manifest["weights_sha256"],
        "layers": len(layers),
        "num_experts": num_experts,
        "cyclic_transition": [layers[-1]["source_layer"], layers[-1]["target_layer"]],
        "max_torchscript_manual_error": max(parity.values(), default=None),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
