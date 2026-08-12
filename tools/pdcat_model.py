#!/usr/bin/env python3
"""Inspect GGUF MoE topology through a configurable tensor mapping adapter."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


class ModelInspectionError(RuntimeError):
    """Raised when a GGUF model is incompatible with the configured MoE adapter."""


@dataclass(frozen=True)
class ExpertTensor:
    layer: int
    part: str
    tensor_name: str
    tensor_type: str
    shape: tuple[int, ...]
    tensor_offset: int
    tensor_nbytes: int
    expert_axis: int
    expert_count: int
    expert_stride: int

    def expert_offset(self, expert: int) -> int:
        if expert < 0 or expert >= self.expert_count:
            raise IndexError(f"expert {expert} is outside [0, {self.expert_count})")
        return self.tensor_offset + expert * self.expert_stride


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_value(reader: GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def require_adapter(adapter: dict[str, Any]) -> tuple[re.Pattern[str], int]:
    pattern_text = adapter.get("tensor_pattern")
    if not isinstance(pattern_text, str) or not pattern_text:
        raise ModelInspectionError("tensor_adapter.tensor_pattern must be a non-empty string")
    try:
        pattern = re.compile(pattern_text)
    except re.error as error:
        raise ModelInspectionError(f"invalid tensor adapter regex: {error}") from error
    required_groups = {"layer", "part"}
    if not required_groups.issubset(pattern.groupindex):
        raise ModelInspectionError(
            "tensor_adapter.tensor_pattern needs named groups 'layer' and 'part'"
        )

    expert_axis = adapter.get("expert_axis")
    if not isinstance(expert_axis, int):
        raise ModelInspectionError("tensor_adapter.expert_axis must be an integer")
    return pattern, expert_axis


def discover_expert_tensors(
    reader: GGUFReader,
    adapter: dict[str, Any],
) -> list[ExpertTensor]:
    pattern, configured_axis = require_adapter(adapter)
    tensors: list[ExpertTensor] = []

    for tensor in reader.tensors:
        match = pattern.fullmatch(tensor.name)
        if match is None:
            continue

        shape = tuple(int(value) for value in tensor.shape.tolist())
        axis = configured_axis if configured_axis >= 0 else len(shape) + configured_axis
        if axis < 0 or axis >= len(shape):
            raise ModelInspectionError(
                f"{tensor.name}: expert axis {configured_axis} is invalid for shape {shape}"
            )
        expert_count = shape[axis]
        if expert_count <= 0:
            raise ModelInspectionError(f"{tensor.name}: invalid expert count {expert_count}")
        tensor_nbytes = int(tensor.n_bytes)
        if tensor_nbytes % expert_count != 0:
            raise ModelInspectionError(
                f"{tensor.name}: {tensor_nbytes} bytes are not divisible by "
                f"{expert_count} experts"
            )

        tensors.append(
            ExpertTensor(
                layer=int(match.group("layer")),
                part=match.group("part"),
                tensor_name=tensor.name,
                tensor_type=str(tensor.tensor_type).split(".")[-1],
                shape=shape,
                tensor_offset=int(tensor.data_offset),
                tensor_nbytes=tensor_nbytes,
                expert_axis=axis,
                expert_count=expert_count,
                expert_stride=tensor_nbytes // expert_count,
            )
        )

    if not tensors:
        raise ModelInspectionError(
            "the configured tensor adapter did not match any expert tensors"
        )
    return tensors


def _single_or_mapping(values: dict[int, int]) -> int | dict[str, int]:
    unique = set(values.values())
    if len(unique) == 1:
        return next(iter(unique))
    return {str(layer): value for layer, value in sorted(values.items())}


def inspect_model(
    model_path: Path,
    adapter: dict[str, Any],
    hash_model: bool = False,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise ModelInspectionError(f"GGUF model does not exist: {model_path}")

    reader = GGUFReader(model_path)
    architecture = field_value(reader, "general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ModelInspectionError("GGUF general.architecture is missing")

    tensors = discover_expert_tensors(reader, adapter)
    by_layer: dict[int, list[ExpertTensor]] = {}
    for tensor in tensors:
        by_layer.setdefault(tensor.layer, []).append(tensor)

    expert_counts: dict[int, int] = {}
    object_bytes: dict[int, int] = {}
    parts_by_layer: dict[int, list[str]] = {}
    for layer, layer_tensors in sorted(by_layer.items()):
        counts = {tensor.expert_count for tensor in layer_tensors}
        if len(counts) != 1:
            raise ModelInspectionError(
                f"layer {layer} has inconsistent expert counts: {sorted(counts)}"
            )
        roles = [tensor.part for tensor in layer_tensors]
        if len(roles) != len(set(roles)):
            raise ModelInspectionError(f"layer {layer} has duplicate expert tensor roles")
        expert_counts[layer] = next(iter(counts))
        object_bytes[layer] = sum(tensor.expert_stride for tensor in layer_tensors)
        parts_by_layer[layer] = sorted(roles)

    prefix = architecture + "."
    topology = {
        "architecture": architecture,
        "block_count": field_value(reader, prefix + "block_count"),
        "leading_dense_block_count": field_value(
            reader, prefix + "leading_dense_block_count"
        ),
        "expert_count": _single_or_mapping(expert_counts),
        "expert_used_count": field_value(reader, prefix + "expert_used_count"),
        "moe_layer_count": len(by_layer),
        "moe_layers": sorted(by_layer),
        "parts": sorted({tensor.part for tensor in tensors}),
        "parts_by_layer": {
            str(layer): parts for layer, parts in sorted(parts_by_layer.items())
        },
        "expert_object_bytes": _single_or_mapping(object_bytes),
        "expert_tensor_count": len(tensors),
        "expert_object_count": sum(expert_counts.values()),
    }

    return {
        "schema_version": 1,
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path) if hash_model else None,
            "general_name": field_value(reader, "general.name"),
            "file_type": field_value(reader, "general.file_type"),
        },
        "tensor_adapter": adapter,
        "topology": topology,
        "expert_tensors": [asdict(tensor) for tensor in tensors],
    }


def validate_expected(
    inspection: dict[str, Any],
    model_config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    model = inspection["model"]
    topology = inspection["topology"]

    expected_size = model_config.get("size_bytes")
    if expected_size is not None and model["size_bytes"] != expected_size:
        errors.append(
            f"model size mismatch: expected {expected_size}, got {model['size_bytes']}"
        )

    expected_arch = model_config.get("gguf_architecture")
    if expected_arch is not None and topology["architecture"] != expected_arch:
        errors.append(
            f"architecture mismatch: expected {expected_arch}, "
            f"got {topology['architecture']}"
        )

    expected = model_config.get("expected_topology", {})
    if not isinstance(expected, dict):
        errors.append("model.expected_topology must be an object")
        return errors
    for key, expected_value in expected.items():
        if key not in topology:
            errors.append(f"unsupported expected topology key: {key}")
            continue
        actual = topology[key]
        if key == "parts" and isinstance(expected_value, list):
            expected_value = sorted(expected_value)
        if actual != expected_value:
            errors.append(
                f"topology mismatch for {key}: expected {expected_value!r}, got {actual!r}"
            )
    return errors


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelInspectionError(f"cannot load {path}: {error}") from error
    if not isinstance(data, dict):
        raise ModelInspectionError(f"{path}: top-level JSON value must be an object")
    return data


def inspect_from_run_config(
    config: dict[str, Any],
    config_path: Path,
    hash_model: bool = False,
) -> dict[str, Any]:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ModelInspectionError("config.model must be an object")
    raw_model_path = model_config.get("path")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise ModelInspectionError("config.model.path must be a non-empty string")
    model_path = Path(raw_model_path)
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()

    adapter = model_config.get("tensor_adapter")
    if not isinstance(adapter, dict):
        raise ModelInspectionError("config.model.tensor_adapter must be an object")

    inspection = inspect_model(model_path, adapter, hash_model=hash_model)
    errors = validate_expected(inspection, model_config)
    if errors:
        raise ModelInspectionError("; ".join(errors))
    return inspection


def expert_objects(tensors: Iterable[ExpertTensor]) -> Iterable[tuple[int, int, list[ExpertTensor]]]:
    grouped: dict[tuple[int, int], list[ExpertTensor]] = {}
    tensors_list = list(tensors)
    for tensor in tensors_list:
        for expert in range(tensor.expert_count):
            grouped.setdefault((tensor.layer, expert), []).append(tensor)
    for (layer, expert), parts in sorted(grouped.items()):
        yield layer, expert, sorted(parts, key=lambda item: (item.tensor_offset, item.part))
