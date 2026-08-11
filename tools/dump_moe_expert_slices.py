#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


PART_ORDER = {
    "up": 0,
    "gate": 1,
    "down": 2,
    "gate_up": 3,
}

MOE_TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<part>gate_up|gate|down|up)_exps(?:\.weight)?$"
)


@dataclass(frozen=True)
class TensorPart:
    layer: int
    expert: int
    part: str
    tensor_name: str
    tensor_type: str
    tensor_shape: tuple[int, ...]
    tensor_offset: int
    tensor_nbytes: int
    tensor_n_expert: int
    expert_stride: int
    file_id: int
    file_offset: int
    file_size: int


@dataclass
class LayoutPart:
    tensor: TensorPart
    order: int
    slice_index: int
    offset_in_slice: int
    offset_in_buffer: int


@dataclass
class MergedSlice:
    file_id: int
    file_offset: int
    file_size: int

    @property
    def end(self) -> int:
        return self.file_offset + self.file_size

    def can_merge(self, other: TensorPart) -> bool:
        return self.file_id == other.file_id and self.end >= other.file_offset

    def merge(self, other: TensorPart) -> None:
        end = max(self.end, other.file_offset + other.file_size)
        self.file_size = end - self.file_offset


def iter_moe_tensor_parts(reader: GGUFReader, layers: set[int] | None) -> Iterable[TensorPart]:
    for tensor in reader.tensors:
        match = MOE_TENSOR_RE.match(tensor.name)
        if match is None:
            continue

        layer = int(match.group("layer"))
        if layers is not None and layer not in layers:
            continue

        part = match.group("part")
        shape = tuple(int(v) for v in tensor.shape.tolist())
        if len(shape) < 3:
            print(
                f"warning: skip {tensor.name}: expected at least 3 dimensions, got {shape}",
                file=sys.stderr,
            )
            continue

        n_expert = shape[2]
        if n_expert <= 0:
            print(f"warning: skip {tensor.name}: invalid expert count {n_expert}", file=sys.stderr)
            continue
        if tensor.n_bytes % n_expert != 0:
            print(
                f"warning: skip {tensor.name}: n_bytes {tensor.n_bytes} is not divisible by n_expert {n_expert}",
                file=sys.stderr,
            )
            continue

        expert_stride = tensor.n_bytes // n_expert
        for expert in range(n_expert):
            yield TensorPart(
                layer=layer,
                expert=expert,
                part=part,
                tensor_name=tensor.name,
                tensor_type=str(tensor.tensor_type).split(".")[-1],
                tensor_shape=shape,
                tensor_offset=int(tensor.data_offset),
                tensor_nbytes=int(tensor.n_bytes),
                tensor_n_expert=n_expert,
                expert_stride=expert_stride,
                file_id=0,
                file_offset=int(tensor.data_offset + expert * expert_stride),
                file_size=expert_stride,
            )


def build_expert_layout(parts: list[TensorPart]) -> tuple[list[LayoutPart], list[MergedSlice], int]:
    sorted_parts = sorted(parts, key=lambda p: (p.file_id, p.file_offset, PART_ORDER[p.part]))
    merged_slices: list[MergedSlice] = []
    layout: list[LayoutPart] = []
    offset_in_buffer = 0

    for order, part in enumerate(sorted_parts):
        if merged_slices and merged_slices[-1].can_merge(part):
            merged_slices[-1].merge(part)
            slice_index = len(merged_slices) - 1
        else:
            merged_slices.append(MergedSlice(part.file_id, part.file_offset, part.file_size))
            slice_index = len(merged_slices) - 1

        merged = merged_slices[slice_index]
        layout.append(
            LayoutPart(
                tensor=part,
                order=order,
                slice_index=slice_index,
                offset_in_slice=part.file_offset - merged.file_offset,
                offset_in_buffer=offset_in_buffer,
            )
        )
        offset_in_buffer += part.file_size

    loaded_size = sum(s.file_size for s in merged_slices)
    return layout, merged_slices, loaded_size


def summarize(layouts: dict[tuple[int, int], tuple[list[LayoutPart], list[MergedSlice], int]]) -> None:
    by_layer: dict[int, list[tuple[int, list[LayoutPart], list[MergedSlice], int]]] = defaultdict(list)
    for (layer, expert), (layout, slices, loaded_size) in layouts.items():
        by_layer[layer].append((expert, layout, slices, loaded_size))

    print("MoE ExpertManager registration summary", file=sys.stderr)
    print(f"  layers: {len(by_layer)}", file=sys.stderr)
    print(f"  experts: {len(layouts)}", file=sys.stderr)

    for layer in sorted(by_layer):
        rows = sorted(by_layer[layer], key=lambda item: item[0])
        signatures = {
            tuple((p.tensor.part, p.offset_in_buffer, p.tensor.file_size) for p in layout)
            for _, layout, _, _ in rows
        }
        file_orders = {
            tuple(p.tensor.part for p in layout)
            for _, layout, _, _ in rows
        }
        loaded_sizes = [loaded_size for _, _, _, loaded_size in rows]
        slot_sizes = [sum(p.tensor.file_size for p in layout) for _, layout, _, _ in rows]
        example_layout = rows[0][1] if rows else []
        offsets = ", ".join(
            f"{p.tensor.part}@{p.offset_in_buffer}+{p.tensor.file_size}" for p in example_layout
        )
        print(
            "  layer "
            f"{layer}: experts={len(rows)} parts={len(example_layout)} "
            f"slot_size={min(slot_sizes) if slot_sizes else 0}"
            f"{'' if len(set(slot_sizes)) == 1 else '..' + str(max(slot_sizes))} "
            f"loaded_size={min(loaded_sizes) if loaded_sizes else 0}"
            f"{'' if len(set(loaded_sizes)) == 1 else '..' + str(max(loaded_sizes))} "
            f"layout_consistent={len(signatures) == 1} "
            f"file_order_consistent={len(file_orders) == 1} "
            f"layout=[{offsets}]",
            file=sys.stderr,
        )


def write_csv(layouts: dict[tuple[int, int], tuple[list[LayoutPart], list[MergedSlice], int]]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "layer",
            "expert",
            "part",
            "part_order_in_expert_buffer",
            "part_offset_in_expert_buffer",
            "part_size",
            "slice_index",
            "offset_in_merged_slice",
            "merged_slice_file_offset",
            "merged_slice_file_size",
            "file_id",
            "file_offset",
            "tensor_name",
            "tensor_type",
            "tensor_shape",
            "tensor_offset",
            "tensor_nbytes",
            "tensor_n_expert",
            "expert_stride",
            "expert_loaded_size",
        ],
    )
    writer.writeheader()
    for (layer, expert) in sorted(layouts):
        layout, slices, loaded_size = layouts[(layer, expert)]
        for part in layout:
            merged = slices[part.slice_index]
            tensor = part.tensor
            writer.writerow(
                {
                    "layer": layer,
                    "expert": expert,
                    "part": tensor.part,
                    "part_order_in_expert_buffer": part.order,
                    "part_offset_in_expert_buffer": part.offset_in_buffer,
                    "part_size": tensor.file_size,
                    "slice_index": part.slice_index,
                    "offset_in_merged_slice": part.offset_in_slice,
                    "merged_slice_file_offset": merged.file_offset,
                    "merged_slice_file_size": merged.file_size,
                    "file_id": tensor.file_id,
                    "file_offset": tensor.file_offset,
                    "tensor_name": tensor.tensor_name,
                    "tensor_type": tensor.tensor_type,
                    "tensor_shape": "x".join(str(v) for v in tensor.tensor_shape),
                    "tensor_offset": tensor.tensor_offset,
                    "tensor_nbytes": tensor.tensor_nbytes,
                    "tensor_n_expert": tensor.tensor_n_expert,
                    "expert_stride": tensor.expert_stride,
                    "expert_loaded_size": loaded_size,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump MoE expert slices as they would be registered in the ExpertManager."
    )
    parser.add_argument("model", type=Path, help="GGUF model path")
    parser.add_argument(
        "--layer",
        type=int,
        action="append",
        help="Only dump this layer. Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the per-layer summary to stderr; do not emit CSV rows.",
    )
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    layers = set(args.layer) if args.layer is not None else None

    grouped: dict[tuple[int, int], list[TensorPart]] = defaultdict(list)
    for part in iter_moe_tensor_parts(reader, layers):
        grouped[(part.layer, part.expert)].append(part)

    layouts = {
        key: build_expert_layout(parts)
        for key, parts in grouped.items()
    }

    summarize(layouts)
    if not args.summary_only:
        write_csv(layouts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
