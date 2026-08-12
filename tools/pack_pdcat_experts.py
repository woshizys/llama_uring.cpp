#!/usr/bin/env python3
"""Build and verify an aligned, read-only PDCat Expert sidecar pack."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

from pdcat_model import (
    ExpertTensor,
    ModelInspectionError,
    inspect_from_run_config,
    load_json,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "pdcat" / "configs" / "deepseek-v2-lite-q8_0.json"
)
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class PackError(RuntimeError):
    """Raised when a sidecar pack cannot be built or verified."""


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_output(path: Path | None, fallback: Path) -> Path:
    if path is None:
        return fallback.resolve()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def tensor_from_dict(value: dict[str, Any]) -> ExpertTensor:
    converted = dict(value)
    converted["shape"] = tuple(converted["shape"])
    return ExpertTensor(**converted)


def grouped_objects(
    tensors: list[ExpertTensor],
    part_order: list[str],
) -> list[tuple[int, int, list[ExpertTensor]]]:
    rank = {part: index for index, part in enumerate(part_order)}
    grouped: dict[tuple[int, int], list[ExpertTensor]] = {}
    for tensor in tensors:
        for expert in range(tensor.expert_count):
            grouped.setdefault((tensor.layer, expert), []).append(tensor)

    result: list[tuple[int, int, list[ExpertTensor]]] = []
    for (layer, expert), parts in sorted(grouped.items()):
        ordered = sorted(
            parts,
            key=lambda item: (rank.get(item.part, len(rank)), item.part, item.tensor_name),
        )
        result.append((layer, expert, ordered))
    return result


def copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    source_offset: int,
    size: int,
    digests: tuple[Any, ...],
) -> str:
    source.seek(source_offset)
    remaining = size
    part_digest = hashlib.sha256()
    while remaining:
        chunk = source.read(min(remaining, COPY_CHUNK_BYTES))
        if not chunk:
            raise PackError(
                f"short read at source offset {source_offset + size - remaining}"
            )
        destination.write(chunk)
        part_digest.update(chunk)
        for digest in digests:
            digest.update(chunk)
        remaining -= len(chunk)
    return part_digest.hexdigest()


def write_zeros(destination: BinaryIO, size: int, digest: Any) -> None:
    zero_chunk = bytes(min(COPY_CHUNK_BYTES, max(size, 1)))
    remaining = size
    while remaining:
        chunk = zero_chunk[: min(remaining, len(zero_chunk))]
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def default_paths(model_path: Path) -> tuple[Path, Path]:
    pack = model_path.with_name(model_path.name + ".pdcat-experts.pack")
    manifest = pack.with_suffix(pack.suffix + ".json")
    return pack, manifest


def build_pack(
    config_path: Path,
    pack_path: Path,
    manifest_path: Path,
    alignment: int,
    verify_model_hash: bool,
) -> dict[str, Any]:
    if alignment < 4096 or alignment & (alignment - 1):
        raise PackError("alignment must be a power of two and at least 4096")
    if pack_path.exists() or manifest_path.exists():
        raise PackError(
            "refusing to overwrite an existing pack or manifest; remove the exact "
            "target explicitly after reviewing it"
        )

    config = load_json(config_path)
    inspection = inspect_from_run_config(
        config, config_path, hash_model=verify_model_hash
    )
    model_config = config["model"]
    expected_model_hash = str(model_config["sha256"]).lower()
    actual_model_hash = inspection["model"].get("sha256")
    if actual_model_hash is not None and actual_model_hash.lower() != expected_model_hash:
        raise PackError(
            f"model SHA-256 mismatch: expected {expected_model_hash}, "
            f"got {actual_model_hash}"
        )

    model_path = Path(inspection["model"]["path"])
    tensors = [
        tensor_from_dict(value) for value in inspection["expert_tensors"]
    ]
    part_order = list(model_config["tensor_adapter"].get("part_order", []))
    if not part_order:
        part_order = sorted({tensor.part for tensor in tensors})
    objects = grouped_objects(tensors, part_order)

    pack_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_pack = pack_path.with_name(
        f".{pack_path.name}.tmp-{os.getpid()}"
    )
    pack_digest = hashlib.sha256()
    object_records: list[dict[str, Any]] = []
    pack_offset = 0

    try:
        with model_path.open("rb") as source, temporary_pack.open("xb") as target:
            for layer, expert, parts in objects:
                object_offset = align_up(pack_offset, alignment)
                if object_offset != pack_offset:
                    write_zeros(target, object_offset - pack_offset, pack_digest)
                    pack_offset = object_offset

                object_digest = hashlib.sha256()
                part_records: list[dict[str, Any]] = []
                payload_size = 0
                for tensor in parts:
                    source_offset = tensor.expert_offset(expert)
                    part_pack_offset = pack_offset
                    checksum = copy_exact(
                        source,
                        target,
                        source_offset,
                        tensor.expert_stride,
                        (object_digest, pack_digest),
                    )
                    part_records.append(
                        {
                            "part": tensor.part,
                            "tensor_name": tensor.tensor_name,
                            "tensor_type": tensor.tensor_type,
                            "shape": list(tensor.shape),
                            "source_offset": source_offset,
                            "pack_offset": part_pack_offset,
                            "valid_length": tensor.expert_stride,
                            "sha256": checksum,
                        }
                    )
                    payload_size += tensor.expert_stride
                    pack_offset += tensor.expert_stride

                padded_length = align_up(payload_size, alignment)
                padding_length = padded_length - payload_size
                if padding_length:
                    write_zeros(target, padding_length, pack_digest)
                    pack_offset += padding_length
                object_records.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        "offset": object_offset,
                        "valid_length": payload_size,
                        "padded_length": padded_length,
                        "padding_length": padding_length,
                        "valid_sha256": object_digest.hexdigest(),
                        "parts": part_records,
                    }
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_pack, pack_path)
    except BaseException:
        try:
            temporary_pack.unlink()
        except FileNotFoundError:
            pass
        raise

    pack_size = pack_path.stat().st_size
    if pack_size != pack_offset:
        raise PackError(f"pack size mismatch: expected {pack_offset}, got {pack_size}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": "pdcat-expert-pack",
        "created_at_utc": utc_now(),
        "alignment": alignment,
        "model": {
            "path": str(model_path),
            "size_bytes": inspection["model"]["size_bytes"],
            "sha256": expected_model_hash,
            "sha256_verified_during_pack": actual_model_hash is not None,
            "architecture": inspection["topology"]["architecture"],
            "general_name": inspection["model"].get("general_name"),
            "file_type": inspection["model"].get("file_type"),
        },
        "tensor_adapter": inspection["tensor_adapter"],
        "topology": inspection["topology"],
        "pack": {
            "path": str(pack_path),
            "file_name": pack_path.name,
            "size_bytes": pack_size,
            "sha256": pack_digest.hexdigest(),
            "read_only": True,
        },
        "object_count": len(object_records),
        "objects": object_records,
    }

    verification = verify_pack_data(
        manifest,
        pack_path,
        model_path=model_path,
        verify_source=True,
    )
    manifest["verification"] = verification
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        try:
            temporary_manifest.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(pack_path, 0o444)
    os.chmod(manifest_path, 0o444)
    return manifest


def read_exact(source: BinaryIO, offset: int, size: int) -> bytes:
    source.seek(offset)
    value = source.read(size)
    if len(value) != size:
        raise PackError(f"short read at offset {offset}: expected {size}, got {len(value)}")
    return value


def verify_pack_data(
    manifest: dict[str, Any],
    pack_path: Path,
    model_path: Path | None,
    verify_source: bool,
) -> dict[str, Any]:
    alignment = int(manifest["alignment"])
    expected_size = int(manifest["pack"]["size_bytes"])
    actual_size = pack_path.stat().st_size
    if actual_size != expected_size:
        raise PackError(
            f"pack size mismatch: expected {expected_size}, got {actual_size}"
        )

    previous_end = 0
    checked_parts = 0
    checked_objects = 0
    with pack_path.open("rb") as packed:
        source_context = model_path.open("rb") if verify_source and model_path else None
        try:
            for obj in manifest["objects"]:
                offset = int(obj["offset"])
                valid_length = int(obj["valid_length"])
                padded_length = int(obj["padded_length"])
                if offset % alignment or padded_length % alignment:
                    raise PackError(
                        f"unaligned object layer={obj['layer']} expert={obj['expert']}"
                    )
                if offset < previous_end or offset + padded_length > actual_size:
                    raise PackError(
                        f"invalid object bounds layer={obj['layer']} expert={obj['expert']}"
                    )

                object_digest = hashlib.sha256()
                payload_end = offset
                for part in obj["parts"]:
                    part_offset = int(part["pack_offset"])
                    part_size = int(part["valid_length"])
                    if part_offset != payload_end:
                        raise PackError(
                            f"non-contiguous part layout at layer={obj['layer']} "
                            f"expert={obj['expert']} part={part['part']}"
                        )
                    value = read_exact(packed, part_offset, part_size)
                    checksum = hashlib.sha256(value).hexdigest()
                    if checksum != part["sha256"]:
                        raise PackError(
                            f"pack checksum mismatch at layer={obj['layer']} "
                            f"expert={obj['expert']} part={part['part']}"
                        )
                    if source_context is not None:
                        source_value = read_exact(
                            source_context,
                            int(part["source_offset"]),
                            part_size,
                        )
                        if hashlib.sha256(source_value).hexdigest() != checksum:
                            raise PackError(
                                f"source checksum mismatch at layer={obj['layer']} "
                                f"expert={obj['expert']} part={part['part']}"
                            )
                    object_digest.update(value)
                    payload_end += part_size
                    checked_parts += 1

                if payload_end != offset + valid_length:
                    raise PackError(
                        f"object payload length mismatch at layer={obj['layer']} "
                        f"expert={obj['expert']}"
                    )
                if object_digest.hexdigest() != obj["valid_sha256"]:
                    raise PackError(
                        f"object checksum mismatch at layer={obj['layer']} "
                        f"expert={obj['expert']}"
                    )
                padding = read_exact(
                    packed,
                    offset + valid_length,
                    padded_length - valid_length,
                )
                if any(padding):
                    raise PackError(
                        f"non-zero padding at layer={obj['layer']} expert={obj['expert']}"
                    )
                previous_end = offset + padded_length
                checked_objects += 1
        finally:
            if source_context is not None:
                source_context.close()

    if checked_objects != int(manifest["object_count"]):
        raise PackError(
            f"object count mismatch: expected {manifest['object_count']}, "
            f"checked {checked_objects}"
        )
    return {
        "verified_at_utc": utc_now(),
        "pack_bytes_checked": actual_size,
        "objects_checked": checked_objects,
        "parts_checked": checked_parts,
        "source_bytes_checked": verify_source,
        "valid": True,
    }


def verify_manifest(
    manifest_path: Path,
    pack_override: Path | None,
    model_override: Path | None,
    verify_model_hash: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("format") != "pdcat-expert-pack":
        raise PackError("unsupported sidecar manifest format")

    raw_pack_path = pack_override or Path(manifest["pack"]["path"])
    if not raw_pack_path.is_absolute():
        raw_pack_path = manifest_path.parent / raw_pack_path
    pack_path = raw_pack_path.resolve()
    model_path = model_override or Path(manifest["model"]["path"])
    if not model_path.is_absolute():
        model_path = manifest_path.parent / model_path
    model_path = model_path.resolve()

    if verify_model_hash:
        actual = sha256_file(model_path)
        expected = str(manifest["model"]["sha256"]).lower()
        if actual.lower() != expected:
            raise PackError(
                f"model SHA-256 mismatch: expected {expected}, got {actual}"
            )
    result = verify_pack_data(
        manifest,
        pack_path,
        model_path=model_path,
        verify_source=True,
    )
    pack_sha = sha256_file(pack_path)
    if pack_sha != manifest["pack"]["sha256"]:
        raise PackError(
            f"pack SHA-256 mismatch: expected {manifest['pack']['sha256']}, "
            f"got {pack_sha}"
        )
    result["pack_sha256_verified"] = True
    result["model_sha256_verified"] = verify_model_hash
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-model-hash", action="store_true")
    parser.add_argument("--model", type=Path, help="Model override for verify-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    try:
        if args.verify_only:
            if args.manifest is None:
                raise PackError("--manifest is required with --verify-only")
            result = verify_manifest(
                args.manifest.resolve(),
                args.pack.resolve() if args.pack else None,
                args.model.resolve() if args.model else None,
                args.verify_model_hash,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        config = load_json(config_path)
        inspection = inspect_from_run_config(config, config_path, hash_model=False)
        model_path = Path(inspection["model"]["path"])
        default_pack, default_manifest = default_paths(model_path)
        pack_path = resolve_output(args.pack, default_pack)
        manifest_path = resolve_output(args.manifest, default_manifest)
        manifest = build_pack(
            config_path,
            pack_path,
            manifest_path,
            args.alignment,
            args.verify_model_hash,
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "pack": manifest["pack"],
                    "object_count": manifest["object_count"],
                    "verification": manifest["verification"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, KeyError, TypeError, ValueError, ModelInspectionError, PackError) as error:
        print(f"sidecar pack error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
