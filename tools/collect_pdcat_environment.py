#!/usr/bin/env python3
"""Collect a reproducible Jetson/PDCat experiment environment snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


WORKSPACE = Path("/workspace")
DEFAULT_REPOSITORIES = (
    WORKSPACE / "llama_uring.cpp",
    WORKSPACE / "InterfaceIO",
    WORKSPACE / "EK-Edge",
)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip().rstrip("\x00")
    except OSError:
        return None


def run(command: list[str], timeout: int = 10) -> dict[str, Any]:
    executable = command[0]
    if "/" not in executable:
        resolved = shutil.which(executable)
        if resolved is None:
            return {"command": command, "available": False}
        command = [resolved, *command[1:]]

    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "available": True,
            "error": str(error),
        }

    return {
        "command": command,
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def run_json(command: list[str]) -> Any:
    result = run(command)
    if result.get("returncode") != 0:
        return {"error": result}
    try:
        return json.loads(result.get("stdout", ""))
    except json.JSONDecodeError as error:
        return {"error": str(error), "raw": result.get("stdout", "")}


def parse_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    text = read_text(Path("/etc/os-release")) or ""
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    text = read_text(Path("/proc/meminfo")) or ""
    for line in text.splitlines():
        key, _, remainder = line.partition(":")
        fields = remainder.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024
        values[key] = value
    return values


def tool_version(command: list[str]) -> dict[str, Any]:
    result = run(command)
    output = result.get("stdout") or result.get("stderr") or ""
    result["version"] = output.splitlines()[0] if output else None
    return result


def git(repo: Path, *args: str) -> dict[str, Any]:
    return run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args])


def repository_snapshot(repo: Path) -> dict[str, Any]:
    branch = git(repo, "branch", "--show-current")
    commit = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1")
    remote = git(repo, "remote", "get-url", "origin")
    return {
        "path": str(repo),
        "branch": branch.get("stdout"),
        "commit": commit.get("stdout"),
        "origin": remote.get("stdout") if remote.get("returncode") == 0 else None,
        "dirty": bool(status.get("stdout")),
        "working_tree": status.get("stdout", "").splitlines(),
    }


def nvme_snapshot() -> list[dict[str, Any]]:
    controllers: list[dict[str, Any]] = []
    for controller in sorted(Path("/sys/class/nvme").glob("nvme*")):
        if not controller.name[4:].isdigit():
            continue
        device = controller / "device"
        controllers.append(
            {
                "name": controller.name,
                "model": read_text(controller / "model"),
                "serial": read_text(controller / "serial"),
                "firmware_revision": read_text(controller / "firmware_rev"),
                "current_link_speed": read_text(device / "current_link_speed"),
                "current_link_width": read_text(device / "current_link_width"),
                "max_link_speed": read_text(device / "max_link_speed"),
                "max_link_width": read_text(device / "max_link_width"),
            }
        )
    return controllers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_snapshot(path: Path, hash_files: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "exists": True,
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path) if hash_files else None,
        }

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".gguf", ".safetensors"}
    )
    return {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "directory",
        "file_count": len(files),
        "size_bytes": sum(candidate.stat().st_size for candidate in files),
        "files": [
            {
                "path": str(candidate.resolve()),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256(candidate) if hash_files else None,
            }
            for candidate in files
        ],
    }


def collect(models: list[Path], hash_models: bool) -> dict[str, Any]:
    nvcc = Path("/usr/local/cuda/bin/nvcc")
    notes = [
        "Power/clock commands are unavailable in this container; record nvpmodel and jetson_clocks on the host before final runs.",
        "Record GGUF architecture, quantization, layers, experts, top-k, and expert byte layout in the run configuration.",
    ]
    if not hash_models:
        notes.append("Model SHA-256 was omitted; rerun with --hash-models before publishing results.")
    return {
        "schema_version": 1,
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": {
            "device_model": read_text(Path("/proc/device-tree/model")),
            "l4t_release": read_text(Path("/etc/nv_tegra_release")),
            "os_release": parse_os_release(),
            "kernel": platform.release(),
            "uname": " ".join(platform.uname()),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory": parse_meminfo(),
        },
        "software": {
            "nvcc": tool_version([str(nvcc), "--version"]) if nvcc.exists() else {"available": False},
            "gcc": tool_version(["gcc", "--version"]),
            "gxx": tool_version(["g++", "--version"]),
            "cmake": tool_version(["cmake", "--version"]),
            "rustc": tool_version(["rustc", "--version"]),
            "cargo": tool_version(["cargo", "--version"]),
        },
        "power_and_clocks": {
            "nvpmodel": run(["nvpmodel", "-q"]),
            "jetson_clocks": run(["jetson_clocks", "--show"]),
        },
        "storage": {
            "workspace_mount": run_json(["findmnt", "-J", "-T", str(WORKSPACE)]),
            "data_mount": run_json(["findmnt", "-J", "-T", "/data"]),
            "block_devices": run_json(
                ["lsblk", "-J", "-b", "-o", "NAME,MODEL,SERIAL,FSTYPE,SIZE,FSAVAIL,FSUSE%"]
            ),
            "nvme_controllers": nvme_snapshot(),
        },
        "repositories": [repository_snapshot(repo) for repo in DEFAULT_REPOSITORIES],
        "models": [model_snapshot(model, hash_models) for model in models],
        "notes": notes,
    }


def mib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1048576:.1f} MiB"


def markdown(snapshot: dict[str, Any]) -> str:
    platform_data = snapshot["platform"]
    memory = platform_data["memory"]
    l4t_release = (platform_data.get("l4t_release") or "unknown").splitlines()[0]
    lines = [
        "# PDCat experiment environment",
        "",
        f"- Collected at (UTC): `{snapshot['collected_at_utc']}`",
        f"- Device: {platform_data.get('device_model') or 'unknown'}",
        f"- Architecture/kernel: `{platform_data.get('architecture')} / {platform_data.get('kernel')}`",
        f"- L4T: `{l4t_release}`",
        f"- Memory total/available: {mib(memory.get('MemTotal'))} / {mib(memory.get('MemAvailable'))}",
        "",
        "## Software",
        "",
        "| Tool | Version |",
        "| --- | --- |",
    ]
    for name, detail in snapshot["software"].items():
        lines.append(f"| {name} | {detail.get('version') or 'unavailable'} |")

    lines.extend(["", "## Repositories", "", "| Repository | Branch | Commit | Dirty |", "| --- | --- | --- | --- |"])
    for repo in snapshot["repositories"]:
        lines.append(
            f"| `{repo['path']}` | `{repo.get('branch') or '-'}` | "
            f"`{repo.get('commit') or '-'}` | {'yes' if repo['dirty'] else 'no'} |"
        )

    lines.extend(["", "## NVMe", "", "| Controller | Model | Firmware | Link |", "| --- | --- | --- | --- |"])
    for controller in snapshot["storage"]["nvme_controllers"]:
        link = f"{controller.get('current_link_speed') or '?'} x{controller.get('current_link_width') or '?'}"
        lines.append(
            f"| {controller['name']} | {controller.get('model') or 'unknown'} | "
            f"{controller.get('firmware_revision') or 'unknown'} | {link} |"
        )

    lines.extend(["", "## Candidate models", ""])
    if not snapshot["models"]:
        lines.append("No primary model was supplied. Select a runnable MoE GGUF before baseline and Gate A.")
    for model in snapshot["models"]:
        lines.append(
            f"- `{model['path']}`: exists={model['exists']}, "
            f"size={mib(model.get('size_bytes'))}, sha256=`{model.get('sha256') or 'pending'}`"
        )

    lines.extend(["", "## Outstanding controls", ""])
    lines.extend(f"- {note}" for note in snapshot["notes"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/pdcat"))
    parser.add_argument("--model", action="append", type=Path, default=[])
    parser.add_argument("--hash-models", action="store_true")
    args = parser.parse_args()

    snapshot = collect(args.model, args.hash_models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "environment.md").write_text(markdown(snapshot), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
