#!/usr/bin/env python3
"""Run one reproducible PDCat llama-cli experiment from a versioned JSON config."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .monitor_pdcat_process import summarize_jsonl as summarize_resource_jsonl
    from .pdcat_model import ModelInspectionError, inspect_from_run_config, sha256_file
    from .summarize_pdcat_io import TraceError, summarize_jsonl
except ImportError:
    from monitor_pdcat_process import summarize_jsonl as summarize_resource_jsonl
    from pdcat_model import ModelInspectionError, inspect_from_run_config, sha256_file
    from summarize_pdcat_io import TraceError, summarize_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "pdcat" / "configs" / "deepseek-v2-lite-q8_0.json"
)
DEFAULT_REPOSITORIES = (
    REPO_ROOT,
    REPO_ROOT.parent / "InterfaceIO",
    REPO_ROOT.parent / "EK-Edge",
)
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MODEL_FLAGS = {"-m", "--model"}
PROMPT_FLAGS = {"-p", "--prompt"}
PREDICT_FLAGS = {"-n", "--n-predict"}
EXPERT_PACK_FLAGS = {"--expert-pack-manifest"}
SINGLE_VALUE_OVERRIDE_FLAGS = {"--expert-cache-capacity"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
LLAMA_PERF_LINE = re.compile(
    r"^(?:llama_perf_context_print|common_perf_print):\s*"
    r"(?P<label>load time|prompt eval time|eval time|total time)\s*="
    r"\s*(?P<milliseconds>[0-9]+(?:\.[0-9]+)?) ms"
    r"(?:\s*/\s*(?P<count>[0-9]+)\s*(?P<unit>tokens|runs)"
    r"(?:\s*\(\s*(?P<ms_per>[0-9]+(?:\.[0-9]+)?) ms per token,\s*"
    r"(?P<tokens_per_second>[0-9]+(?:\.[0-9]+)?) tokens per second\))?)?\s*$"
)


class ConfigError(ValueError):
    """Raised for an invalid PDCat run configuration."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level JSON value must be an object")
    return data


def require_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return value


def require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{context} must be an array of strings")
    return list(value)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    require_string(config, "experiment_id", "config")

    model = config.get("model")
    if not isinstance(model, dict):
        raise ConfigError("config.model must be an object")
    require_string(model, "path", "config.model")
    require_string(model, "sha256", "config.model")
    if not isinstance(model.get("tensor_adapter"), dict):
        raise ConfigError("config.model.tensor_adapter must be an object")

    expert_pack = model.get("expert_pack")
    if expert_pack is not None:
        if not isinstance(expert_pack, dict):
            raise ConfigError("config.model.expert_pack must be an object")
        for key in ("manifest", "path"):
            require_string(expert_pack, key, "config.model.expert_pack")
        for key in ("manifest_sha256", "sha256"):
            value = require_string(expert_pack, key, "config.model.expert_pack")
            if not SHA256_PATTERN.fullmatch(value):
                raise ConfigError(
                    f"config.model.expert_pack.{key} must be a SHA-256 digest"
                )
        for key in ("size_bytes", "object_count"):
            value = expert_pack.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"config.model.expert_pack.{key} must be a positive integer"
                )
        alignment = expert_pack.get("alignment")
        if (
            not isinstance(alignment, int)
            or alignment < 4096
            or alignment & (alignment - 1)
        ):
            raise ConfigError(
                "config.model.expert_pack.alignment must be a power of two >= 4096"
            )

    command = config.get("command")
    if not isinstance(command, dict):
        raise ConfigError("config.command must be an object")
    require_string(command, "binary", "config.command")
    base_args = require_string_list(command.get("base_args", []), "config.command.base_args")

    workload = config.get("workload")
    if not isinstance(workload, dict):
        raise ConfigError("config.workload must be an object")
    require_string(workload, "prompt", "config.workload")
    if not isinstance(workload.get("n_predict"), int) or workload["n_predict"] < 0:
        raise ConfigError("config.workload.n_predict must be a non-negative integer")

    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError("config.profiles must be a non-empty object")
    default_profile = require_string(config, "default_profile", "config")
    if default_profile not in profiles:
        raise ConfigError(f"default profile {default_profile!r} is not defined")

    all_args = list(base_args)
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("profile names must be non-empty strings")
        if not isinstance(profile, dict):
            raise ConfigError(f"config.profiles.{name} must be an object")
        use_expert_pack = profile.get("use_expert_pack", False)
        if not isinstance(use_expert_pack, bool):
            raise ConfigError(
                f"config.profiles.{name}.use_expert_pack must be a boolean"
            )
        if use_expert_pack and expert_pack is None:
            raise ConfigError(
                f"config.profiles.{name} requires config.model.expert_pack"
            )
        all_args.extend(
            require_string_list(profile.get("args", []), f"config.profiles.{name}.args")
        )
        environment = profile.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, (str, int, float, bool))
            for key, value in environment.items()
        ):
            raise ConfigError(
                f"config.profiles.{name}.environment must contain scalar values"
            )

    forbidden = MODEL_FLAGS | PROMPT_FLAGS | PREDICT_FLAGS | EXPERT_PACK_FLAGS
    collisions = sorted({argument for argument in all_args if argument in forbidden})
    if collisions:
        raise ConfigError(
            "model, prompt, and n-predict are structured fields and cannot appear in "
            f"raw argument lists: {collisions}"
        )


def resolve_from_repo(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()

def expand_environment(values: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value).replace("${REPO_ROOT}", str(REPO_ROOT))
        for key, value in values.items()
    }


def inspect_expert_pack(
    config: dict[str, Any],
    profile_name: str,
    verify_pack_hash: bool,
) -> dict[str, Any] | None:
    profile = config["profiles"][profile_name]
    if not profile.get("use_expert_pack", False):
        return None

    configured = config["model"]["expert_pack"]
    manifest_path = resolve_from_repo(configured["manifest"])
    pack_path = resolve_from_repo(configured["path"])
    manifest = read_json(manifest_path)

    if manifest.get("schema_version") != 1 or manifest.get("format") != "pdcat-expert-pack":
        raise ConfigError(f"{manifest_path}: unsupported expert pack manifest")
    if not isinstance(manifest.get("model"), dict):
        raise ConfigError(f"{manifest_path}: model metadata is missing")
    if not isinstance(manifest.get("pack"), dict):
        raise ConfigError(f"{manifest_path}: pack metadata is missing")
    if not isinstance(manifest.get("verification"), dict) or not manifest[
        "verification"
    ].get("valid", False):
        raise ConfigError(f"{manifest_path}: pack verification is not valid")

    try:
        actual_manifest_hash = sha256_file(manifest_path)
        actual_pack_size = pack_path.stat().st_size
    except OSError as error:
        raise ConfigError(f"cannot inspect expert pack: {error}") from error

    if actual_manifest_hash.lower() != configured["manifest_sha256"].lower():
        raise ConfigError(
            "expert pack manifest SHA-256 mismatch: expected "
            f"{configured['manifest_sha256']}, got {actual_manifest_hash}"
        )

    manifest_model = manifest["model"]
    expected_model_hash = config["model"]["sha256"].lower()
    if str(manifest_model.get("sha256", "")).lower() != expected_model_hash:
        raise ConfigError("expert pack manifest is bound to a different model SHA-256")
    if int(manifest_model.get("size_bytes", -1)) != int(config["model"]["size_bytes"]):
        raise ConfigError("expert pack manifest model size does not match the run config")

    manifest_pack = manifest["pack"]
    raw_manifest_pack_path = Path(
        require_string(manifest_pack, "path", "expert pack manifest.pack")
    )
    if not raw_manifest_pack_path.is_absolute():
        raw_manifest_pack_path = manifest_path.parent / raw_manifest_pack_path
    manifest_pack_path = raw_manifest_pack_path.resolve()
    if manifest_pack_path != pack_path:
        raise ConfigError(
            f"expert pack path mismatch: config has {pack_path}, "
            f"manifest has {manifest_pack_path}"
        )
    if actual_pack_size != int(configured["size_bytes"]):
        raise ConfigError(
            f"expert pack size mismatch: expected {configured['size_bytes']}, "
            f"got {actual_pack_size}"
        )
    if int(manifest_pack.get("size_bytes", -1)) != actual_pack_size:
        raise ConfigError("expert pack manifest file size does not match the pack")
    expected_pack_hash = configured["sha256"].lower()
    if str(manifest_pack.get("sha256", "")).lower() != expected_pack_hash:
        raise ConfigError("expert pack SHA-256 differs between config and manifest")
    if int(manifest.get("alignment", -1)) != int(configured["alignment"]):
        raise ConfigError("expert pack alignment differs between config and manifest")
    if int(manifest.get("object_count", -1)) != int(configured["object_count"]):
        raise ConfigError("expert pack object count differs between config and manifest")

    actual_pack_hash = None
    if verify_pack_hash:
        try:
            actual_pack_hash = sha256_file(pack_path)
        except OSError as error:
            raise ConfigError(f"cannot hash expert pack: {error}") from error
        if actual_pack_hash.lower() != expected_pack_hash:
            raise ConfigError(
                f"expert pack SHA-256 mismatch: expected {expected_pack_hash}, "
                f"got {actual_pack_hash}"
            )

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "pack_path": str(pack_path),
        "pack_size_bytes": actual_pack_size,
        "pack_sha256_expected": expected_pack_hash,
        "pack_sha256_verified": actual_pack_hash is not None,
        "alignment": int(configured["alignment"]),
        "object_count": int(configured["object_count"]),
        "source_bytes_verified_during_pack": bool(
            manifest["verification"].get("source_bytes_checked", False)
        ),
        "verified_at_utc": manifest["verification"].get("verified_at_utc"),
    }


def git_result(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def git_snapshot(repo: Path) -> dict[str, Any]:
    if not repo.is_dir():
        return {"path": str(repo), "exists": False}
    commit = git_result(repo, "rev-parse", "HEAD")
    branch = git_result(repo, "branch", "--show-current")
    status = git_result(repo, "status", "--porcelain=v1")
    remote = git_result(repo, "remote", "get-url", "origin")
    return {
        "path": str(repo),
        "exists": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "origin": remote.stdout.strip() if remote.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "working_tree": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def runtime_artifact_snapshot(binary: Path) -> list[dict[str, Any]]:
    """Hash the executable and locally built shared objects selected by ldd."""
    resolved_binary = binary.resolve()
    local_directory = resolved_binary.parent
    paths = {resolved_binary}
    linked = subprocess.run(
        ["ldd", str(resolved_binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if linked.returncode == 0:
        for line in linked.stdout.splitlines():
            candidate: str | None = None
            if "=>" in line:
                target = line.split("=>", 1)[1].strip().split(maxsplit=1)
                if target and target[0].startswith("/"):
                    candidate = target[0]
            else:
                target = line.strip().split(maxsplit=1)
                if target and target[0].startswith("/"):
                    candidate = target[0]
            if candidate is None:
                continue
            dependency = Path(candidate).resolve()
            if dependency.parent == local_directory and dependency.is_file():
                paths.add(dependency)

    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=str)
    ]


def compact_inspection(inspection: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(inspection)
    result["expert_tensors"] = {
        "count": len(inspection.get("expert_tensors", [])),
        "stored_in": "model_manifest.json",
    }
    return result


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_llama_performance(path: Path) -> dict[str, Any]:
    labels = {
        "load time": "load",
        "prompt eval time": "prompt_eval",
        "eval time": "decode_eval",
        "total time": "total",
    }
    result: dict[str, Any] = {}
    with path.open("r", encoding="utf-8", errors="replace") as source_file:
        for raw_line in source_file:
            match = LLAMA_PERF_LINE.match(raw_line.strip())
            if match is None:
                continue
            row: dict[str, Any] = {
                "milliseconds": float(match.group("milliseconds")),
            }
            if match.group("count") is not None:
                row.update(
                    {
                        "count": int(match.group("count")),
                        "count_unit": match.group("unit"),
                    }
                )
            if match.group("ms_per") is not None:
                row.update(
                    {
                        "milliseconds_per_token": float(match.group("ms_per")),
                        "tokens_per_second": float(match.group("tokens_per_second")),
                    }
                )
            result[labels[match.group("label")]] = row
    return result


def default_run_id(experiment_id: str, profile: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{experiment_id}-{profile}"


def apply_single_value_overrides(
    configured_args: list[str],
    extra_args: list[str],
) -> list[str]:
    overrides = {
        flag
        for flag in SINGLE_VALUE_OVERRIDE_FLAGS
        if flag in extra_args or any(arg.startswith(flag + "=") for arg in extra_args)
    }
    if not overrides:
        return configured_args

    result: list[str] = []
    index = 0
    while index < len(configured_args):
        argument = configured_args[index]
        matched = next(
            (
                flag
                for flag in overrides
                if argument == flag or argument.startswith(flag + "=")
            ),
            None,
        )
        if matched is None:
            result.append(argument)
            index += 1
            continue
        index += 2 if argument == matched else 1
    return result


def command_for(
    config: dict[str, Any],
    inspection: dict[str, Any],
    profile_name: str,
    expert_pack: dict[str, Any] | None,
    binary_override: Path | None,
    prompt_override: str | None,
    predict_override: int | None,
    extra_args: list[str],
) -> tuple[list[str], dict[str, str], Path]:
    command_config = config["command"]
    binary = binary_override or resolve_from_repo(command_config["binary"])
    profile = config["profiles"][profile_name]
    prompt = prompt_override if prompt_override is not None else config["workload"]["prompt"]
    n_predict = (
        predict_override
        if predict_override is not None
        else config["workload"]["n_predict"]
    )

    sidecar_args = (
        ["--expert-pack-manifest", expert_pack["manifest_path"]]
        if expert_pack is not None
        else []
    )
    base_args = list(command_config.get("base_args", []))
    if binary.name == "llama-debug":
        base_args = [arg for arg in base_args if arg != "--no-display-prompt"]
    configured_args = apply_single_value_overrides(
        [*base_args, *profile.get("args", [])],
        extra_args,
    )
    command = [
        str(binary),
        "--model",
        inspection["model"]["path"],
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
        *configured_args,
        *sidecar_args,
        *extra_args,
    ]
    environment = expand_environment(command_config.get("environment", {}))
    environment.update(expand_environment(profile.get("environment", {})))
    return command, environment, binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Named profile from the config")
    parser.add_argument("--binary", type=Path, help="Override llama-cli binary")
    parser.add_argument("--prompt", help="Override the configured smoke-test prompt")
    parser.add_argument("--n-predict", type=int, help="Override generated token count")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--verify-model-hash", action="store_true")
    parser.add_argument(
        "--verify-expert-pack-hash",
        action="store_true",
        help="Hash the full sidecar pack before starting the run",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Sample process RSS/CPU/I/O/temperature and tegrastats when available",
    )
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    try:
        config = read_json(config_path)
        validate_config(config)
        profile_name = args.profile or config["default_profile"]
        if profile_name not in config["profiles"]:
            raise ConfigError(f"profile {profile_name!r} is not defined")
        if args.n_predict is not None and args.n_predict < 0:
            raise ConfigError("--n-predict must be non-negative")
        if args.monitor_interval <= 0:
            raise ConfigError("--monitor-interval must be positive")
        inspection = inspect_from_run_config(
            config, config_path, hash_model=args.verify_model_hash
        )
        expert_pack = inspect_expert_pack(
            config, profile_name, args.verify_expert_pack_hash
        )
    except (ConfigError, ModelInspectionError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    expected_hash = config["model"]["sha256"].lower()
    actual_hash = inspection["model"].get("sha256")
    if actual_hash is not None and actual_hash.lower() != expected_hash:
        print(
            f"model SHA-256 mismatch: expected {expected_hash}, got {actual_hash}",
            file=sys.stderr,
        )
        return 2

    extra_args: list[str] = []
    for value in args.extra_arg:
        try:
            parsed = shlex.split(value)
        except ValueError as error:
            print(f"invalid --extra-arg value: {error}", file=sys.stderr)
            return 2
        extra_args.extend(parsed)

    environment_overrides: dict[str, str] = {}
    for value in args.env:
        key, separator, item = value.partition("=")
        if not separator or not ENV_NAME.fullmatch(key):
            print(f"invalid --env {value!r}; expected KEY=VALUE", file=sys.stderr)
            return 2
        environment_overrides[key] = item

    command, configured_environment, binary = command_for(
        config,
        inspection,
        profile_name,
        expert_pack,
        args.binary.resolve() if args.binary else None,
        args.prompt,
        args.n_predict,
        extra_args,
    )
    configured_environment.update(environment_overrides)
    if not binary.is_file():
        print(f"llama-cli binary does not exist: {binary}", file=sys.stderr)
        return 2
    if not os.access(binary, os.X_OK):
        print(f"llama-cli binary is not executable: {binary}", file=sys.stderr)
        return 2

    run_id = args.run_id or default_run_id(config["experiment_id"], profile_name)
    if not SAFE_RUN_ID.fullmatch(run_id):
        print(
            "run id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or dash",
            file=sys.stderr,
        )
        return 2

    output_root = args.output_root
    if output_root is None:
        output_root = resolve_from_repo(
            config.get("output_root", "experiments/pdcat/runs")
        )
    elif not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    run_dir = output_root / run_id
    trace_config = config.get("trace", {})
    configured_environment["PDCAT_RUN_ID"] = run_id
    configured_environment.setdefault("PDCAT_REQUEST_ID", "request-0")
    artifacts: dict[str, str] = {}
    if trace_config.get("moe_gate", False):
        gate_trace = run_dir / "moe_gate.jsonl"
        configured_environment["LLAMA_MOE_GATE_TRACE_JSONL"] = str(gate_trace)
        artifacts["moe_gate_jsonl"] = str(gate_trace)
    if trace_config.get("expert_io", False):
        io_trace = run_dir / "expert_io.jsonl"
        configured_environment["PDCAT_IO_TRACE_PATH"] = str(io_trace)
        artifacts["expert_io_jsonl"] = str(io_trace)
        artifacts["expert_io_summary"] = str(run_dir / "expert_io_summary.json")
    monitor_script = REPO_ROOT / "tools" / "monitor_pdcat_process.py"
    tegrastats_path = shutil.which("tegrastats") if args.monitor else None
    if args.monitor:
        artifacts["resource_samples_jsonl"] = str(run_dir / "resource_samples.jsonl")
        artifacts["resource_summary"] = str(run_dir / "resource_summary.json")
        artifacts["resource_monitor_stderr"] = str(
            run_dir / "resource_monitor.stderr.log"
        )
        if tegrastats_path is not None:
            artifacts["tegrastats_log"] = str(run_dir / "tegrastats.log")
            artifacts["tegrastats_stderr"] = str(
                run_dir / "tegrastats.stderr.log"
            )



    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "profile": profile_name,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at_utc": utc_now(),
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
        },
        "repositories": [git_snapshot(repo) for repo in DEFAULT_REPOSITORIES],
        "runtime_artifacts": runtime_artifact_snapshot(binary),
        "model": compact_inspection(inspection),
        "model_sha256_expected": expected_hash,
        "model_sha256_verified": actual_hash is not None,
        "expert_pack": expert_pack,
        "command": command,
        "command_display": shlex.join(command),
        "configured_environment": configured_environment,
        "artifacts": artifacts,
        "monitoring": {
            "enabled": args.monitor,
            "interval_seconds": args.monitor_interval,
            "resource_sampler": str(monitor_script) if args.monitor else None,
            "tegrastats_available": tegrastats_path is not None,
            "tegrastats_path": tegrastats_path,
        },
        "status": "dry-run" if args.dry_run else "pending",
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"run directory already exists: {run_dir}", file=sys.stderr)
        return 2


    write_json(run_dir / "config.snapshot.json", config)
    write_json(run_dir / "model_manifest.json", inspection)
    manifest["status"] = "running"
    manifest["started_at_utc"] = utc_now()
    write_json(run_dir / "run_manifest.json", manifest)

    environment = os.environ.copy()
    environment.update(configured_environment)
    start = time.monotonic()
    resource_monitor: subprocess.Popen[bytes] | None = None
    tegrastats_monitor: subprocess.Popen[bytes] | None = None
    monitor_streams: list[Any] = []
    with (run_dir / "stdout.log").open("wb") as stdout_file, (
        run_dir / "stderr.log"
    ).open("wb") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
            )
            manifest["monitoring"]["target_pid"] = process.pid
            if args.monitor:
                resource_stderr = Path(
                    artifacts["resource_monitor_stderr"]
                ).open("wb")
                monitor_streams.append(resource_stderr)
                resource_monitor = subprocess.Popen(
                    [
                        sys.executable,
                        str(monitor_script),
                        "--pid",
                        str(process.pid),
                        "--run-id",
                        run_id,
                        "--output",
                        artifacts["resource_samples_jsonl"],
                        "--interval",
                        str(args.monitor_interval),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=resource_stderr,
                )
                if tegrastats_path is not None:
                    tegrastats_output = Path(artifacts["tegrastats_log"]).open("wb")
                    tegrastats_stderr = Path(
                        artifacts["tegrastats_stderr"]
                    ).open("wb")
                    monitor_streams.extend([tegrastats_output, tegrastats_stderr])
                    tegrastats_monitor = subprocess.Popen(
                        [
                            tegrastats_path,
                            "--interval",
                            str(max(1, int(args.monitor_interval * 1000))),
                        ],
                        stdout=tegrastats_output,
                        stderr=tegrastats_stderr,
                    )
            write_json(run_dir / "run_manifest.json", manifest)
            returncode = process.wait()
        except OSError as error:
            manifest["execution_error"] = str(error)
            returncode = 127
        finally:
            if resource_monitor is not None:
                try:
                    resource_monitor.wait(timeout=max(2.0, args.monitor_interval * 2))
                except subprocess.TimeoutExpired:
                    resource_monitor.terminate()
                    try:
                        resource_monitor.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        resource_monitor.kill()
                        resource_monitor.wait()
            if tegrastats_monitor is not None:
                tegrastats_monitor.terminate()
                try:
                    tegrastats_monitor.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    tegrastats_monitor.kill()
                    tegrastats_monitor.wait()
            for stream in monitor_streams:
                stream.close()

    manifest["finished_at_utc"] = utc_now()
    manifest["elapsed_seconds"] = time.monotonic() - start
    manifest["returncode"] = returncode
    manifest["status"] = "completed" if returncode == 0 else "failed"

    try:
        performance = parse_llama_performance(run_dir / "stderr.log")
        if performance:
            manifest["llama_performance"] = performance
        elif returncode == 0:
            manifest["llama_performance_error"] = "no llama timing lines found"
    except OSError as error:
        manifest["llama_performance_error"] = str(error)

    io_trace_path = artifacts.get("expert_io_jsonl")
    if io_trace_path and Path(io_trace_path).is_file():
        try:
            summary = summarize_jsonl(Path(io_trace_path))
            write_json(Path(artifacts["expert_io_summary"]), summary)
            manifest["expert_io_summary"] = summary
        except (OSError, TraceError, ValueError) as error:
            manifest["expert_io_summary_error"] = str(error)
    resource_trace_path = artifacts.get("resource_samples_jsonl")
    if resource_trace_path and Path(resource_trace_path).is_file():
        try:
            summary = summarize_resource_jsonl(Path(resource_trace_path))
            write_json(Path(artifacts["resource_summary"]), summary)
            manifest["resource_summary"] = summary
        except (OSError, json.JSONDecodeError, ValueError) as error:
            manifest["resource_summary_error"] = str(error)



    write_json(run_dir / "run_manifest.json", manifest)

    print(run_dir)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
