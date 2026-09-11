#!/usr/bin/env python3
"""Build or migrate a LlamaFactory tokenized cache with a persisted length column.

The script supports PT and SFT YAML files.  If ``--source-cache`` is supplied,
it migrates that existing cache to the configured or explicitly selected output.
Otherwise it reuses the configured cache when present, or invokes LlamaFactory's
normal preprocessing pipeline when absent.
The final cache is written through a sibling temporary directory and renamed
only after schema, row-count, full length, Arrow-payload, and provenance checks
pass.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "length_cache_manifest.json"
MANIFEST_REFRESH_LOCK_NAME = ".length_cache_manifest.refresh.lock"
MANIFEST_SCHEMA_VERSION = 2
PT_EXP2_DATASET = "BrickNet-PT-exp2-text8m"
PT_EXP2_EXPECTED_ROWS = 7_698_261
PT_EXP2_EVAL_DATASET = "BrickNet-PT-exp2-text-val1000"
PT_EXP2_EVAL_EXPECTED_ROWS = 1_000
PT_EXP2_EVAL_SCHEMA_VERSION = 1
PT_EXP2_EVAL_EXPERIMENT = "PT-exp2-text8m"
PT_EXP2_EVAL_ROLE = "official_style_pt_loss_validation"
PT_EXP2_EVAL_SEED = 0
PT_EXP2_MODEL = "Qwen/Qwen3.5-0.8B"
PT_EXP2_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PT_EXP2_SNAPSHOT = (
    Path("/home/jiahao/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B")
    / "snapshots"
    / PT_EXP2_MODEL_REVISION
)
# These are the files used by the reviewed local Qwen snapshot.  Keeping the
# complete five-file identity here makes a tokenizer-only cache drift
# detectable while retaining the model-weight/index binding used by the
# launcher.
PT_EXP2_SNAPSHOT_HASHES = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "model.safetensors.index.json": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    "model.safetensors-00001-of-00001.safetensors": (
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
    ),
}
# The first PT-exp2 text8m cache was published before the builder reloaded
# its DatasetDict after ``save_to_disk``.  datasets therefore persisted the
# same rows/schema with different in-memory fingerprints.  These values are
# deliberately narrow: ``--refresh-manifest`` may repair only this reviewed
# cache drift and the builder identity recorded by that cache.
PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256 = "c081a3e31bca133b18093a98652cb3581129068c9b132870d5e81426a8f0e513"
PT_EXP2_LEGACY_CACHE_FINGERPRINTS = {
    "train": "9dee8e7009b4e5ef",
    "validation": "bf6aa8cb5302a855",
}
# These are the live fingerprints observed after reloading the published
# cache.  Both sides of the migration are fixed: accepting merely the old
# recorded values would allow an arbitrary, previously unseen live cache
# identity to be blessed by ``--refresh-manifest``.
PT_EXP2_STABLE_CACHE_FINGERPRINTS = {
    "train": "073292293b123a8e",
    "validation": "6e18a6ca8e96df18",
}
# The repair is tied to one reviewed manifest, not to any cache that happens
# to have the same shape.  The original bytes are the historical manifest
# before the repair record is added.
PT_EXP2_LEGACY_MANIFEST_SHA256 = "dc40d74dfbb81a31825fed2df0ae0f06c0d6e268352b6722bbe5b9d24fa39dbb"
PT_EXP2_MANIFEST_REPAIR_KIND = "pt_exp2_cache_manifest_repair"
PT_EXP2_MANIFEST_REPAIR_SCHEMA_VERSION = 1
PT_EXP2_MANIFEST_REPAIRED_FIELDS = [
    "builder.script_sha256",
    "cache_metadata_evidence",
]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _builder_script_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_json_bytes(value: Any) -> bytes:
    """Serialize a manifest exactly as this builder has historically written it."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _dataset_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise ValueError(f"dataset selection must be a string or list of strings, got {value!r}")


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("llamafactory", "datasets", "transformers", "pyarrow", "omegaconf"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _git_state() -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(ROOT), "commit": None, "dirty": None, "status_sha256": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return result
    result.update({"commit": commit, "dirty": bool(status), "status_sha256": _json_sha256(status)})
    return result


def _dataset_registry_contract(config: dict[str, Any]) -> dict[str, Any]:
    train_datasets = _dataset_names(config.get("dataset"))
    eval_datasets = _dataset_names(config.get("eval_dataset"))
    selected = list(dict.fromkeys([*train_datasets, *eval_datasets]))
    dataset_dir = _resolve(config.get("dataset_dir", "data")).resolve()
    registry_path = dataset_dir / "dataset_info.json"
    entries: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            raise ValueError(f"dataset registry is not a mapping: {registry_path}")
        entries = {name: registry.get(name) for name in selected}
    elif PT_EXP2_DATASET in train_datasets:
        raise FileNotFoundError(f"PT-exp2 dataset registry is missing: {registry_path}")
    return {
        "train": train_datasets,
        "eval": eval_datasets,
        "dataset_dir": str(dataset_dir),
        "registry": str(registry_path),
        # Only selected entries are bound. Unrelated registry additions must
        # not stale an already-built cache.
        "selected_entries": entries,
        "selected_entries_sha256": _json_sha256(entries),
    }


def _strict_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _strict_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string, got {value!r}")
    return value


def _resolve_strict_registry_file(dataset_contract: dict[str, Any], dataset_name: str) -> Path:
    """Resolve a selected registry file while rejecting path escapes/symlinks."""
    entry = dataset_contract["selected_entries"].get(dataset_name)
    if not isinstance(entry, dict) or not isinstance(entry.get("file_name"), str):
        raise ValueError(f"{dataset_name} has no valid file_name in the dataset registry")
    file_name = entry["file_name"]
    if not file_name or "\x00" in file_name or "\\" in file_name:
        raise ValueError(f"{dataset_name} file_name is unsafe: {file_name!r}")

    raw = Path(file_name)
    if any(part in {".", ".."} for part in raw.parts):
        raise ValueError(f"{dataset_name} file_name is unsafe: {file_name!r}")
    dataset_dir = Path(dataset_contract["dataset_dir"]).resolve()
    candidate = raw if raw.is_absolute() else dataset_dir / raw
    try:
        relative = candidate.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError(f"{dataset_name} file_name escapes dataset_dir: {file_name!r}") from exc

    current = dataset_dir
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{dataset_name} file_name traverses a symlink: {file_name!r}")

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError(f"{dataset_name} file_name resolves outside dataset_dir: {file_name!r}") from exc
    return resolved


def _pt_exp2_snapshot_contract() -> dict[str, Any]:
    """Return the verified local snapshot identity without any hub access."""
    snapshot = PT_EXP2_SNAPSHOT.resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"PT-exp2 model snapshot is missing: {snapshot}")
    files: dict[str, str] = {}
    for name, expected in PT_EXP2_SNAPSHOT_HASHES.items():
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"PT-exp2 model snapshot file is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"PT-exp2 model snapshot hash mismatch for {name}: "
                f"actual={actual} expected={expected}"
            )
        files[name] = actual
    return {
        "path": str(snapshot),
        "revision": PT_EXP2_MODEL_REVISION,
        "files": files,
    }


def _inspect_pt_exp2_shards(
    manifest_path: Path,
    manifest: dict[str, Any],
    registered: Path,
    expected_rows: int,
) -> dict[str, Any]:
    """Verify the immutable canonical shards and return live content evidence.

    The manifest is treated as a claim, never as the source of truth for
    bytes, hashes, or row counts.  The scan also reconstructs the ordered
    corpus digest used by ``build_bricknet_text8m.py`` so a stale or edited
    shard cannot remain covered merely by an unchanged audit report.
    """
    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("PT-exp2 canonical manifest has no valid stats object")
    declared_rows = _strict_int(stats.get("unique_rows"), "stats.unique_rows")
    if declared_rows != expected_rows:
        raise ValueError(
            f"PT-exp2 canonical row count is {declared_rows!r}, expected {expected_rows}"
        )

    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("PT-exp2 canonical manifest has no valid audit object")
    if audit.get("completed") is not True:
        raise ValueError("PT-exp2 canonical manifest audit is not completed")
    if audit.get("parse_eligible") is not True:
        raise ValueError("PT-exp2 canonical manifest audit is not parse-eligible")
    if audit.get("eligible") is not True:
        raise ValueError("PT-exp2 canonical manifest audit is not eligible")
    if type(audit.get("rows_audited")) is not int or audit.get("rows_audited") != declared_rows:
        raise ValueError("PT-exp2 canonical manifest audit row count is stale")
    if "expected_rows" in audit and (
        type(audit.get("expected_rows")) is not int or audit.get("expected_rows") != declared_rows
    ):
        raise ValueError("PT-exp2 canonical manifest audit expected row count is stale")
    if audit.get("parse_errors") != []:
        raise ValueError("PT-exp2 canonical manifest audit contains parse errors")

    declared_ordered = _strict_sha256(
        manifest.get("ordered_corpus_sha256"), "ordered_corpus_sha256"
    )
    declared_dataset = _strict_sha256(manifest.get("dataset_sha256"), "dataset_sha256")
    declared_shard_set = _strict_sha256(manifest.get("shard_set_sha256"), "shard_set_sha256")
    if declared_dataset != declared_ordered:
        raise ValueError("dataset_sha256 does not equal ordered_corpus_sha256")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("PT-exp2 canonical manifest has no shards")
    names: list[str] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"PT-exp2 canonical shard {index} is not an object")
        name = shard.get("file")
        # ``Path.name`` alone is not sufficient on POSIX because a backslash
        # is an ordinary character there; reject both path separator forms.
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or name != Path(name).name
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise ValueError(f"PT-exp2 canonical shard {index} has an unsafe basename: {name!r}")
        if name in names:
            raise ValueError(f"PT-exp2 canonical shard names are duplicated: {name!r}")
        if not name.endswith(".jsonl"):
            raise ValueError(f"PT-exp2 canonical shard is not JSONL: {name!r}")
        _strict_int(shard.get("rows"), f"shards[{index}].rows")
        _strict_int(shard.get("bytes"), f"shards[{index}].bytes")
        _strict_sha256(shard.get("sha256"), f"shards[{index}].sha256")
        names.append(name)
    if names != sorted(names):
        raise ValueError("PT-exp2 canonical shard list is not in deterministic order")

    if not registered.is_dir() or registered.is_symlink():
        raise FileNotFoundError(f"PT-exp2 registered train view is missing or symlinked: {registered}")
    view_names = sorted(path.name for path in registered.iterdir())
    if view_names != names:
        raise ValueError("PT-exp2 registered train view does not match the canonical shard set")

    canonical_root = manifest_path.parent.resolve()
    canonical_names = sorted(path.name for path in canonical_root.glob("part-*.jsonl"))
    if canonical_names != names:
        raise ValueError("PT-exp2 canonical directory does not match the declared shard set")

    ordered_digest = hashlib.sha256()
    shard_set_digest = hashlib.sha256()
    actual_shards: list[dict[str, Any]] = []
    actual_rows = 0
    actual_bytes = 0
    for index, shard in enumerate(shards):
        name = names[index]
        canonical_shard = canonical_root / name
        view_shard = registered / name
        if not canonical_shard.is_file() or canonical_shard.is_symlink():
            raise ValueError(f"PT-exp2 canonical shard is not a regular file: {canonical_shard}")
        if not view_shard.is_file() or view_shard.resolve() != canonical_shard:
            raise ValueError(f"PT-exp2 train view shard is not bound to its canonical shard: {view_shard}")

        digest = hashlib.sha256()
        rows = 0
        byte_count = 0
        with canonical_shard.open("rb", buffering=8 * 1024 * 1024) as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                byte_count += len(raw)
                rows += 1
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{canonical_shard}:{line_number}: invalid JSONL row") from exc
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"{canonical_shard}:{line_number}: expected a text string")
                text = row["text"].encode("utf-8")
                ordered_digest.update(len(text).to_bytes(8, "big"))
                ordered_digest.update(text)
        actual_sha = digest.hexdigest()
        expected_shard_rows = _strict_int(shard["rows"], f"shards[{index}].rows")
        expected_shard_bytes = _strict_int(shard["bytes"], f"shards[{index}].bytes")
        expected_shard_sha = _strict_sha256(shard["sha256"], f"shards[{index}].sha256")
        if rows != expected_shard_rows:
            raise ValueError(f"PT-exp2 shard row count drift: {canonical_shard}")
        if byte_count != expected_shard_bytes:
            raise ValueError(f"PT-exp2 shard byte count drift: {canonical_shard}")
        if actual_sha != expected_shard_sha:
            raise ValueError(f"PT-exp2 shard hash drift: {canonical_shard}")
        shard_set_digest.update(actual_sha.encode("ascii"))
        actual_rows += rows
        actual_bytes += byte_count
        actual_shards.append(
            {"file": name, "rows": rows, "bytes": byte_count, "sha256": actual_sha}
        )

    actual_ordered = ordered_digest.hexdigest()
    actual_shard_set = shard_set_digest.hexdigest()
    if actual_rows != declared_rows:
        raise ValueError(
            f"PT-exp2 canonical shard rows total {actual_rows:,} != declared {declared_rows:,}"
        )
    if actual_ordered != declared_ordered:
        raise ValueError("PT-exp2 ordered corpus hash drift")
    if actual_shard_set != declared_shard_set:
        raise ValueError("PT-exp2 shard-set hash drift")

    return {
        "rows": declared_rows,
        "actual_rows": actual_rows,
        "bytes": actual_bytes,
        "shards": len(actual_shards),
        "shard_evidence": actual_shards,
        "ordered_corpus_sha256": actual_ordered,
        "shard_set_sha256": actual_shard_set,
        "dataset_sha256": declared_dataset,
    }


def _pt_exp2_canonical_contract(dataset_contract: dict[str, Any]) -> dict[str, Any] | None:
    if PT_EXP2_DATASET not in dataset_contract["train"]:
        return None
    entry = dataset_contract["selected_entries"].get(PT_EXP2_DATASET)
    if not isinstance(entry, dict) or not isinstance(entry.get("file_name"), str):
        raise ValueError(f"{PT_EXP2_DATASET} has no valid file_name in the dataset registry")
    registered = Path(entry["file_name"])
    if not registered.is_absolute():
        registered = Path(dataset_contract["dataset_dir"]) / registered
    if registered.is_symlink():
        raise ValueError(f"PT-exp2 registered train view must be a real directory: {registered}")
    registered = registered.resolve()
    manifest_path = registered.parent / "text8m" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PT-exp2 canonical manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"PT-exp2 canonical manifest is not a mapping: {manifest_path}")
    live = _inspect_pt_exp2_shards(
        manifest_path,
        manifest,
        registered,
        PT_EXP2_EXPECTED_ROWS,
    )
    snapshot = _pt_exp2_snapshot_contract()
    return {
        "dataset": PT_EXP2_DATASET,
        "registered_train_view": str(registered),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        **live,
        "audit": {
            "completed": True,
            "parse_eligible": True,
            "eligible": True,
            "rows_audited": live["rows"],
            "parse_errors": [],
        },
        "snapshot": snapshot,
        "audit_eligible": True,
    }


def _inspect_pt_exp2_eval_file(path: Path, expected_rows: int) -> dict[str, Any]:
    """Hash and strictly parse the small PT loss validation JSONL."""
    digest = hashlib.sha256()
    rows = 0
    byte_count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            byte_count += len(raw)
            rows += 1
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"PT-exp2 eval JSONL row {line_number} is invalid") from exc
            if type(row) is not dict or set(row) != {"text"} or type(row["text"]) is not str:
                raise ValueError(
                    f"PT-exp2 eval JSONL row {line_number} must contain only a text string"
                )
    if rows != expected_rows:
        raise ValueError(f"PT-exp2 eval JSONL has {rows} rows, expected {expected_rows}")
    return {"rows": rows, "bytes": byte_count, "sha256": digest.hexdigest()}


def _pt_exp2_eval_contract(dataset_contract: dict[str, Any]) -> dict[str, Any] | None:
    """Bind the live VAL1000 file and its sidecar for the strict PT path."""
    if PT_EXP2_DATASET not in dataset_contract["train"]:
        return None
    if dataset_contract["eval"] != [PT_EXP2_EVAL_DATASET]:
        raise ValueError(
            f"PT-exp2 eval_dataset must be exactly [{PT_EXP2_EVAL_DATASET!r}], "
            f"got {dataset_contract['eval']!r}"
        )

    eval_path = _resolve_strict_registry_file(dataset_contract, PT_EXP2_EVAL_DATASET)
    if eval_path.suffix != ".jsonl":
        raise ValueError(f"PT-exp2 eval file must be JSONL: {eval_path}")
    if eval_path.is_symlink() or not eval_path.is_file():
        raise FileNotFoundError(f"PT-exp2 eval file is missing or symlinked: {eval_path}")
    live = _inspect_pt_exp2_eval_file(eval_path, PT_EXP2_EVAL_EXPECTED_ROWS)

    sidecar_path = eval_path.with_suffix(".manifest.json")
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise FileNotFoundError(f"PT-exp2 eval sidecar is missing or symlinked: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"PT-exp2 eval sidecar is invalid: {sidecar_path}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError(f"PT-exp2 eval sidecar is not a mapping: {sidecar_path}")

    if (
        type(sidecar.get("schema_version")) is not int
        or sidecar.get("schema_version") != PT_EXP2_EVAL_SCHEMA_VERSION
    ):
        raise ValueError("PT-exp2 eval sidecar schema_version is not 1")
    if sidecar.get("experiment") != PT_EXP2_EVAL_EXPERIMENT:
        raise ValueError("PT-exp2 eval sidecar experiment is not PT-exp2-text8m")
    if sidecar.get("role") != PT_EXP2_EVAL_ROLE:
        raise ValueError("PT-exp2 eval sidecar role is not official_style_pt_loss_validation")
    if type(sidecar.get("seed")) is not int or sidecar.get("seed") != PT_EXP2_EVAL_SEED:
        raise ValueError("PT-exp2 eval sidecar seed is not 0")
    if (
        type(sidecar.get("rows")) is not int
        or sidecar.get("rows") != PT_EXP2_EVAL_EXPECTED_ROWS
    ):
        raise ValueError("PT-exp2 eval sidecar rows is not 1000")
    if sidecar.get("output") != str(eval_path):
        raise ValueError("PT-exp2 eval sidecar output path does not match the registry file")
    declared_output_sha = _strict_sha256(
        sidecar.get("output_sha256"), "PT-exp2 eval sidecar output_sha256"
    )
    if declared_output_sha != live["sha256"]:
        raise ValueError("PT-exp2 eval sidecar output_sha256 does not match the live JSONL")
    if sidecar.get("not_val511_overfit") is not True:
        raise ValueError("PT-exp2 eval sidecar not_val511_overfit must be true")

    sidecar_bytes = sidecar_path.stat().st_size
    return {
        "dataset": PT_EXP2_EVAL_DATASET,
        "path": str(eval_path),
        **live,
        "manifest": str(sidecar_path),
        "manifest_bytes": sidecar_bytes,
        "manifest_sha256": _sha256(sidecar_path),
        "manifest_contract": {
            "schema_version": PT_EXP2_EVAL_SCHEMA_VERSION,
            "experiment": PT_EXP2_EVAL_EXPERIMENT,
            "role": PT_EXP2_EVAL_ROLE,
            "seed": PT_EXP2_EVAL_SEED,
            "rows": PT_EXP2_EVAL_EXPECTED_ROWS,
            "output": str(eval_path),
            "output_sha256": live["sha256"],
            "not_val511_overfit": True,
        },
    }


def _input_contract(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_contract = _dataset_registry_contract(config)
    model_name = config.get("model_name_or_path")
    model_revision = config.get("model_revision", "main")
    eval_dataset = _pt_exp2_eval_contract(dataset_contract)
    canonical = _pt_exp2_canonical_contract(dataset_contract)
    if canonical is not None:
        if model_name != PT_EXP2_MODEL:
            raise ValueError(f"PT-exp2 model must be {PT_EXP2_MODEL!r}, got {model_name!r}")
        if model_revision != PT_EXP2_MODEL_REVISION:
            raise ValueError(f"PT-exp2 model_revision must be {PT_EXP2_MODEL_REVISION!r}, got {model_revision!r}")
    model_contract = {
        "name_or_path": model_name,
        "revision": model_revision,
        "revision_explicit": "model_revision" in config,
    }
    if canonical is not None:
        model_contract["snapshot"] = canonical["snapshot"]
        canonical = {key: value for key, value in canonical.items() if key != "snapshot"}
    return {
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "model": model_contract,
        "dataset": dataset_contract,
        "preprocessing": {
            "stage": config.get("stage"),
            "template": config.get("template"),
            "cutoff_len": config.get("cutoff_len"),
            "packing": config.get("packing", False),
        },
        "canonical_dataset": canonical,
        "eval_dataset": eval_dataset,
    }


def _cache_datasets(path: Path) -> DatasetDict:
    loaded = load_from_disk(str(path))
    return DatasetDict({"train": loaded}) if isinstance(loaded, Dataset) else loaded


def _cache_schema(datasets: DatasetDict) -> dict[str, list[str]]:
    return {split: dataset.column_names for split, dataset in datasets.items()}


def _detailed_schema(datasets: DatasetDict) -> dict[str, Any]:
    return {
        split: {"columns": dataset.column_names, "features": dataset.features.to_dict()}
        for split, dataset in datasets.items()
    }


def _feature_dtype(dataset: Dataset, column: str) -> str:
    feature = dataset.features[column]
    return str(getattr(feature, "dtype", feature))


def _full_length_validation(dataset: Dataset, length_column: str, input_column: str) -> dict[str, Any]:
    declared = dataset.data.column(length_column)
    actual = pc.list_value_length(dataset.data.column(input_column))
    mismatch = pc.fill_null(pc.not_equal(declared, actual), True)
    mismatch_count = int(pc.sum(mismatch.cast("int64")).as_py() or 0)
    return {
        "method": "pyarrow.compute.list_value_length",
        "rows": len(dataset),
        "mismatch_count": mismatch_count,
        "eligible": mismatch_count == 0,
    }


def _pt_exp2_eval_split(datasets: DatasetDict) -> str:
    """Return the current LlamaFactory validation split for strict PT caches."""
    candidates = [split for split in datasets if split == "validation" or split.startswith("validation_")]
    if len(candidates) != 1:
        raise ValueError(
            "PT-exp2 tokenized cache must contain exactly one validation split "
            f"(validation or validation_*), got {candidates!r}"
        )
    split = candidates[0]
    if len(datasets[split]) != PT_EXP2_EVAL_EXPECTED_ROWS:
        raise ValueError(
            f"PT-exp2 tokenized cache eval split {split!r} has {len(datasets[split])} rows, "
            f"expected {PT_EXP2_EVAL_EXPECTED_ROWS}"
        )
    return split


def _known_pt_exp2_metadata_fingerprint_drift(
    recorded: Any,
    live: Any,
) -> bool:
    """Recognize only the reviewed pre-reload metadata fingerprint drift."""
    if not isinstance(recorded, dict) or not isinstance(live, dict):
        return False
    if set(recorded) != set(live):
        return False
    for field in ("rows", "schema", "metadata_files", "metadata_set_sha256"):
        if recorded.get(field) != live.get(field):
            return False
    if recorded.get("fingerprints") != PT_EXP2_LEGACY_CACHE_FINGERPRINTS:
        return False
    if live.get("fingerprints") != PT_EXP2_STABLE_CACHE_FINGERPRINTS:
        return False
    return recorded.get("fingerprints") != live.get("fingerprints")


def _is_known_pt_exp2_legacy_builder_hash(value: Any) -> bool:
    return value == PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256


def _is_valid_pt_exp2_manifest_repair(
    value: Any,
    current_builder_sha256: str,
) -> bool:
    """Validate the complete, immutable provenance record for the repair.

    The record is intentionally exact.  A repaired cache keeps the builder
    hash that produced its payload, so the repair record is the only place
    where the current verifier identity is recorded.  Its timestamp is
    evidence of when the repair was made, not part of the cache identity;
    nevertheless it must be a canonical, timezone-aware UTC ISO timestamp.
    """
    if not isinstance(value, dict):
        return False
    if set(value) != {
        "kind",
        "schema_version",
        "original_manifest_sha256",
        "old_fingerprints",
        "new_fingerprints",
        "repaired_fields",
        "repair_script_sha256",
        "repaired_at",
    }:
        return False
    if value.get("kind") != PT_EXP2_MANIFEST_REPAIR_KIND:
        return False
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != PT_EXP2_MANIFEST_REPAIR_SCHEMA_VERSION
    ):
        return False
    if value.get("original_manifest_sha256") != PT_EXP2_LEGACY_MANIFEST_SHA256:
        return False
    if value.get("old_fingerprints") != PT_EXP2_LEGACY_CACHE_FINGERPRINTS:
        return False
    if value.get("new_fingerprints") != PT_EXP2_STABLE_CACHE_FINGERPRINTS:
        return False
    if (
        type(value.get("repaired_fields")) is not list
        or value.get("repaired_fields") != PT_EXP2_MANIFEST_REPAIRED_FIELDS
    ):
        return False
    if value.get("repair_script_sha256") != current_builder_sha256:
        return False
    repaired_at = value.get("repaired_at")
    if not isinstance(repaired_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(repaired_at)
    except ValueError:
        return False
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        return False
    return parsed.isoformat() == repaired_at


def _reconstruct_pt_exp2_legacy_manifest_bytes(
    manifest: Any,
    live_metadata: Any,
) -> bytes | None:
    """Rebuild the reviewed pre-repair manifest from a repaired manifest.

    Only the persisted Dataset fingerprints were changed by the repair.  All
    other metadata is required to equal the live cache evidence and is kept
    logically intact when the historical JSON representation is rebuilt.
    Hashing these bytes authenticates fields outside the ordinary runtime
    contract too, including action, source provenance, and nested builder
    metadata.
    """
    if not isinstance(manifest, dict) or "repair" not in manifest:
        return None
    recorded_metadata = manifest.get("cache_metadata_evidence")
    if not isinstance(recorded_metadata, dict) or recorded_metadata != live_metadata:
        return None
    if recorded_metadata.get("fingerprints") != PT_EXP2_STABLE_CACHE_FINGERPRINTS:
        return None

    reconstructed = dict(manifest)
    reconstructed.pop("repair")
    legacy_metadata = dict(recorded_metadata)
    legacy_metadata["fingerprints"] = dict(PT_EXP2_LEGACY_CACHE_FINGERPRINTS)
    # Replacing an existing dict value retains its top-level insertion order,
    # recreating the builder's historical indent=2 + newline representation.
    reconstructed["cache_metadata_evidence"] = legacy_metadata
    return _manifest_json_bytes(reconstructed)


def _is_valid_pt_exp2_repaired_manifest(
    manifest: Any,
    live_metadata: Any,
    current_builder_sha256: str,
) -> bool:
    """Authenticate both the repair record and its complete legacy origin."""
    if not isinstance(manifest, dict):
        return False
    if not _is_valid_pt_exp2_manifest_repair(
        manifest.get("repair"), current_builder_sha256
    ):
        return False
    reconstructed = _reconstruct_pt_exp2_legacy_manifest_bytes(manifest, live_metadata)
    if reconstructed is None:
        return False
    return hashlib.sha256(reconstructed).hexdigest() == PT_EXP2_LEGACY_MANIFEST_SHA256


def check_cache(
    path: Path,
    length_column: str,
    input_column: str = "input_ids",
    *,
    config_path: Path | None = None,
    require_manifest: bool = False,
    full_length_validation: bool = False,
    _allow_known_repair_drift: bool = False,
) -> dict[str, Any]:
    """Validate cache structure and, when supplied, its persisted input contract.

    Callers that gate training should pass ``config_path`` and
    ``require_manifest=True``.  A PT-exp2 canonical contract always forces a
    full Arrow length validation, even when the caller leaves
    ``full_length_validation`` false.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "length_column": length_column,
        "input_column": input_column,
        "eligible": False,
    }
    if not path.is_dir():
        result["error"] = "CACHE_MISSING"
        return result
    if path.is_symlink() and require_manifest:
        result["error"] = "CACHE_ROOT_SYMLINK"
        return result

    try:
        datasets = _cache_datasets(path)
        schema = _cache_schema(datasets)
    except Exception as exc:
        result["error"] = f"CACHE_LOAD_FAILED: {exc}"
        return result

    result["schema"] = schema
    if "train" not in schema:
        result["error"] = "TRAIN_SPLIT_MISSING"
        return result
    elif input_column not in schema["train"]:
        result["error"] = "INPUT_COLUMN_MISSING"
        return result
    elif length_column not in schema["train"]:
        result["error"] = "LENGTH_COLUMN_MISSING"
        return result

    contract: dict[str, Any] | None = None
    strict_manifest = require_manifest
    if config_path is not None:
        try:
            config_path = config_path.resolve()
            contract = _input_contract(config_path, _load_config(config_path))
        except Exception as exc:
            result["error"] = f"CACHE_CONTRACT_INPUT_INVALID: {type(exc).__name__}: {exc}"
            return result
        strict_manifest = strict_manifest or contract["canonical_dataset"] is not None

    pt_eval_split: str | None = None
    if contract is not None and contract["canonical_dataset"] is not None:
        try:
            pt_eval_split = _pt_exp2_eval_split(datasets)
        except ValueError as exc:
            result["error"] = f"PT_EXP2_EVAL_SPLIT_INVALID: {exc}"
            return result

    result["train_rows"] = len(datasets["train"])
    result["length_dtype"] = _feature_dtype(datasets["train"], length_column)
    # ``require_manifest`` is the explicit strict gate when no YAML is
    # available; with a YAML, the PT canonical contract is the strict gate.
    must_validate_all_lengths = full_length_validation or strict_manifest
    if must_validate_all_lengths:
        try:
            if pt_eval_split is None:
                validation = _full_length_validation(datasets["train"], length_column, input_column)
            else:
                validation = {
                    split: _full_length_validation(datasets[split], length_column, input_column)
                    for split in ("train", pt_eval_split)
                }
        except Exception as exc:
            result["error"] = f"FULL_LENGTH_VALIDATION_FAILED: {type(exc).__name__}: {exc}"
            return result
        result["length_validation"] = validation
        eligible = validation["eligible"] if pt_eval_split is None else all(
            split_validation["eligible"] for split_validation in validation.values()
        )
        if not eligible:
            result["error"] = "LENGTH_VALUE_MISMATCH"
            return result

    manifest_path = path / MANIFEST_NAME
    result["manifest_path"] = str(manifest_path)
    if config_path is None:
        if require_manifest:
            if not manifest_path.is_file():
                result["error"] = "CACHE_MANIFEST_MISSING"
                return result
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                result["error"] = f"CACHE_MANIFEST_INVALID: {exc}"
                return result
            if not isinstance(manifest, dict):
                result["error"] = "CACHE_MANIFEST_INVALID"
                return result
            mismatches = []
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                mismatches.append("schema_version")
            if manifest.get("kind") != "llamafactory_tokenized_cache_with_length":
                mismatches.append("kind")
            if mismatches:
                result["contract_mismatches"] = mismatches
                result["error"] = "CACHE_MANIFEST_CONTRACT_MISMATCH"
                return result
        result["eligible"] = True
        return result

    assert contract is not None
    if contract["canonical_dataset"] is not None and path.is_symlink():
        result["error"] = "CACHE_ROOT_SYMLINK"
        return result
    cache_data_files = None
    cache_metadata_evidence = None
    if contract["canonical_dataset"] is not None:
        try:
            cache_data_files = _cache_data_files_evidence(path, reject_symlinks=True)
            cache_metadata_evidence = _cache_metadata_evidence(
                datasets, path, reject_symlinks=True
            )
        except (OSError, ValueError) as exc:
            result["error"] = f"CACHE_EVIDENCE_FAILED: {type(exc).__name__}: {exc}"
            return result
        if not cache_data_files["files"]:
            result["error"] = "CACHE_ARROW_DATA_MISSING"
            return result
    if not manifest_path.is_file():
        if strict_manifest:
            result["error"] = "CACHE_MANIFEST_MISSING"
            return result
        result["eligible"] = True
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"CACHE_MANIFEST_INVALID: {exc}"
        return result
    result["manifest_sha256"] = _sha256(manifest_path)
    if not isinstance(manifest, dict):
        result["error"] = "CACHE_MANIFEST_INVALID"
        return result
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION and not strict_manifest:
        result["legacy_manifest"] = True
        result["eligible"] = True
        return result

    mismatches: list[str] = []
    repair_drift: list[str] = []
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "llamafactory_tokenized_cache_with_length",
        "config_sha256": contract["config_sha256"],
        "model": contract["model"],
        "dataset": contract["dataset"],
        "preprocessing": contract["preprocessing"],
        "canonical_dataset": contract["canonical_dataset"],
        "input_column": input_column,
        "length_column": length_column,
        "length_dtype": result["length_dtype"],
        "train_rows": result["train_rows"],
        "output_cache": str(path.resolve()),
    }
    if contract["eval_dataset"] is not None:
        expected["eval_dataset"] = contract["eval_dataset"]
    if pt_eval_split is not None:
        expected["eval_split"] = pt_eval_split
    if cache_data_files is not None:
        expected["cache_data_files"] = cache_data_files
    if cache_metadata_evidence is not None:
        expected["cache_metadata_evidence"] = cache_metadata_evidence
    repair_record_present = "repair" in manifest
    current_builder_sha256 = _builder_script_sha256()
    repair_record_valid = (
        repair_record_present
        and _is_valid_pt_exp2_repaired_manifest(
            manifest,
            cache_metadata_evidence,
            current_builder_sha256,
        )
    )
    for field, expected_value in expected.items():
        if manifest.get(field) == expected_value:
            continue
        if (
            field == "cache_metadata_evidence"
            and _allow_known_repair_drift
            and not repair_record_present
            and _known_pt_exp2_metadata_fingerprint_drift(
                manifest.get(field), expected_value
            )
        ):
            repair_drift.append(field)
            continue
        mismatches.append(field)
    manifest_stats = manifest.get("stats", {})
    stats_train = manifest_stats.get("train") if isinstance(manifest_stats, dict) else None
    if not isinstance(stats_train, dict) or stats_train.get("rows") != result["train_rows"]:
        mismatches.append("stats.train.rows")
    if must_validate_all_lengths:
        live_stats = {split: _length_stats(dataset, length_column) for split, dataset in datasets.items()}
        if manifest.get("stats") != live_stats:
            mismatches.append("stats")
        live_validations = {
            split: _full_length_validation(dataset, length_column, input_column)
            for split, dataset in datasets.items()
        }
        if manifest.get("length_validation") != live_validations:
            mismatches.append("length_validation")
        live_samples = {
            split: _validate_lengths(dataset, length_column, input_column) for split, dataset in datasets.items()
        }
        if manifest.get("sampled_validation_rows") != live_samples:
            mismatches.append("sampled_validation_rows")
    if manifest.get("dataset_schema") != _detailed_schema(datasets):
        mismatches.append("dataset_schema")
    builder = manifest.get("builder", {})
    if not isinstance(builder, dict):
        mismatches.append("builder.script_sha256")
    else:
        builder_sha256 = builder.get("script_sha256")
        if builder_sha256 == current_builder_sha256:
            if repair_record_present:
                # A repair record is only valid when the payload-producing
                # builder remains the reviewed legacy builder.
                mismatches.append("repair")
        elif _is_known_pt_exp2_legacy_builder_hash(builder_sha256):
            if repair_record_valid and contract["canonical_dataset"] is not None:
                pass
            elif (
                _allow_known_repair_drift
                and contract["canonical_dataset"] is not None
                and not repair_record_present
            ):
                repair_drift.append("builder.script_sha256")
            else:
                mismatches.append("builder.script_sha256")
        else:
            mismatches.append("builder.script_sha256")
    if repair_record_present and (
        contract["canonical_dataset"] is None
        or not repair_record_valid
        or not isinstance(builder, dict)
        or not _is_known_pt_exp2_legacy_builder_hash(builder.get("script_sha256"))
    ):
        mismatches.append("repair")
    if mismatches:
        result["contract_mismatches"] = sorted(set(mismatches))
        result["error"] = "CACHE_MANIFEST_CONTRACT_MISMATCH"
        return result
    if repair_drift:
        result["repair_drift"] = sorted(set(repair_drift))
    result["contract_verified"] = True
    result["eligible"] = True
    return result


def _add_length(
    dataset: Dataset,
    length_column: str,
    input_column: str,
    batch_size: int,
    num_proc: int,
    cache_file: Path,
) -> Dataset:
    if input_column not in dataset.column_names:
        raise ValueError(f"input column {input_column!r} is missing from {dataset.column_names}")
    if length_column in dataset.column_names:
        return dataset

    features = Features({**dataset.features, length_column: Value("int32")})

    def compute_lengths(batch: dict[str, list[Any]]) -> dict[str, np.ndarray]:
        return {length_column: np.fromiter((len(ids) for ids in batch[input_column]), dtype=np.int32)}

    return dataset.map(
        compute_lengths,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        features=features,
        cache_file_name=str(cache_file),
        desc=f"Adding {length_column} column",
    )


def _length_stats(dataset: Dataset, length_column: str) -> dict[str, int | float | None]:
    column = dataset.data.column(length_column)
    if not len(dataset):
        return {"rows": 0, "nulls": 0, "non_nulls": 0, "sum": 0, "min": None, "max": None, "mean": None}
    bounds = pc.min_max(column).as_py()
    total = pc.sum(column).as_py()
    non_nulls = len(dataset) - column.null_count
    return {
        "rows": len(dataset),
        "nulls": column.null_count,
        "non_nulls": non_nulls,
        "sum": int(total) if total is not None else None,
        "min": int(bounds["min"]) if bounds["min"] is not None else None,
        "max": int(bounds["max"]) if bounds["max"] is not None else None,
        "mean": float(total / non_nulls) if total is not None and non_nulls else None,
    }


def _validate_lengths(dataset: Dataset, length_column: str, input_column: str) -> list[int]:
    if not len(dataset):
        return []
    indices = sorted({0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4, len(dataset) - 1})
    for index in indices:
        row = dataset[index]
        if int(row[length_column]) != len(row[input_column]):
            raise ValueError(f"length mismatch at row {index}")
    return indices


def _cache_metadata_evidence(
    datasets: DatasetDict,
    path: Path,
    *,
    reject_symlinks: bool = False,
) -> dict[str, Any]:
    metadata_files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.name in {"dataset_dict.json", "dataset_info.json", "state.json"}:
            if reject_symlinks and candidate.is_symlink():
                raise ValueError(f"cache metadata file is a symlink: {candidate}")
            if not candidate.is_file():
                continue
            metadata_files.append(
                {
                    "path": str(candidate.relative_to(path)),
                    "bytes": candidate.stat().st_size,
                    "sha256": _sha256(candidate),
                }
            )
    return {
        "rows": {split: len(dataset) for split, dataset in datasets.items()},
        "fingerprints": {split: dataset._fingerprint for split, dataset in datasets.items()},
        "schema": _detailed_schema(datasets),
        "metadata_files": metadata_files,
        "metadata_set_sha256": _json_sha256(metadata_files),
    }


def _cache_data_files_evidence(path: Path, *, reject_symlinks: bool = False) -> dict[str, Any]:
    """Hash persisted Arrow payloads using paths relative to the cache root."""
    files: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*.arrow")):
        if reject_symlinks and candidate.is_symlink():
            raise ValueError(f"cache Arrow file is a symlink: {candidate}")
        if not candidate.is_file():
            continue
        files.append(
            {
                "path": str(candidate.relative_to(path)),
                "bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    return {
        "files": files,
        "set_sha256": _json_sha256(files),
    }


def _write_json_fsync(path: Path, value: Any) -> None:
    payload = _manifest_json_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class _ManifestRefreshLock:
    """Stable cache-root advisory lock held for one complete manifest refresh."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root.resolve()
        self.path = self.cache_root / MANIFEST_REFRESH_LOCK_NAME
        self._fd: int | None = None
        self._identity: tuple[int, int] | None = None

    def __enter__(self) -> _ManifestRefreshLock:
        if self._fd is not None:
            raise RuntimeError("manifest refresh lock cannot be entered twice")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError(f"cannot open PT-exp2 cache manifest refresh lock: {self.path}: {exc}") from exc
        try:
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise RuntimeError(f"PT-exp2 cache manifest refresh lock is not a regular file: {self.path}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"PT-exp2 cache manifest refresh lock is already held: {self.path}"
                ) from exc
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._identity = (lock_stat.st_dev, lock_stat.st_ino)
        return self

    def assert_held_for(self, path: Path) -> None:
        """Reject writes outside this active lock or after lock-path replacement."""
        if self._fd is None or self._identity is None:
            raise RuntimeError("manifest replacement requires an active refresh lock")
        if path.parent.resolve() != self.cache_root:
            raise RuntimeError(f"manifest replacement path is outside the locked cache: {path}")
        try:
            descriptor_stat = os.fstat(self._fd)
            path_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"manifest refresh lock identity cannot be verified: {self.path}") from exc
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        path_identity = (path_stat.st_dev, path_stat.st_ino)
        if (
            identity != self._identity
            or path_identity != self._identity
            or not stat.S_ISREG(path_stat.st_mode)
        ):
            raise RuntimeError(f"manifest refresh lock path changed while held: {self.path}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        self._identity = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _atomic_replace_bytes_if_unchanged(
    path: Path,
    expected_bytes: bytes,
    replacement_bytes: bytes,
    *,
    lock: _ManifestRefreshLock,
) -> None:
    """Replace exact expected bytes atomically while the cache lock is held."""
    lock.assert_held_for(path)
    if path.is_symlink() or not path.is_file() or path.read_bytes() != expected_bytes:
        raise RuntimeError(f"file changed during atomic replacement: {path}")
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(replacement_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        lock.assert_held_for(path)
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected_bytes:
            raise RuntimeError(f"file changed during atomic replacement: {path}")
        temporary.replace(path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_json_if_unchanged(
    path: Path,
    expected_bytes: bytes,
    value: Any,
    *,
    lock: _ManifestRefreshLock,
) -> None:
    _atomic_replace_bytes_if_unchanged(
        path,
        expected_bytes,
        _manifest_json_bytes(value),
        lock=lock,
    )


def _parse_build_args(config: dict[str, Any], output: Path) -> tuple[Any, Any, Any, Any]:
    from transformers import HfArgumentParser

    from llamafactory.hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
        TrainingArguments,
    )

    build_config = dict(config)
    build_config.update(
        {
            "tokenized_path": str(output),
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
            "report_to": "none",
            # Cache construction is tokenizer-only.  Do not let training YAML
            # precision flags or visible GPUs change parser behavior here.
            "use_cpu": True,
            "bf16": False,
            "fp16": False,
        }
    )
    # The project train parser intentionally rejects a non-distributed process.
    # Cache construction must remain single-process at the launcher level, so
    # parse the same dataclasses without applying train-launch-only validation.
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments)
    )
    model_args, data_args, training_args, finetuning_args, _ = parser.parse_dict(build_config)
    return model_args, data_args, training_args, finetuning_args


def _build_base_cache(config: dict[str, Any], output: Path) -> None:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer

    model_args, data_args, training_args, finetuning_args = _parse_build_args(config, output)
    if finetuning_args.stage not in {"pt", "sft"}:
        raise ValueError(f"only PT and SFT are supported, got stage={finetuning_args.stage!r}")

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage=finetuning_args.stage,
        **tokenizer_module,
    )
    if not output.is_dir():
        raise RuntimeError(f"LlamaFactory did not create tokenized cache: {output}")


def build_with_length(
    config_path: Path,
    source: Path | None,
    output: Path,
    length_column: str,
    input_column: str,
    batch_size: int,
    num_proc: int,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_argument = Path(output).expanduser()
    raw_output = output_argument if output_argument.is_absolute() else ROOT / output_argument
    requested_source = source.resolve() if source is not None else None
    script_path = Path(__file__).resolve()
    initial_script_sha256 = _builder_script_sha256()
    config = _load_config(config_path)
    initial_contract = _input_contract(config_path, config)
    if initial_contract["canonical_dataset"] is not None and raw_output.is_symlink():
        raise ValueError(f"PT-exp2 cache output must not be a symlink: {raw_output}")
    output = raw_output.resolve()
    if initial_contract["canonical_dataset"] is not None and requested_source is not None:
        raise ValueError(
            "PT-exp2 cache must be built from the pinned YAML and canonical train view; "
            "an externally supplied source cache cannot establish dataset provenance"
        )

    if output.exists():
        check = check_cache(
            output,
            length_column,
            input_column,
            config_path=config_path,
            require_manifest=initial_contract["canonical_dataset"] is not None,
            full_length_validation=initial_contract["canonical_dataset"] is not None,
        )
        if check["eligible"]:
            return {"action": "reuse", **check}
        raise FileExistsError(f"output exists but is not eligible: {json.dumps(check, ensure_ascii=False)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.name}.building-{os.getpid()}"
    base = work / "base"
    maps = work / "maps"
    final = work / "final"
    if work.exists():
        raise FileExistsError(f"temporary work directory already exists: {work}")
    work.mkdir()
    maps.mkdir()

    try:
        if source is None:
            _build_base_cache(config, base)
            source = base
        elif not source.is_dir():
            raise FileNotFoundError(f"source cache does not exist: {source}")

        source_dict = _cache_datasets(source)
        source_evidence = _cache_metadata_evidence(
            source_dict,
            source,
            reject_symlinks=initial_contract["canonical_dataset"] is not None,
        )
        output_dict = DatasetDict(
            {
                split: _add_length(
                    dataset,
                    length_column,
                    input_column,
                    batch_size,
                    num_proc,
                    maps / f"{split}.arrow",
                )
                for split, dataset in source_dict.items()
            }
        )
        eval_split = _pt_exp2_eval_split(output_dict) if initial_contract["canonical_dataset"] is not None else None
        stats = {split: _length_stats(dataset, length_column) for split, dataset in output_dict.items()}
        samples = {
            split: _validate_lengths(dataset, length_column, input_column) for split, dataset in output_dict.items()
        }
        full_validations = {
            split: _full_length_validation(dataset, length_column, input_column)
            for split, dataset in output_dict.items()
        }
        failed_splits = [split for split, result in full_validations.items() if not result["eligible"]]
        if failed_splits:
            raise RuntimeError(f"length validation failed for splits: {failed_splits}")
        train_rows = stats.get("train", {}).get("rows")
        train_length_dtype = _feature_dtype(output_dict["train"], length_column) if "train" in output_dict else None
        canonical = initial_contract["canonical_dataset"]
        if canonical is not None and train_rows != canonical["rows"]:
            raise RuntimeError(f"PT-exp2 cache has {train_rows!r} train rows, expected {canonical['rows']}")
        if canonical is not None and train_length_dtype != "int32":
            raise RuntimeError(f"PT-exp2 cache length dtype is {train_length_dtype!r}, expected 'int32'")
        cutoff_len = initial_contract["preprocessing"]["cutoff_len"]
        if canonical is not None and isinstance(cutoff_len, int) and stats["train"]["max"] > cutoff_len:
            raise RuntimeError(
                f"PT-exp2 cache max length {stats['train']['max']} exceeds configured cutoff {cutoff_len}"
            )
        output_dict.save_to_disk(str(final), num_proc=num_proc)

        # ``save_to_disk`` persists Dataset metadata that can change the
        # Dataset fingerprint.  From this point onward the persisted,
        # reloaded DatasetDict is the sole source for manifest evidence and
        # validation, so a later strict check observes the same identity.
        output_dict = _cache_datasets(final)
        eval_split = _pt_exp2_eval_split(output_dict) if canonical is not None else None
        stats = {split: _length_stats(dataset, length_column) for split, dataset in output_dict.items()}
        samples = {
            split: _validate_lengths(dataset, length_column, input_column) for split, dataset in output_dict.items()
        }
        full_validations = {
            split: _full_length_validation(dataset, length_column, input_column)
            for split, dataset in output_dict.items()
        }
        failed_splits = [split for split, result in full_validations.items() if not result["eligible"]]
        if failed_splits:
            raise RuntimeError(f"persisted cache length validation failed for splits: {failed_splits}")
        train_rows = stats.get("train", {}).get("rows")
        train_length_dtype = _feature_dtype(output_dict["train"], length_column) if "train" in output_dict else None
        if canonical is not None and train_rows != canonical["rows"]:
            raise RuntimeError(f"persisted PT-exp2 cache has {train_rows!r} train rows, expected {canonical['rows']}")
        if canonical is not None and train_length_dtype != "int32":
            raise RuntimeError(f"persisted PT-exp2 cache length dtype is {train_length_dtype!r}, expected 'int32'")
        if canonical is not None and isinstance(cutoff_len, int) and stats["train"]["max"] > cutoff_len:
            raise RuntimeError(
                f"persisted PT-exp2 cache max length {stats['train']['max']} exceeds configured cutoff {cutoff_len}"
            )

        check = check_cache(final, length_column, input_column, full_length_validation=True)
        if not check["eligible"]:
            raise RuntimeError(f"new cache failed validation: {check}")
        final_contract = _input_contract(config_path, _load_config(config_path))
        if final_contract != initial_contract:
            raise RuntimeError("cache input contract changed during construction; refusing atomic publication")
        if _builder_script_sha256() != initial_script_sha256:
            raise RuntimeError("cache builder script changed during construction; refusing atomic publication")
        cache_data_files = (
            _cache_data_files_evidence(final, reject_symlinks=True)
            if canonical is not None
            else None
        )
        cache_metadata_evidence = (
            _cache_metadata_evidence(output_dict, final, reject_symlinks=True)
            if canonical is not None
            else None
        )
        if canonical is not None and not cache_data_files["files"]:
            raise RuntimeError("PT-exp2 cache contains no Arrow data files")
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": "llamafactory_tokenized_cache_with_length",
            "action": "build",
            "created_at": datetime.now(UTC).isoformat(),
            "config": initial_contract["config"],
            "config_sha256": initial_contract["config_sha256"],
            "model": initial_contract["model"],
            "dataset": initial_contract["dataset"],
            "preprocessing": initial_contract["preprocessing"],
            "canonical_dataset": canonical,
            "source_cache": str(requested_source) if requested_source is not None else None,
            "source_cache_evidence": source_evidence,
            "output_cache": str(output),
            "input_column": input_column,
            "length_column": length_column,
            "length_dtype": train_length_dtype,
            "train_rows": train_rows,
            "dataset_schema": _detailed_schema(output_dict),
            "cache_data_files": cache_data_files,
            "cache_metadata_evidence": cache_metadata_evidence,
            "stats": stats,
            "length_validation": full_validations,
            "sampled_validation_rows": samples,
            "builder": {
                "script": str(script_path),
                "script_sha256": initial_script_sha256,
                "mode": "migrate_source_cache" if requested_source is not None else "build_from_config",
                "batch_size": batch_size,
                "num_proc": num_proc,
                "atomic_publish": "sibling_staging_directory_then_rename",
                "git": _git_state(),
                "software": _package_versions(),
            },
        }
        if initial_contract["eval_dataset"] is not None:
            manifest["eval_dataset"] = initial_contract["eval_dataset"]
            manifest["eval_split"] = eval_split
        _write_json_fsync(final / MANIFEST_NAME, manifest)
        # Validate the logical destination before publishing. The staged cache
        # remains invisible at ``output`` until every contract check has passed.
        staged_manifest = json.loads((final / MANIFEST_NAME).read_text(encoding="utf-8"))
        if staged_manifest != manifest:
            raise RuntimeError("staged cache manifest failed round-trip validation")
        if canonical is not None and _cache_data_files_evidence(
            final,
            reject_symlinks=True,
        ) != cache_data_files:
            raise RuntimeError("cache Arrow data changed during manifest construction")
        if canonical is not None and _cache_metadata_evidence(
            output_dict,
            final,
            reject_symlinks=True,
        ) != cache_metadata_evidence:
            raise RuntimeError("cache metadata changed during manifest construction")
        if output.exists():
            raise FileExistsError(f"output appeared during cache construction: {output}")
        final.rename(output)
        return {"action": "build", "eligible": True, **manifest}
    finally:
        if work.exists():
            shutil.rmtree(work)


def refresh_manifest(
    config_path: Path,
    output: Path,
    length_column: str,
    input_column: str,
) -> dict[str, Any]:
    """Repair the reviewed PT-exp2 persisted-metadata drift in place.

    This operation never tokenizes or rewrites Arrow data.  It first performs
    the complete strict cache check (including canonical input provenance,
    Arrow hashes, and full persisted length validation), while allowing only
    the known pre-reload fingerprints and legacy builder hash.  The manifest
    is then replaced with a lock-scoped atomic write after an exact-byte
    recheck and checked strictly again.  The legacy builder hash is deliberately
    retained: it is the identity of the code that produced the Arrow payload.
    The current script identity is recorded separately in a complete repair
    provenance record.  Any other manifest or payload drift aborts without a
    write.
    """
    config_path = config_path.resolve()
    output_argument = Path(output).expanduser()
    raw_output = output_argument if output_argument.is_absolute() else ROOT / output_argument
    config = _load_config(config_path)
    contract = _input_contract(config_path, config)
    if contract["canonical_dataset"] is None:
        raise ValueError("--refresh-manifest requires the pinned PT-exp2 canonical YAML contract")
    if raw_output.is_symlink():
        raise ValueError(f"PT-exp2 cache output must not be a symlink: {raw_output}")
    output = raw_output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"PT-exp2 cache does not exist: {output}")

    with _ManifestRefreshLock(output) as lock:
        return _refresh_manifest_locked(
            config_path,
            output,
            length_column,
            input_column,
            lock=lock,
        )


def _refresh_manifest_locked(
    config_path: Path,
    output: Path,
    length_column: str,
    input_column: str,
    *,
    lock: _ManifestRefreshLock,
) -> dict[str, Any]:
    """Execute a complete refresh while ``lock`` remains exclusively held."""
    manifest_path = output / MANIFEST_NAME
    lock.assert_held_for(manifest_path)
    if manifest_path.is_symlink():
        raise ValueError(f"PT-exp2 cache manifest must not be a symlink: {manifest_path}")
    try:
        original_bytes = manifest_path.read_bytes()
        original_manifest = json.loads(original_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"PT-exp2 cache manifest is invalid: {manifest_path}: {exc}") from exc
    if not isinstance(original_manifest, dict):
        raise ValueError(f"PT-exp2 cache manifest is not a mapping: {manifest_path}")

    preflight = check_cache(
        output,
        length_column,
        input_column,
        config_path=config_path,
        require_manifest=True,
        full_length_validation=True,
        _allow_known_repair_drift=True,
    )
    if not preflight["eligible"]:
        raise RuntimeError(
            "PT-exp2 cache failed strict refresh preflight: "
            + json.dumps(preflight, ensure_ascii=False)
        )
    repair_drift = preflight.get("repair_drift", [])
    if not repair_drift:
        if manifest_path.is_symlink() or manifest_path.read_bytes() != original_bytes:
            raise RuntimeError("PT-exp2 cache manifest changed during refresh preflight")
        return {"action": "refresh_manifest", "changed": False, **preflight}

    expected_repair_fields = sorted(PT_EXP2_MANIFEST_REPAIRED_FIELDS)
    if sorted(set(repair_drift)) != expected_repair_fields:
        raise RuntimeError(
            "PT-exp2 cache refresh found an incomplete known drift: "
            + json.dumps(repair_drift, ensure_ascii=False)
        )
    original_manifest_sha256 = _sha256(manifest_path)
    if original_manifest_sha256 != PT_EXP2_LEGACY_MANIFEST_SHA256:
        raise RuntimeError(
            "PT-exp2 cache manifest is not the reviewed legacy manifest: "
            f"actual={original_manifest_sha256} expected={PT_EXP2_LEGACY_MANIFEST_SHA256}"
        )
    if "repair" in original_manifest:
        raise RuntimeError("PT-exp2 cache has a repair record despite pending refresh drift")

    # Re-read all persisted evidence after the strict preflight.  In
    # particular, this prevents a concurrent Arrow/metadata change from
    # being hidden behind the old manifest's claims.
    datasets = _cache_datasets(output)
    live_data_files = _cache_data_files_evidence(output, reject_symlinks=True)
    live_metadata = _cache_metadata_evidence(datasets, output, reject_symlinks=True)
    if not live_data_files["files"]:
        raise RuntimeError("PT-exp2 cache contains no Arrow data files")
    if original_manifest.get("cache_data_files") != live_data_files:
        raise RuntimeError("PT-exp2 cache Arrow data changed during refresh preflight")
    metadata_drift_is_known = _known_pt_exp2_metadata_fingerprint_drift(
        original_manifest.get("cache_metadata_evidence"), live_metadata
    )
    if "cache_metadata_evidence" in repair_drift:
        if not metadata_drift_is_known:
            raise RuntimeError("PT-exp2 cache metadata changed outside the known fingerprint drift")
    elif original_manifest.get("cache_metadata_evidence") != live_metadata:
        raise RuntimeError("PT-exp2 cache metadata changed during refresh preflight")
    current_builder_sha256 = _builder_script_sha256()
    builder = original_manifest.get("builder")
    if not isinstance(builder, dict) or not _is_known_pt_exp2_legacy_builder_hash(
        builder.get("script_sha256")
    ):
        raise RuntimeError("PT-exp2 cache builder hash changed outside the known legacy drift")
    if manifest_path.read_bytes() != original_bytes:
        raise RuntimeError("PT-exp2 cache manifest changed during refresh preflight")

    repaired_manifest = json.loads(original_bytes.decode("utf-8"))
    repaired_manifest["cache_metadata_evidence"] = live_metadata
    repaired_manifest["repair"] = {
        "kind": PT_EXP2_MANIFEST_REPAIR_KIND,
        "schema_version": PT_EXP2_MANIFEST_REPAIR_SCHEMA_VERSION,
        "original_manifest_sha256": PT_EXP2_LEGACY_MANIFEST_SHA256,
        "old_fingerprints": dict(PT_EXP2_LEGACY_CACHE_FINGERPRINTS),
        "new_fingerprints": dict(PT_EXP2_STABLE_CACHE_FINGERPRINTS),
        "repaired_fields": list(PT_EXP2_MANIFEST_REPAIRED_FIELDS),
        "repair_script_sha256": current_builder_sha256,
        "repaired_at": datetime.now(UTC).isoformat(),
    }
    if not _is_valid_pt_exp2_repaired_manifest(
        repaired_manifest,
        live_metadata,
        current_builder_sha256,
    ):
        raise RuntimeError("PT-exp2 cache repair provenance construction failed")
    _atomic_replace_json_if_unchanged(
        manifest_path,
        original_bytes,
        repaired_manifest,
        lock=lock,
    )
    published_bytes = manifest_path.read_bytes()

    postflight_error: Exception | None = None
    try:
        postflight = check_cache(
            output,
            length_column,
            input_column,
            config_path=config_path,
            require_manifest=True,
            full_length_validation=True,
        )
    except Exception as exc:
        postflight = None
        postflight_error = exc
    if postflight_error is not None or not postflight["eligible"]:
        # The same exclusive lock covers publication, postflight, and this
        # rollback.  Exact-byte verification additionally catches changes by
        # writers that do not honor the advisory lock.
        try:
            _atomic_replace_bytes_if_unchanged(
                manifest_path,
                published_bytes,
                original_bytes,
                lock=lock,
            )
        except RuntimeError as rollback_error:
            raise RuntimeError(
                "PT-exp2 cache failed strict postflight and manifest rollback failed: "
                + str(rollback_error)
            ) from rollback_error
        if postflight_error is not None:
            raise RuntimeError(
                "PT-exp2 cache strict refresh postflight raised and manifest was rolled back: "
                + str(postflight_error)
            ) from postflight_error
        assert postflight is not None
        raise RuntimeError(
            "PT-exp2 cache failed strict refresh postflight: "
            + json.dumps(postflight, ensure_ascii=False)
        )
    return {
        "action": "refresh_manifest",
        "eligible": True,
        "changed": True,
        "repaired_fields": sorted(repair_drift),
        "preflight": preflight,
        "postflight": postflight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="PT or SFT training YAML")
    parser.add_argument("--source-cache", type=Path, help="existing cache to migrate; omit to build from YAML")
    parser.add_argument(
        "--output-cache",
        type=Path,
        help="override YAML tokenized_path; use a new path when migrating the configured cache",
    )
    parser.add_argument("--length-column", help="override YAML length_column_name")
    parser.add_argument("--input-column", default="input_ids")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="repair only the reviewed PT-exp2 persisted-metadata manifest drift",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="fail check-only when the cache has no schema-v2 provenance manifest",
    )
    parser.add_argument(
        "--full-length-validation",
        action="store_true",
        help="in check-only mode compare every persisted length against input_ids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.num_proc < 1:
        raise SystemExit("--num-proc must be positive")
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_value = args.output_cache or config.get("tokenized_path")
    if not output_value:
        raise SystemExit("tokenized_path is missing; set it in YAML or pass --output-cache")
    length_column = args.length_column or config.get("length_column_name")
    if not length_column:
        raise SystemExit("length_column_name is missing; set it in YAML or pass --length-column")

    output = _resolve(output_value)
    if args.check_only and args.refresh_manifest:
        raise SystemExit("--check-only and --refresh-manifest are mutually exclusive")
    if args.refresh_manifest and args.source_cache:
        raise SystemExit("--source-cache cannot be combined with --refresh-manifest")
    if not args.check_only and args.source_cache and _resolve(args.source_cache).resolve() == output.resolve():
        raise SystemExit("--source-cache and output cache must differ; pass --output-cache with a new path")
    if args.refresh_manifest:
        result = refresh_manifest(
            config_path=config_path,
            output=output,
            length_column=str(length_column),
            input_column=args.input_column,
        )
    elif args.check_only:
        result = check_cache(
            output,
            str(length_column),
            args.input_column,
            config_path=config_path,
            require_manifest=args.require_manifest,
            full_length_validation=args.full_length_validation,
        )
    else:
        source = _resolve(args.source_cache) if args.source_cache else None
        result = build_with_length(
            config_path=config_path,
            source=source,
            output=output,
            length_column=str(length_column),
            input_column=args.input_column,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("eligible"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
