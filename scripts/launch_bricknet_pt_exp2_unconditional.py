#!/usr/bin/env python3
"""Gate-protected PT-exp2 text8m unconditional generation and evaluation.

Every action is a read-only plan unless ``--execute`` is supplied.  The
launcher never guesses a checkpoint: the historical final adapter and all
four recorded hashes are checked before a model process can be started.
Generation writes to a sibling ``.building`` directory.  Evaluation is staged
inside that directory and ``verify --execute`` is the only operation that can
promote it to the immutable final output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = ROOT.parent / "BrickNet"
LLAMAFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BASE_SNAPSHOT = Path(
    "/home/jiahao/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
) / "snapshots" / BASE_REVISION

TRAIN_CONFIG = (
    ROOT
    / "examples/train_lora/qwen35_08b_bricknet_pt_exp2_text8m.yaml"
)
DEFAULT_ADAPTER_PATH = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora/"
    "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_"
    "lora64_len6401_nopack"
)

GENERATE_SCRIPT = BRICKNET_ROOT / "scripts/generate.py"
EVALUATE_SCRIPT = BRICKNET_ROOT / "scripts/evaluate_unconditional_generation.py"
SCORE_SCRIPT = BRICKNET_ROOT / "src/bricknet/score.py"
# Keep the collision-mesh source on the repository's pinned data mount.  Do
# not silently fall back to platformdirs or another user's cache: that would
# make the structure scores depend on ambient machine state.
BRICKNET_DATA = BRICKNET_ROOT / "data/bricknet_datasets"
CATALOG_ROOT = BRICKNET_ROOT / "src/bricknet/_data/v1"

OUTPUT_ROOT = (
    BRICKNET_ROOT
    / "outputs_pt/qwen35_08b/pt_exp2_text8m_250k_unconditional_v1"
)
BUILDING_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".building")
SMOKE_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".smoke")

EXPECTED_SAMPLES = 2_048
SMOKE_SAMPLES = 1
EXPECTED_BATCH_SIZE = 8
EXPECTED_MAX_NEW_TOKENS = 4_096
EXPECTED_TEMPERATURE = 1.0
EXPECTED_TOP_K = 20
EXPECTED_TOP_P = 0.95
EXPECTED_DTYPE = "bfloat16"
EXPECTED_PROMPT = "a"
EXPECTED_STOP_AFTER_NEWLINES = 199
EXPECTED_PROMPT_TOKEN_IDS = (64,)
EXPECTED_NEWLINE_TOKEN = "Ċ"
EXPECTED_NEWLINE_TOKEN_ID = 198
EXPECTED_EOS_TOKEN = "<|im_end|>"
EXPECTED_EOS_TOKEN_ID = 248_046
EXPECTED_TRAIN_STEPS = 250_000
EXPECTED_COLLISION_MESHES = 21_084
EXPECTED_CATALOG = "v1"

EXPECTED_ADAPTER_HASHES = {
    "adapter_model.safetensors": (
        "a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd"
    ),
    "adapter_config.json": (
        "f3d911f759aa05a05a7eaf0192f3a1b537dd816f524b5b99e7bfbeb22c39446a"
    ),
    "trainer_state.json": (
        "dd55f2ea0c42e3d9f562f78b5a5fd74d3d8a017ec3ab1e7c50d9956adfead997"
    ),
    "train_results.json": (
        "b07662baf32da06765e43d46c518403560dfa9775eff2b8ec05c096ca2c6e7b4"
    ),
}
# Named aliases keep the checkpoint identity easy to inspect from a shell or
# a test while the mapping above remains the single source used by the gates.
ADAPTER_MODEL_SHA256 = EXPECTED_ADAPTER_HASHES["adapter_model.safetensors"]
ADAPTER_CONFIG_SHA256 = EXPECTED_ADAPTER_HASHES["adapter_config.json"]
TRAINER_STATE_SHA256 = EXPECTED_ADAPTER_HASHES["trainer_state.json"]
TRAIN_RESULTS_SHA256 = EXPECTED_ADAPTER_HASHES["train_results.json"]
EXPECTED_SNAPSHOT_HASHES = {
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "model.safetensors.index.json": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
    "model.safetensors-00001-of-00001.safetensors": (
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
    ),
}

# The training YAML and the reviewed adapter identity both use exactly these
# seven targets.  Keep this semantic check alongside the config hash so a
# malformed or incompatible adapter cannot pass on hash metadata alone.
EXPECTED_TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
CATALOG_FILES = (
    "part_names.json",
    "color_names.json",
    "labels.json.xz",
    "part_aliases.json.xz",
)
EVALUATION_OUTPUTS = (
    "scored.jsonl",
    "metrics.json",
    "metrics.md",
    "evaluation_manifest.json",
)
FINAL_REQUIRED_OUTPUTS = ("run_spec.json", "out.jsonl", *EVALUATION_OUTPUTS)
# The launcher manifest is written only after this exact set has been
# validated.  In particular, evaluator staging directories, shard files, and
# hidden temporary files must never be promoted with the scored bundle.
EXPECTED_BUILDING_ENTRIES = frozenset(FINAL_REQUIRED_OUTPUTS)
RUN_SPEC_NAME = "run_spec.json"
FINAL_MANIFEST_NAME = "manifest.json"
EVALUATION_SUBDIR = "evaluation"
SCHEMA_VERSION = "bricknet-pt-exp2-unconditional-launch-v1"
GPU_SELECTOR_PATTERN = re.compile(r"^(0|[1-9][0-9]*)$")


class GateError(RuntimeError):
    """Raised when a destructive or expensive step fails a gate."""


class PreflightReport:
    """Machine-readable preflight result."""

    def __init__(
        self,
        action: str,
        adapter_path: Path,
        checks: dict[str, Any],
        blockers: list[str],
    ) -> None:
        self.action = action
        self.adapter_path = adapter_path
        self.checks = checks
        self.blockers = blockers

    @property
    def eligible(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "action": self.action,
            "eligible": self.eligible,
            "adapter_path": str(self.adapter_path),
            "output_root": str(OUTPUT_ROOT),
            "building_root": str(_building_root()),
            "checks": self.checks,
            "blockers": self.blockers,
        }


def _building_root() -> Path:
    """Return the formal staging path from the current output constant."""
    return OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".building")


def _smoke_root() -> Path:
    """Return the isolated smoke artifact path from the current output constant."""
    return OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".smoke")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GateError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise GateError(f"blank JSONL row: {path}:{line_number}")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise GateError(f"invalid JSONL row: {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise GateError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except OSError as exc:
        raise GateError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_valid_gpu_selector(value: str | None) -> bool:
    """Accept one canonical decimal physical-GPU index only."""
    return value is not None and GPU_SELECTOR_PATTERN.fullmatch(value) is not None


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _check_adapter(adapter_path: Path) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {"path": str(adapter_path), "eligible": False}
    blockers: list[str] = []
    if not adapter_path.is_dir() or adapter_path.is_symlink():
        checks["directory"] = {"ok": False, "exists": adapter_path.exists()}
        return checks, ["ADAPTER_DIRECTORY_MISSING_OR_SYMLINK"]

    file_checks: dict[str, Any] = {}
    for name, expected in EXPECTED_ADAPTER_HASHES.items():
        path = adapter_path / name
        actual = _sha256(path)
        match = actual == expected
        file_checks[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": actual,
            "expected_sha256": expected,
            "match": match,
        }
        if not match:
            blockers.append(f"ADAPTER_HASH_MISMATCH:{name}")
    checks["directory"] = {"ok": True, "exists": True}
    checks["files"] = file_checks

    config = _load_json(adapter_path / "adapter_config.json")
    target_modules = config.get("target_modules") if config else None
    target_modules_ok = (
        isinstance(target_modules, list)
        and all(isinstance(module, str) for module in target_modules)
        and len(target_modules) == len(EXPECTED_TARGET_MODULES)
        and frozenset(target_modules) == EXPECTED_TARGET_MODULES
    )
    lora_alpha_ok = (
        config is not None
        and _is_int(config.get("lora_alpha"))
        and config.get("lora_alpha") == 128
    )
    config_ok = (
        config is not None
        and config.get("base_model_name_or_path") == BASE_MODEL
        and config.get("r") == 64
        and lora_alpha_ok
        and config.get("peft_type") == "LORA"
        and target_modules_ok
    )
    checks["adapter_config"] = {
        "ok": config_ok,
        "base_model_name_or_path": config.get("base_model_name_or_path") if config else None,
        "r": config.get("r") if config else None,
        "lora_alpha": config.get("lora_alpha") if config else None,
        "peft_type": config.get("peft_type") if config else None,
        "target_modules": config.get("target_modules") if config else None,
        "expected_base_model_name_or_path": BASE_MODEL,
        "expected_r": 64,
        "expected_lora_alpha": 128,
        "expected_target_modules": sorted(EXPECTED_TARGET_MODULES),
    }
    if not config_ok:
        blockers.append("ADAPTER_CONFIG_CONTRACT_FAILED")

    trainer_state = _load_json(adapter_path / "trainer_state.json")
    steps_ok = (
        trainer_state is not None
        and trainer_state.get("global_step") == EXPECTED_TRAIN_STEPS
        and trainer_state.get("max_steps") == EXPECTED_TRAIN_STEPS
    )
    checks["trainer_state"] = {
        "ok": steps_ok,
        "global_step": trainer_state.get("global_step") if trainer_state else None,
        "max_steps": trainer_state.get("max_steps") if trainer_state else None,
        "expected_steps": EXPECTED_TRAIN_STEPS,
    }
    if not steps_ok:
        blockers.append("TRAINER_STATE_STEPS_CONTRACT_FAILED")

    results = _load_json(adapter_path / "train_results.json")
    checks["train_results"] = {"ok": results is not None, "valid_json_object": results is not None}
    if results is None:
        blockers.append("TRAIN_RESULTS_INVALID_OR_MISSING")

    checks["eligible"] = not blockers
    return checks, blockers


def _check_snapshot() -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {
        "model": BASE_MODEL,
        "revision": BASE_REVISION,
        "path": str(BASE_SNAPSHOT),
        "revision_path_match": BASE_SNAPSHOT.name == BASE_REVISION,
    }
    blockers: list[str] = []
    if not BASE_SNAPSHOT.is_dir() or BASE_SNAPSHOT.is_symlink():
        blockers.append("BASE_SNAPSHOT_MISSING_OR_SYMLINK")
    if not checks["revision_path_match"]:
        blockers.append("BASE_SNAPSHOT_REVISION_MISMATCH")
    files: dict[str, Any] = {}
    for name, expected in EXPECTED_SNAPSHOT_HASHES.items():
        path = BASE_SNAPSHOT / name
        actual = _sha256(path)
        match = actual == expected
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": actual,
            "expected_sha256": expected,
            "match": match,
        }
        if not match:
            blockers.append(f"BASE_SNAPSHOT_HASH_MISMATCH:{name}")
    for name in REQUIRED_SNAPSHOT_FILES:
        if not (BASE_SNAPSHOT / name).is_file():
            blockers.append(f"BASE_SNAPSHOT_FILE_MISSING:{name}")
    model_files = sorted(path.name for path in BASE_SNAPSHOT.glob("*.safetensors") if path.is_file())
    if not model_files:
        blockers.append("BASE_SNAPSHOT_WEIGHTS_MISSING")
    checks["files"] = files
    checks["model_weight_files"] = model_files
    checks["eligible"] = not blockers
    return checks, blockers


def _tokenizer_protocol() -> dict[str, Any]:
    return {
        "tokenizer_json": str(BASE_SNAPSHOT / "tokenizer.json"),
        "tokenizer_config": str(BASE_SNAPSHOT / "tokenizer_config.json"),
        "prompt": EXPECTED_PROMPT,
        "prompt_token_ids": list(EXPECTED_PROMPT_TOKEN_IDS),
        "newline_token": EXPECTED_NEWLINE_TOKEN,
        "newline_token_id": EXPECTED_NEWLINE_TOKEN_ID,
        "eos_token": EXPECTED_EOS_TOKEN,
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
    }


def _check_tokenizer_semantics() -> tuple[dict[str, Any], list[str]]:
    """Check stop-token semantics from JSON without instantiating a tokenizer."""
    tokenizer_path = BASE_SNAPSHOT / "tokenizer.json"
    config_path = BASE_SNAPSHOT / "tokenizer_config.json"
    tokenizer = _load_json(tokenizer_path)
    tokenizer_config = _load_json(config_path)
    blockers: list[str] = []

    model = tokenizer.get("model") if tokenizer else None
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    prompt_id = vocab.get(EXPECTED_PROMPT) if isinstance(vocab, Mapping) else None
    newline_id = vocab.get(EXPECTED_NEWLINE_TOKEN) if isinstance(vocab, Mapping) else None
    prompt_ids = [prompt_id] if _is_int(prompt_id) else None
    prompt_ok = prompt_ids == list(EXPECTED_PROMPT_TOKEN_IDS)
    newline_ok = newline_id == EXPECTED_NEWLINE_TOKEN_ID and _is_int(newline_id)

    added_tokens = tokenizer.get("added_tokens") if tokenizer else None
    eos_added_ids = (
        [
            item.get("id")
            for item in added_tokens
            if isinstance(item, Mapping) and item.get("content") == EXPECTED_EOS_TOKEN
        ]
        if isinstance(added_tokens, list)
        else []
    )
    eos_added_ok = eos_added_ids == [EXPECTED_EOS_TOKEN_ID]

    eos_token = tokenizer_config.get("eos_token") if tokenizer_config else None
    decoder = tokenizer_config.get("added_tokens_decoder") if tokenizer_config else None
    decoder_entry = (
        decoder.get(str(EXPECTED_EOS_TOKEN_ID))
        if isinstance(decoder, Mapping)
        else None
    )
    decoder_content = (
        decoder_entry.get("content") if isinstance(decoder_entry, Mapping) else None
    )
    eos_config_ok = (
        eos_token == EXPECTED_EOS_TOKEN and decoder_content == EXPECTED_EOS_TOKEN
    )

    if tokenizer is None:
        blockers.append("TOKENIZER_FILE_MISSING_OR_INVALID:tokenizer.json")
    if tokenizer_config is None:
        blockers.append("TOKENIZER_FILE_MISSING_OR_INVALID:tokenizer_config.json")
    if not prompt_ok:
        blockers.append("TOKENIZER_PROMPT_TOKEN_IDS_MISMATCH")
    if not newline_ok:
        blockers.append("TOKENIZER_NEWLINE_TOKEN_ID_MISMATCH")
    if not eos_added_ok:
        blockers.append("TOKENIZER_EOS_ADDED_TOKEN_MISMATCH")
    if not eos_config_ok:
        blockers.append("TOKENIZER_EOS_CONFIG_MISMATCH")

    checks = {
        "expected": _tokenizer_protocol(),
        "tokenizer_json": {
            "path": str(tokenizer_path),
            "exists": tokenizer_path.is_file(),
            "prompt_token_ids": prompt_ids,
            "newline_token": EXPECTED_NEWLINE_TOKEN,
            "newline_token_id": newline_id,
            "eos_added_ids": eos_added_ids,
            "eos_added_match": eos_added_ok,
        },
        "tokenizer_config": {
            "path": str(config_path),
            "exists": config_path.is_file(),
            "eos_token": eos_token,
            "eos_decoder_content": decoder_content,
            "eos_config_match": eos_config_ok,
        },
        "eligible": not blockers,
    }
    return checks, blockers


def _check_runtime() -> tuple[dict[str, Any], list[str]]:
    paths = {
        "llamafactory_python": LLAMAFACTORY_PYTHON,
        "bricknet_python": BRICKNET_PYTHON,
        "generate_script": GENERATE_SCRIPT,
        "evaluate_script": EVALUATE_SCRIPT,
        "score_script": SCORE_SCRIPT,
        "train_config": TRAIN_CONFIG,
    }
    checks = {name: {"path": str(path), "exists": path.is_file()} for name, path in paths.items()}
    blockers = [f"RUNTIME_MISSING:{name}" for name, item in checks.items() if not item["exists"]]
    checks["eligible"] = not blockers
    return checks, blockers


def _check_collision_assets() -> tuple[dict[str, Any], list[str]]:
    inset = BRICKNET_DATA / "inset"
    mesh_count = (
        sum(1 for path in inset.glob("*.ply") if path.is_file())
        if inset.is_dir()
        else 0
    )
    catalog_files = {name: (CATALOG_ROOT / name).is_file() for name in CATALOG_FILES}
    checks = {
        "bricknet_data": str(BRICKNET_DATA),
        "catalog": EXPECTED_CATALOG,
        "inset": {"path": str(inset), "exists": inset.is_dir(), "files": mesh_count},
        "catalog_files": catalog_files,
    }
    blockers: list[str] = []
    if mesh_count != EXPECTED_COLLISION_MESHES:
        blockers.append(f"COLLISION_MESH_COUNT_MISMATCH:{mesh_count}")
    blockers.extend(f"CATALOG_FILE_MISSING:{name}" for name, ok in catalog_files.items() if not ok)
    checks["eligible"] = not blockers
    return checks, blockers


def _check_output_start() -> tuple[dict[str, Any], list[str]]:
    building = _building_root()
    checks = {
        "parent": {
            "path": str(OUTPUT_ROOT.parent),
            "ok": not OUTPUT_ROOT.parent.exists() or OUTPUT_ROOT.parent.is_dir(),
        },
        "final_absent": not OUTPUT_ROOT.exists() and not OUTPUT_ROOT.is_symlink(),
        "building_absent": not building.exists() and not building.is_symlink(),
    }
    blockers: list[str] = []
    if not checks["parent"]["ok"]:
        blockers.append("OUTPUT_PARENT_NOT_DIRECTORY")
    if not checks["final_absent"]:
        blockers.append("FINAL_OUTPUT_ALREADY_EXISTS")
    if not checks["building_absent"]:
        blockers.append("FORMAL_BUILDING_DIRECTORY_ALREADY_EXISTS")
    checks["eligible"] = not blockers
    return checks, blockers


def _validate_raw_rows(path: Path, expected_samples: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len(rows) != expected_samples:
        raise GateError(f"expected {expected_samples} generated rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if "path" in row:
            raise GateError(f"generated row {index} contains forbidden path field")
        if not _is_int(row.get("id")) or row["id"] != index:
            raise GateError(f"generated row {index} has invalid ordered integer id")
        if not _is_int(row.get("sample")) or row["sample"] != 0:
            raise GateError(f"generated row {index} has invalid integer sample=0")
        if row.get("source") != "" or row.get("caption") != "":
            raise GateError(f"generated row {index} requires empty source and caption")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or not text.startswith(EXPECTED_PROMPT):
            raise GateError(f"generated row {index} has invalid text/prompt")
    return rows


def _validate_run_spec_raw_artifact(
    spec: Mapping[str, Any], raw_path: Path, raw_rows: Sequence[Mapping[str, Any]]
) -> None:
    if spec.get("raw_rows") != len(raw_rows):
        raise GateError("run_spec raw_rows differs from the current out.jsonl")
    if spec.get("raw_sha256") != _sha256(raw_path):
        raise GateError("run_spec raw_sha256 differs from the current out.jsonl")


def _validate_run_spec_evaluation_artifacts(
    spec: Mapping[str, Any], root: Path, scored_rows: Sequence[Mapping[str, Any]]
) -> None:
    expected = {
        "scored_sha256": _sha256(root / "scored.jsonl"),
        "metrics_sha256": _sha256(root / "metrics.json"),
        "evaluation_manifest_sha256": _sha256(root / "evaluation_manifest.json"),
        "scored_rows": len(scored_rows),
    }
    for field, actual in expected.items():
        if spec.get(field) != actual:
            raise GateError(f"run_spec {field} differs from the current evaluation artifacts")


def _validate_scored_rows(raw: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]) -> None:
    if len(scored) != len(raw):
        raise GateError(f"expected {len(raw)} scored rows, found {len(scored)}")
    for index, (source, row) in enumerate(zip(raw, scored)):
        required = ("id", "sample", "text", "n_actions", "invalid", "collisions")
        if any(field not in row for field in required):
            raise GateError(f"scored row {index} is missing a required field")
        if row["id"] != source["id"] or row["sample"] != source["sample"] or row["text"] != source["text"]:
            raise GateError(f"scored row {index} does not match raw row")
        if not _is_int(row["n_actions"]) or row["n_actions"] < 0:
            raise GateError(f"scored row {index} has invalid n_actions")
        if row["invalid"] is not None and (not _is_int(row["invalid"]) or row["invalid"] < 0):
            raise GateError(f"scored row {index} has invalid invalid index")
        if not isinstance(row["collisions"], list) or any(
            not _is_int(value) or value < 0 for value in row["collisions"]
        ):
            raise GateError(f"scored row {index} has invalid collisions")


def _aggregate_scored_metrics(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the evaluator's three primary metrics from scored rows."""
    if not scored:
        raise GateError("cannot aggregate an empty scored bundle")
    parsable_count = sum(row["invalid"] is None for row in scored)
    clean_count = sum(row["invalid"] is None and not row["collisions"] for row in scored)
    first_failures = []
    for row in scored:
        invalid = row["n_actions"] if row["invalid"] is None else row["invalid"]
        first_failures.append(min([invalid, *row["collisions"]]))
    sample_count = len(scored)
    return {
        "samples": sample_count,
        "parsable_count": parsable_count,
        "parsable_rate": parsable_count / sample_count,
        "clean_count": clean_count,
        "clean_rate": clean_count / sample_count,
        "mean_actions_before_first_failure": sum(first_failures) / sample_count,
    }


def _validate_run_spec_contract(spec: Mapping[str, Any], adapter_path: Path) -> None:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise GateError("run_spec schema mismatch")
    if spec.get("experiment_id") != "PT-exp2-text8m-250k-unconditional-v1":
        raise GateError("run_spec experiment identity mismatch")
    if spec.get("scope") != "formal":
        raise GateError("formal verify requires a formal run_spec")
    if spec.get("protocol") != _protocol(EXPECTED_SAMPLES, EXPECTED_BATCH_SIZE):
        raise GateError("run_spec protocol differs from the frozen main protocol")
    if spec.get("tokenizer") != _tokenizer_protocol():
        raise GateError("run_spec tokenizer semantics differ from the frozen protocol")
    if spec.get("code") != _code_identity():
        raise GateError("run_spec code identity differs from the current code")
    adapter = spec.get("adapter")
    if not isinstance(adapter, Mapping) or adapter.get("path") != str(adapter_path):
        raise GateError("adapter path differs from the generated run spec")


def _check_building_entries(building: Path) -> None:
    """Require the exact regular-file set that can be atomically promoted."""
    if not building.is_dir() or building.is_symlink():
        raise GateError(f"formal building directory is missing or symlinked: {building}")
    entries = {path.name for path in building.iterdir()}
    if entries != EXPECTED_BUILDING_ENTRIES:
        missing = sorted(EXPECTED_BUILDING_ENTRIES - entries)
        unexpected = sorted(entries - EXPECTED_BUILDING_ENTRIES)
        raise GateError(
            "formal building entry set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    non_files = [
        name
        for name in sorted(EXPECTED_BUILDING_ENTRIES)
        if not (building / name).is_file() or (building / name).is_symlink()
    ]
    if non_files:
        raise GateError(f"formal building entries must be regular files: {non_files}")


def _check_action_state(
    action: str, *, gpu: str | None, adapter_path: Path | None = None
) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    building = _building_root()
    if action in {"smoke", "generate"}:
        checks["gpu"] = {"value": gpu, "ok": _is_valid_gpu_selector(gpu)}
        if not checks["gpu"]["ok"]:
            blockers.append("GPU_SELECTOR_REQUIRED_OR_INVALID")
        start_checks, start_blockers = _check_output_start()
        checks["output_start"] = start_checks
        blockers.extend(start_blockers)
        smoke = _smoke_root()
        checks["smoke_absent"] = not smoke.exists() and not smoke.is_symlink()
        if action == "smoke" and not checks["smoke_absent"]:
            blockers.append("SMOKE_OUTPUT_ALREADY_EXISTS")
    elif action == "evaluate":
        checks["building_exists"] = building.is_dir() and not building.is_symlink()
        if not checks["building_exists"]:
            blockers.append("FORMAL_BUILDING_DIRECTORY_MISSING")
        else:
            spec = _load_json(building / RUN_SPEC_NAME)
            checks["run_spec_status"] = spec.get("status") if spec else None
            if not spec or spec.get("status") != "generated":
                blockers.append("GENERATION_NOT_COMPLETE_OR_RUN_SPEC_INVALID")
            elif adapter_path is not None:
                try:
                    _validate_run_spec_contract(spec, adapter_path)
                except GateError as exc:
                    checks["run_spec_error"] = str(exc)
                    blockers.append("RUN_SPEC_PROTOCOL_OR_ADAPTER_MISMATCH")
            raw = building / "out.jsonl"
            try:
                raw_rows = _validate_raw_rows(raw, EXPECTED_SAMPLES)
                checks["raw_rows"] = len(raw_rows)
            except GateError as exc:
                checks["raw_rows_error"] = str(exc)
                blockers.append("RAW_GENERATION_INVALID")
            if spec and spec.get("status") == "generated" and "raw_rows" in checks:
                try:
                    _validate_run_spec_raw_artifact(spec, raw, raw_rows)
                except GateError as exc:
                    checks["run_spec_raw_error"] = str(exc)
                    blockers.append("RUN_SPEC_RAW_IDENTITY_MISMATCH")
            eval_dir = building / EVALUATION_SUBDIR
            checks["evaluation_destination_absent"] = not eval_dir.exists() and not eval_dir.is_symlink()
            if not checks["evaluation_destination_absent"]:
                blockers.append("EVALUATION_STAGING_ALREADY_EXISTS")
            for name in EVALUATION_OUTPUTS:
                if (building / name).exists():
                    blockers.append(f"EVALUATION_OUTPUT_ALREADY_EXISTS:{name}")
    elif action == "verify":
        checks["final_absent"] = not OUTPUT_ROOT.exists() and not OUTPUT_ROOT.is_symlink()
        if not checks["final_absent"]:
            blockers.append("FINAL_OUTPUT_ALREADY_EXISTS")
        checks["building_exists"] = building.is_dir() and not building.is_symlink()
        if not checks["building_exists"]:
            blockers.append("FORMAL_BUILDING_DIRECTORY_MISSING")
        else:
            spec = _load_json(building / RUN_SPEC_NAME)
            checks["run_spec_status"] = spec.get("status") if spec else None
            if not spec or spec.get("status") != "evaluated":
                blockers.append("EVALUATION_NOT_COMPLETE_OR_RUN_SPEC_INVALID")
            elif adapter_path is not None:
                try:
                    _validate_run_spec_contract(spec, adapter_path)
                except GateError as exc:
                    checks["run_spec_error"] = str(exc)
                    blockers.append("RUN_SPEC_PROTOCOL_OR_ADAPTER_MISMATCH")
            try:
                raw = _validate_raw_rows(building / "out.jsonl", EXPECTED_SAMPLES)
                scored = _read_jsonl(building / "scored.jsonl")
                _validate_scored_rows(raw, scored)
                checks["raw_rows"] = len(raw)
                checks["scored_rows"] = len(scored)
            except GateError as exc:
                checks["bundle_error"] = str(exc)
                blockers.append("RAW_OR_SCORED_BUNDLE_INVALID")
            if spec and spec.get("status") == "evaluated" and "raw_rows" in checks:
                try:
                    _validate_run_spec_raw_artifact(spec, building / "out.jsonl", raw)
                    _validate_run_spec_evaluation_artifacts(spec, building, scored)
                except GateError as exc:
                    checks["run_spec_artifact_error"] = str(exc)
                    blockers.append("RUN_SPEC_ARTIFACT_IDENTITY_MISMATCH")
            for name in ("metrics.json", "metrics.md", "evaluation_manifest.json"):
                if not (building / name).is_file():
                    blockers.append(f"EVALUATION_OUTPUT_MISSING:{name}")
            if (building / FINAL_MANIFEST_NAME).exists():
                blockers.append("FINAL_MANIFEST_ALREADY_EXISTS")
            eval_dir = building / EVALUATION_SUBDIR
            if eval_dir.exists():
                blockers.append("EVALUATION_STAGING_REMAINS")
            try:
                _check_building_entries(building)
            except GateError as exc:
                checks["building_entries_error"] = str(exc)
                blockers.append("FORMAL_BUILDING_ENTRY_SET_INVALID")
    checks["eligible"] = not blockers
    return checks, blockers


def preflight(action: str, adapter_path: Path | str, gpu: str | None = None) -> PreflightReport:
    """Run read-only checks for one launcher action."""
    adapter_path = Path(adapter_path).expanduser().resolve()
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    for function, key in (
        (_check_adapter, "adapter"),
        (_check_snapshot, "base_snapshot"),
        (_check_tokenizer_semantics, "tokenizer_semantics"),
        (_check_runtime, "runtime"),
        (_check_collision_assets, "collision_assets"),
    ):
        if key == "adapter":
            result, current = function(adapter_path)  # type: ignore[arg-type]
        else:
            result, current = function()  # type: ignore[misc]
        checks[key] = result
        blockers.extend(current)
    if action in {"preflight", "smoke", "generate"}:
        result, current = _check_output_start()
        checks["output_start"] = result
        blockers.extend(current if action in {"smoke", "generate"} else [])
    action_checks, action_blockers = _check_action_state(
        action, gpu=gpu, adapter_path=adapter_path
    )
    checks["action"] = action_checks
    blockers.extend(action_blockers)
    return PreflightReport(action, adapter_path, checks, _dedupe(blockers))


def _protocol(samples: int, batch_size: int) -> dict[str, Any]:
    return {
        "base_model": BASE_MODEL,
        "base_snapshot": str(BASE_SNAPSHOT),
        "base_revision": BASE_REVISION,
        "prompt": EXPECTED_PROMPT,
        "num_samples": samples,
        "batch_size": batch_size,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "temperature": EXPECTED_TEMPERATURE,
        "top_k": EXPECTED_TOP_K,
        "top_p": EXPECTED_TOP_P,
        "dtype": EXPECTED_DTYPE,
        "stop_after_newlines": EXPECTED_STOP_AFTER_NEWLINES,
        "seed": None,
        "prompts_file": None,
        "tokenizer": _tokenizer_protocol(),
    }


def _code_identity() -> dict[str, Any]:
    paths = {
        "launcher": Path(__file__).resolve(),
        "generate_script": GENERATE_SCRIPT,
        "evaluate_script": EVALUATE_SCRIPT,
        "score_script": SCORE_SCRIPT,
        "train_config": TRAIN_CONFIG,
    }
    return {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}


def _environment(gpu: str | None) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "launcher_python": str(LLAMAFACTORY_PYTHON),
        "evaluator_python": str(BRICKNET_PYTHON),
        "gpu": gpu,
        "parent_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bricknet_data": str(BRICKNET_DATA),
        "bricknet_catalog": EXPECTED_CATALOG,
    }


def _manifest_provenance(spec: Mapping[str, Any], gpu: str | None) -> dict[str, Any]:
    """Separate generation provenance from the identity of this verification."""
    generation_code = spec.get("code")
    generation_environment = spec.get("environment")
    if not isinstance(generation_code, Mapping) or not isinstance(
        generation_environment, Mapping
    ):
        raise GateError("run_spec generation provenance is missing or invalid")
    return {
        "generation": {
            "code": dict(generation_code),
            "environment": dict(generation_environment),
        },
        "verification": {
            "code": _code_identity(),
            "environment": _environment(gpu),
        },
    }


def _run_spec(
    adapter_path: Path,
    *,
    status: str,
    scope: str,
    samples: int,
    batch_size: int,
    gpu: str | None,
) -> dict[str, Any]:
    adapter_files = {
        name: {"path": str(adapter_path / name), "sha256": _sha256(adapter_path / name)}
        for name in EXPECTED_ADAPTER_HASHES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "PT-exp2-text8m-250k-unconditional-v1",
        "status": status,
        "scope": scope,
        "created_at": datetime.now(UTC).isoformat(),
        "adapter": {"path": str(adapter_path), "files": adapter_files},
        "protocol": _protocol(samples, batch_size),
        "tokenizer": _tokenizer_protocol(),
        "evaluation": {
            "evaluator": str(EVALUATE_SCRIPT),
            "expected_samples": samples,
            "expected_prompt": EXPECTED_PROMPT,
            "collision": True,
            "newline_target_is_diagnostic_only": True,
        },
        "outputs": {
            "final_root": str(OUTPUT_ROOT if scope == "formal" else _smoke_root()),
            "staging_root": str(_building_root() if scope == "formal" else _smoke_root()),
        },
        "code": _code_identity(),
        "environment": _environment(gpu),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise GateError(f"temporary file already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"cannot update missing run spec: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise GateError(f"temporary file already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _update_run_spec(path: Path, **updates: Any) -> dict[str, Any]:
    spec = _load_json(path)
    if spec is None:
        raise GateError(f"invalid run spec: {path}")
    spec.update(updates)
    _replace_json(path, spec)
    return spec


def _generation_command(
    adapter_path: Path | str,
    output_path: Path | str,
    *,
    samples: int,
    batch_size: int,
) -> list[str]:
    """Build the exact BrickNet generator command.

    Deliberately no ``--seed`` and no ``--prompts_file`` are emitted for the
    main run; the official reference protocol is an unseeded raw ``a`` prompt.
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    return [
        str(LLAMAFACTORY_PYTHON),
        str(GENERATE_SCRIPT),
        "--model",
        str(BASE_SNAPSHOT),
        "--lora",
        str(adapter_path),
        "--output",
        str(output_path),
        "--prompt",
        EXPECTED_PROMPT,
        "--num_samples",
        str(samples),
        "--batch_size",
        str(batch_size),
        "--max_new_tokens",
        str(EXPECTED_MAX_NEW_TOKENS),
        "--temperature",
        str(EXPECTED_TEMPERATURE),
        "--top_k",
        str(EXPECTED_TOP_K),
        "--top_p",
        str(EXPECTED_TOP_P),
        "--dtype",
        EXPECTED_DTYPE,
        "--stop_after_newlines",
        str(EXPECTED_STOP_AFTER_NEWLINES),
    ]


def _evaluation_command(
    input_path: Path | str, output_dir: Path | str, *, expected_samples: int
) -> list[str]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    return [
        str(BRICKNET_PYTHON),
        str(EVALUATE_SCRIPT),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--expected-samples",
        str(expected_samples),
        "--expected-prompt",
        EXPECTED_PROMPT,
        "--workers",
        "1",
        "--execute",
    ]


def _command_env(gpu: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["BRICKNET_DATA"] = str(BRICKNET_DATA)
    env["BRICKNET_CATALOG"] = EXPECTED_CATALOG
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _run_command(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], label: str) -> None:
    print(f"[{label}]\n$ {shlex.join([str(item) for item in command])}", flush=True)
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=dict(env),
        check=False,
    )
    if result.returncode:
        raise GateError(f"{label} failed with exit code {result.returncode}")


def _create_staging(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise GateError(f"refusing to reuse existing staging directory: {root}")
    if root.parent.exists() and not root.parent.is_dir():
        raise GateError(f"staging parent is not a directory: {root.parent}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()


def _run_generation(
    adapter_path: Path,
    *,
    scope: str,
    samples: int,
    batch_size: int,
    gpu: str,
) -> Path:
    root = _building_root() if scope == "formal" else _smoke_root()
    _create_staging(root)
    spec_path = root / RUN_SPEC_NAME
    _write_json_new(
        spec_path,
        _run_spec(
            adapter_path,
            status="planned",
            scope=scope,
            samples=samples,
            batch_size=batch_size,
            gpu=gpu,
        ),
    )
    _update_run_spec(spec_path, status="running", started_at=datetime.now(UTC).isoformat())
    raw_path = root / "out.jsonl"
    try:
        _run_command(
            _generation_command(adapter_path, raw_path, samples=samples, batch_size=batch_size),
            cwd=BRICKNET_ROOT,
            env=_command_env(gpu),
            label=f"{scope} generation",
        )
        rows = _validate_raw_rows(raw_path, samples)
        _update_run_spec(
            spec_path,
            status="generated",
            generated_at=datetime.now(UTC).isoformat(),
            raw_sha256=_sha256(raw_path),
            raw_rows=len(rows),
        )
    except Exception as exc:
        try:
            _update_run_spec(spec_path, status="generation_failed", error=str(exc))
        except GateError:
            pass
        raise
    return raw_path


def _validate_evaluation_bundle(
    root: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    stable_root: Path | None = None,
) -> dict[str, Any]:
    scored_path = root / "scored.jsonl"
    metrics_path = root / "metrics.json"
    report_path = root / "metrics.md"
    manifest_path = root / "evaluation_manifest.json"
    scored = _read_jsonl(scored_path)
    _validate_scored_rows(raw_rows, scored)
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    if metrics is None or manifest is None:
        raise GateError("evaluation metrics or manifest is invalid")
    if not report_path.is_file():
        raise GateError(f"evaluation report is missing: {report_path}")
    if metrics.get("schema_version") != "bricknet-unconditional-evaluation-v1":
        raise GateError("unexpected unconditional evaluator schema")
    recomputed = _aggregate_scored_metrics(scored)
    for key, expected in recomputed.items():
        if metrics.get(key) != expected:
            raise GateError(f"evaluation metric mismatch for {key}")
    if metrics.get("collision") != recomputed["mean_actions_before_first_failure"]:
        raise GateError("evaluation metric mismatch for collision")
    if metrics.get("input_sha256") != _sha256(root / "out.jsonl"):
        raise GateError("evaluation metrics input hash mismatch")
    if metrics.get("scored_sha256") != _sha256(scored_path):
        raise GateError("evaluation metrics scored hash mismatch")
    if manifest.get("status") != "complete":
        raise GateError("evaluation manifest is not complete")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("expected_samples") != len(raw_rows):
        raise GateError("evaluation manifest sample contract mismatch")
    if evaluation.get("expected_prompt") != EXPECTED_PROMPT:
        raise GateError("evaluation manifest prompt contract mismatch")
    if manifest.get("metrics") != metrics:
        raise GateError("evaluation manifest metrics content mismatch")
    input_info = manifest.get("input")
    scored_info = manifest.get("scored")
    if not isinstance(input_info, Mapping) or not isinstance(scored_info, Mapping):
        raise GateError("evaluation manifest lacks input/scored hashes")
    raw_hash = _sha256(root / "out.jsonl")
    scored_hash = _sha256(scored_path)
    if stable_root is None:
        stable_root = root
    if input_info.get("path") != str(stable_root / "out.jsonl"):
        raise GateError("evaluation manifest input path mismatch")
    if scored_info.get("path") != str(stable_root / "scored.jsonl"):
        raise GateError("evaluation manifest scored path mismatch")
    if input_info.get("sha256") != raw_hash or input_info.get("rows") != len(raw_rows):
        raise GateError("evaluation manifest input hash/row count mismatch")
    if scored_info.get("sha256") != scored_hash or scored_info.get("rows") != len(scored):
        raise GateError("evaluation manifest scored hash/row count mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GateError("evaluation manifest lacks launcher artifact hashes")
    expected_artifacts = {
        "input": (stable_root / "out.jsonl", raw_hash),
        "scored": (stable_root / "scored.jsonl", scored_hash),
        "metrics": (stable_root / "metrics.json", _sha256(metrics_path)),
        "report": (stable_root / "metrics.md", _sha256(report_path)),
    }
    for name, (expected_path, expected_hash) in expected_artifacts.items():
        item = artifacts.get(name)
        if not isinstance(item, Mapping):
            raise GateError(f"evaluation manifest lacks artifact record: {name}")
        if item.get("path") != str(expected_path) or item.get("sha256") != expected_hash:
            raise GateError(f"evaluation manifest artifact hash/path mismatch: {name}")
    if manifest.get("metrics_path") != str(stable_root / "metrics.json"):
        raise GateError("evaluation manifest metrics path mismatch")
    if manifest.get("metrics_sha256") != _sha256(metrics_path):
        raise GateError("evaluation manifest metrics hash mismatch")
    if manifest.get("report_path") != str(stable_root / "metrics.md"):
        raise GateError("evaluation manifest report path mismatch")
    if manifest.get("report_sha256") != _sha256(report_path):
        raise GateError("evaluation manifest report hash mismatch")
    return {"metrics": metrics, "manifest": manifest, "scored_rows": len(scored)}


def _rewrite_evaluation_manifest_paths(evaluation_dir: Path, stable_root: Path) -> None:
    """Make evaluator paths survive the outer staging-directory promotion."""
    path = evaluation_dir / "evaluation_manifest.json"
    manifest = _load_json(path)
    if manifest is None:
        raise GateError(f"invalid evaluator manifest: {path}")
    input_info = manifest.get("input")
    scored_info = manifest.get("scored")
    if not isinstance(input_info, dict) or not isinstance(scored_info, dict):
        raise GateError("evaluator manifest lacks input/scored path records")
    input_info["path"] = str(stable_root / "out.jsonl")
    manifest["output_dir"] = str(stable_root)
    scored_info["path"] = str(stable_root / "scored.jsonl")
    raw_path = evaluation_dir.parent / "out.jsonl"
    manifest["metrics_path"] = str(stable_root / "metrics.json")
    manifest["metrics_sha256"] = _sha256(evaluation_dir / "metrics.json")
    manifest["report_path"] = str(stable_root / "metrics.md")
    manifest["report_sha256"] = _sha256(evaluation_dir / "metrics.md")
    manifest["artifacts"] = {
        "input": {"path": str(stable_root / "out.jsonl"), "sha256": _sha256(raw_path)},
        "scored": {
            "path": str(stable_root / "scored.jsonl"),
            "sha256": _sha256(evaluation_dir / "scored.jsonl"),
        },
        "metrics": {
            "path": str(stable_root / "metrics.json"),
            "sha256": _sha256(evaluation_dir / "metrics.json"),
        },
        "report": {
            "path": str(stable_root / "metrics.md"),
            "sha256": _sha256(evaluation_dir / "metrics.md"),
        },
    }
    _replace_json(path, manifest)


def _move_evaluation_bundle(building: Path, *, stable_root: Path | None = None) -> None:
    if stable_root is None:
        stable_root = OUTPUT_ROOT if building == _building_root() else building
    evaluation_dir = building / EVALUATION_SUBDIR
    if not evaluation_dir.is_dir() or evaluation_dir.is_symlink():
        raise GateError(f"evaluator did not create output directory: {evaluation_dir}")
    entries = {path.name for path in evaluation_dir.iterdir()}
    expected = set(EVALUATION_OUTPUTS)
    if entries != expected:
        raise GateError(f"evaluator output set mismatch: expected {sorted(expected)}, found {sorted(entries)}")
    non_files = [
        name
        for name in EVALUATION_OUTPUTS
        if not (evaluation_dir / name).is_file() or (evaluation_dir / name).is_symlink()
    ]
    if non_files:
        raise GateError(f"evaluator outputs must be regular files: {non_files}")
    _rewrite_evaluation_manifest_paths(evaluation_dir, stable_root)
    for name in EVALUATION_OUTPUTS:
        source = evaluation_dir / name
        destination = building / name
        if destination.exists() or destination.is_symlink():
            raise GateError(f"refusing to overwrite evaluation output: {destination}")
        source.replace(destination)
    evaluation_dir.rmdir()


def _run_evaluation(adapter_path: Path, *, gpu: str | None) -> None:
    building = _building_root()
    raw_path = building / "out.jsonl"
    spec = _load_json(building / RUN_SPEC_NAME)
    if spec is None or spec.get("status") != "generated":
        raise GateError("run_spec is not in generated state")
    _validate_run_spec_contract(spec, adapter_path)
    if spec["adapter"].get("files") != _adapter_identity(adapter_path)["files"]:
        raise GateError("adapter hashes differ from the generated run spec")
    raw_rows = _validate_raw_rows(raw_path, EXPECTED_SAMPLES)
    evaluation_dir = building / EVALUATION_SUBDIR
    try:
        _validate_run_spec_raw_artifact(spec, raw_path, raw_rows)
        _run_command(
            _evaluation_command(raw_path, evaluation_dir, expected_samples=EXPECTED_SAMPLES),
            cwd=BRICKNET_ROOT,
            env=_command_env(gpu),
            label="unconditional evaluation",
        )
        _move_evaluation_bundle(building, stable_root=OUTPUT_ROOT)
        bundle = _validate_evaluation_bundle(building, raw_rows, stable_root=OUTPUT_ROOT)
        outputs = {
            "final_root": str(OUTPUT_ROOT),
            "staging_root": str(building),
            "raw": str(OUTPUT_ROOT / "out.jsonl"),
            "scored": str(OUTPUT_ROOT / "scored.jsonl"),
            "metrics": str(OUTPUT_ROOT / "metrics.json"),
            "report": str(OUTPUT_ROOT / "metrics.md"),
            "evaluation_manifest": str(OUTPUT_ROOT / "evaluation_manifest.json"),
        }
        _update_run_spec(
            building / RUN_SPEC_NAME,
            status="evaluated",
            evaluated_at=datetime.now(UTC).isoformat(),
            scored_sha256=_sha256(building / "scored.jsonl"),
            metrics_sha256=_sha256(building / "metrics.json"),
            evaluation_manifest_sha256=_sha256(building / "evaluation_manifest.json"),
            scored_rows=bundle["scored_rows"],
            outputs=outputs,
        )
    except Exception as exc:
        spec_path = building / RUN_SPEC_NAME
        if spec_path.is_file():
            try:
                _update_run_spec(spec_path, status="evaluation_failed", error=str(exc))
            except GateError:
                pass
        raise


def _adapter_identity(adapter_path: Path) -> dict[str, Any]:
    return {
        "path": str(adapter_path),
        "files": {
            name: {"path": str(adapter_path / name), "sha256": _sha256(adapter_path / name)}
            for name in EXPECTED_ADAPTER_HASHES
        },
    }


def _verify_and_promote(adapter_path: Path, *, gpu: str | None) -> Path:
    building = _building_root()
    _check_building_entries(building)
    raw_rows = _validate_raw_rows(building / "out.jsonl", EXPECTED_SAMPLES)
    spec = _load_json(building / RUN_SPEC_NAME)
    if spec is None or spec.get("status") != "evaluated":
        raise GateError("run_spec is not in evaluated state")
    _validate_run_spec_contract(spec, adapter_path)
    _validate_run_spec_raw_artifact(spec, building / "out.jsonl", raw_rows)
    bundle = _validate_evaluation_bundle(building, raw_rows, stable_root=OUTPUT_ROOT)
    scored_rows = _read_jsonl(building / "scored.jsonl")
    _validate_run_spec_evaluation_artifacts(spec, building, scored_rows)
    recorded_adapter = spec["adapter"]
    current_identity = _adapter_identity(adapter_path)
    if recorded_adapter.get("files") != current_identity["files"]:
        raise GateError("adapter hashes differ from the generated run spec")

    artifacts = {}
    for name in ("run_spec.json", "out.jsonl", *EVALUATION_OUTPUTS):
        path = building / name
        if not path.is_file():
            raise GateError(f"missing formal artifact: {path}")
        artifacts[name] = {
            "path": str(OUTPUT_ROOT / name),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "PT-exp2-text8m-250k-unconditional-v1",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "adapter": current_identity,
        "protocol": _protocol(EXPECTED_SAMPLES, EXPECTED_BATCH_SIZE),
        "tokenizer": dict(spec["tokenizer"]),
        "evaluation": {
            "metrics": bundle["metrics"],
            "manifest_sha256": _sha256(building / "evaluation_manifest.json"),
        },
        "artifacts": artifacts,
        "provenance": _manifest_provenance(spec, gpu),
    }
    manifest_path = building / FINAL_MANIFEST_NAME
    _write_json_new(manifest_path, manifest)
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise GateError(f"refusing to overwrite final output: {OUTPUT_ROOT}")
    # Both directories are siblings, so rename is atomic at the filesystem
    # level.  The existence check above makes promotion fail closed for the
    # normal concurrent-creator case.
    building.replace(OUTPUT_ROOT)
    return OUTPUT_ROOT


def _planned_commands(action: str, adapter_path: Path, gpu: str | None) -> list[str]:
    if action == "smoke":
        raw = _smoke_root() / "out.jsonl"
        evaluation = _smoke_root() / EVALUATION_SUBDIR
        commands = [
            _generation_command(adapter_path, raw, samples=SMOKE_SAMPLES, batch_size=1),
            _evaluation_command(raw, evaluation, expected_samples=SMOKE_SAMPLES),
        ]
    elif action == "generate":
        commands = [
            _generation_command(
                adapter_path,
                _building_root() / "out.jsonl",
                samples=EXPECTED_SAMPLES,
                batch_size=EXPECTED_BATCH_SIZE,
            )
        ]
    elif action == "evaluate":
        commands = [
            _evaluation_command(
                _building_root() / "out.jsonl",
                _building_root() / EVALUATION_SUBDIR,
                expected_samples=EXPECTED_SAMPLES,
            )
        ]
    else:
        commands = []
    return [shlex.join([str(item) for item in command]) for command in commands]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--action",
        choices=("preflight", "smoke", "generate", "evaluate", "verify"),
        required=True,
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help=f"Historical PT-exp2 text8m final adapter (default: {DEFAULT_ADAPTER_PATH}).",
    )
    parser.add_argument(
        "--gpu",
        "--gpus",
        dest="gpu",
        help="One physical CUDA selector used by smoke/generate (the --gpus spelling is kept for record.md).",
    )
    parser.add_argument("--execute", action="store_true", help="Perform the selected action; default is dry-run.")
    parser.add_argument(
        "--adapter-approved",
        action="store_true",
        help="Explicitly approve using the manually reviewed historical adapter for smoke/generate.",
    )
    args = parser.parse_args(argv)
    # Keep the plural spelling visible to callers that consume record.md while
    # using the singular internal name required by this one-GPU contract.
    args.gpus = args.gpu
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    adapter_path = args.adapter_path.expanduser().resolve()
    report = preflight(args.action, adapter_path, args.gpu)
    payload = report.as_dict()
    payload["execute"] = bool(args.execute)
    payload["adapter_approved"] = bool(args.adapter_approved)
    payload["planned_commands"] = _planned_commands(args.action, adapter_path, args.gpu)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    if not args.execute:
        return 0
    if args.action in {"smoke", "generate"} and not args.adapter_approved:
        print("ERROR: --adapter-approved is required for executable smoke/generate actions.", file=sys.stderr)
        return 2
    if not report.eligible:
        print("ERROR: preflight blocked execution: " + "; ".join(report.blockers), file=sys.stderr)
        return 2

    try:
        if args.action == "smoke":
            raw_path = _run_generation(
                adapter_path,
                scope="smoke",
                samples=SMOKE_SAMPLES,
                batch_size=1,
                gpu=args.gpu,
            )
            smoke_root = _smoke_root()
            try:
                raw_rows = _validate_raw_rows(raw_path, SMOKE_SAMPLES)
                evaluation_dir = smoke_root / EVALUATION_SUBDIR
                _run_command(
                    _evaluation_command(raw_path, evaluation_dir, expected_samples=SMOKE_SAMPLES),
                    cwd=BRICKNET_ROOT,
                    env=_command_env(args.gpu),
                    label="smoke evaluation",
                )
                _move_evaluation_bundle(smoke_root, stable_root=smoke_root)
                bundle = _validate_evaluation_bundle(smoke_root, raw_rows, stable_root=smoke_root)
                _update_run_spec(
                    smoke_root / RUN_SPEC_NAME,
                    status="evaluated",
                    evaluated_at=datetime.now(UTC).isoformat(),
                    scored_sha256=_sha256(smoke_root / "scored.jsonl"),
                    metrics_sha256=_sha256(smoke_root / "metrics.json"),
                    evaluation_manifest_sha256=_sha256(smoke_root / "evaluation_manifest.json"),
                    scored_rows=bundle["scored_rows"],
                    outputs={
                        "final_root": str(smoke_root),
                        "staging_root": str(smoke_root),
                        "raw": str(smoke_root / "out.jsonl"),
                        "scored": str(smoke_root / "scored.jsonl"),
                        "metrics": str(smoke_root / "metrics.json"),
                        "report": str(smoke_root / "metrics.md"),
                        "evaluation_manifest": str(smoke_root / "evaluation_manifest.json"),
                    },
                )
            except Exception as exc:
                try:
                    _update_run_spec(smoke_root / RUN_SPEC_NAME, status="evaluation_failed", error=str(exc))
                except GateError:
                    pass
                raise
        elif args.action == "generate":
            _run_generation(
                adapter_path,
                scope="formal",
                samples=EXPECTED_SAMPLES,
                batch_size=EXPECTED_BATCH_SIZE,
                gpu=args.gpu,
            )
        elif args.action == "evaluate":
            _run_evaluation(adapter_path, gpu=args.gpu)
        elif args.action == "verify":
            destination = _verify_and_promote(adapter_path, gpu=args.gpu)
            print(json.dumps({"status": "promoted", "output_root": str(destination)}))
        elif args.action == "preflight":
            pass
    except (GateError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
