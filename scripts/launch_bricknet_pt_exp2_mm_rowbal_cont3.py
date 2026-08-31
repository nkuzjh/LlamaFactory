#!/usr/bin/env python3
"""Fail-closed launcher for PT-exp2-mm-rowbal-cont3.

The experiment is a single, continuous three-epoch SFT run over one immutable
row-balanced dataset (135,051 multimodal rows followed by 135,051 explicit
``images=[]`` text rows).  Epoch checkpoints are consumed directly for three
separate VAL512 predictions; this launcher deliberately has no alias or winner
selection action.

The data builder and this launcher share the following contract.  The builder
must atomically publish ``data/bricknet_pt_exp2_mm_rowbal_cont3/`` containing:

* ``PT-exp2-mm-rowbal-cont3.jsonl`` with exactly 270,102 rows and the exact
  top-level schema/order ``id, images, messages``;
* ``dataset_info.json`` registering exactly
  ``BrickNet-PT-exp2-mm-rowbal-cont3`` as a ShareGPT dataset;
* ``manifest.v1.json`` declaring the file, registry, ordered-id/content hashes,
  MM/text counts, schema, and eligible Arrow materialization; and
* ``loader_validation_report.json`` and ``processor_audit.json``.  Both are
  real full-pool audits and bind the manifest/data hashes.  They must report
  zero loader/processor errors and zero truncation.

The default is read-only preflight.  ``prepare-audits``, ``prepare-cache``,
``train``, ``predict`` and ``evaluate`` mutate or execute only with explicit
``--execute``.  Cache
preparation delegates only tokenizer/data preprocessing to the existing
CPU-only cache builder with ``CUDA_VISIBLE_DEVICES=``; it never loads model
weights or starts a Trainer.  Training never builds a cache in-process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/data/jiahao/task/BrickNet")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
LLAMAFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")
MS_SWIFT_ROOT = Path("/data/jiahao/task/ms-swift")
MS_SWIFT_EVALUATOR = MS_SWIFT_ROOT / "examples/train/grpo/plugin/bricknet/evaluate_experiment.py"
EVALUATOR = BRICKNET_ROOT / "scripts/evaluate_experiment.py"
ALIGNMENT_DATASET = BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"

CONFIG_ROOT = ROOT / "examples/train_lora"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
DATA_ROOT = ROOT / "data/bricknet_pt_exp2_mm_rowbal_cont3"
DATA_MANIFEST = DATA_ROOT / "manifest.v1.json"
DATA_FILE = DATA_ROOT / "PT-exp2-mm-rowbal-cont3.jsonl"
DATA_REGISTRY = DATA_ROOT / "dataset_info.json"
LOADER_REPORT = DATA_ROOT / "loader_validation_report.json"
PROCESSOR_AUDIT = DATA_ROOT / "processor_audit.json"
PROCESSOR_AUDIT_EVIDENCE = DATA_ROOT / "processor_audit_full"

TRAIN_CONFIG_NAME = "qwen35_08b_bricknet_pt_exp2_mm_rowbal_cont3.yaml"
TRAIN_CONFIG = CONFIG_ROOT / TRAIN_CONFIG_NAME
TRAIN_OUTPUT_NAME = (
    "train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_"
    "ep3_bs2_gbs16_lora64_len6400"
)
TRAIN_OUTPUT = SAVE_ROOT / TRAIN_OUTPUT_NAME
TRAIN_CACHE_BUILDER = ROOT / "scripts/build_tokenized_cache_with_length.py"
AUDIT_VALIDATOR = ROOT / "scripts/validate_bricknet_pt_exp2_mm_rowbal_cont3.py"
CACHE_BINDING_NAME = "rowbal_cache_manifest.json"

DATASET_NAME = "BrickNet-PT-exp2-mm-rowbal-cont3"
DATA_FILE_NAME = "PT-exp2-mm-rowbal-cont3.jsonl"
EXPECTED_ROWS = 270_102
EXPECTED_MM_ROWS = 135_051
EXPECTED_TEXT_ROWS = 135_051
EXPECTED_MM_TARGET_TOKENS = 26_285_148
EXPECTED_TEXT_TARGET_TOKENS = 227_155_208
EXPECTED_TEXT_TO_MM_TARGET_TOKEN_RATIO = 8.641960395277211
EXPECTED_ROW_BALANCE_RATIO = 1.0
EXPECTED_CUTOFF_LEN = 6_400
EXPECTED_GLOBAL_BATCH = 16
EXPECTED_MICRO_BATCH = 2
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_EPOCHS = 3
EXPECTED_MAX_STEPS = 50_646
EXPECTED_CHECKPOINT_STEPS = {1: 16_882, 2: 33_764, 3: 50_646}
EXPECTED_DATA_SCHEMA = ["id", "images", "messages"]
EXPECTED_MESSAGE_SCHEMA = ["role", "content"]
EXPECTED_ROLES = ["system", "user", "assistant"]
EXPECTED_PARENT_OUTPUT_NAME = (
    "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_"
    "lora64_len6401_nopack"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd"
)
EXPECTED_PARENT_ADAPTER = SAVE_ROOT / EXPECTED_PARENT_OUTPUT_NAME
EXPECTED_GPU = "1"
EXPECTED_VAL_SAMPLES = 512
EXPECTED_VAL_SHA256 = "510c68dc767746f621074fce0d69d81c3943043981fd02a40eebf010c56e3016"
VAL_DATASET = ROOT / "data/BrickNet-MM_PT_VAL.json"
VAL_REGISTRY = ROOT / "data/dataset_info.json"
VAL_REGISTRY_NAME = "BrickNet-MM-PT-VAL"

ALIGNMENT_SCHEMA_VERSION = "bricknet-pt-exp2-mm-alignment-v1"
POSE_TRANSLATION_TOLERANCE = 0.5
POSE_ROTATION_TOLERANCE = 5.0
POSE_SUCCESS_THRESHOLD = 1.0
REWARD_WEIGHTS = {
    "parse_prefix": 0.20,
    "inventory_f1": 0.20,
    "length_score": 0.10,
    "collision_prefix": 0.20,
    "pose_match": 0.30,
}


@dataclass(frozen=True)
class PredictionRun:
    epoch: int
    config_name: str
    checkpoint_step: int
    output_name: str

    @property
    def checkpoint(self) -> Path:
        return TRAIN_OUTPUT / f"checkpoint-{self.checkpoint_step}"

    @property
    def config(self) -> Path:
        return CONFIG_ROOT / self.config_name

    @property
    def output(self) -> Path:
        return SAVE_ROOT / self.output_name


PREDICTION_RUNS = {
    f"ep{epoch}": PredictionRun(
        epoch=epoch,
        config_name=f"qwen35_08b_bricknet_pt_exp2_mm_rowbal_cont3_ep{epoch}_predict.yaml",
        checkpoint_step=EXPECTED_CHECKPOINT_STEPS[epoch],
        output_name=f"eval_PT_exp2_mm_rowbal_cont3_ep{epoch}_ptval_in4096_out4096_p95_t1_k20",
    )
    for epoch in (1, 2, 3)
}
RUN_NAMES = tuple(PREDICTION_RUNS)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL row")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _resolve_under(root: Path, value: str | Path) -> Path:
    """Resolve a builder-declared path and reject paths outside its namespace."""

    path = _resolve(value, base=root).resolve()
    root_resolved = root.resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ValueError(f"declared path escapes data namespace: {path}")
    return path


def _manifest_value(manifest: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in manifest:
            return manifest[key]
    return None


def _report_decl(manifest: dict[str, Any], key: str, default: Path) -> tuple[Path, str | None]:
    declaration = manifest.get(key)
    if declaration is None:
        return default, None
    if isinstance(declaration, str):
        return _resolve_under(DATA_ROOT, declaration), None
    if not isinstance(declaration, dict):
        raise ValueError(f"manifest {key} declaration must be a path or object")
    path_value = declaration.get("path", declaration.get("file", default.name))
    path = _resolve_under(DATA_ROOT, path_value)
    declared_hash = declaration.get("sha256", declaration.get("content_sha256"))
    if declared_hash is not None and not isinstance(declared_hash, str):
        raise ValueError(f"manifest {key}.sha256 must be a string")
    return path, declared_hash


def _bool_value(value: Any) -> bool:
    return value is True


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _number_equal(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == expected


def _get_rows_value(payload: dict[str, Any]) -> int | None:
    for key in ("rows", "count", "preprocessed_rows", "raw_rows"):
        value = _int_value(payload.get(key))
        if value is not None:
            return value
    return None


def _full_hash_audit(path: Path) -> tuple[int, int, int, str, str, list[str]]:
    """Audit all data rows and return count, MM/text counts, content/id hashes."""

    content = hashlib.sha256()
    ordered_ids = hashlib.sha256()
    ids: set[str] = set()
    total = multimodal = text = 0
    first_schema: list[str] | None = None
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL row")
            content.update(raw)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            schema = list(row)
            if first_schema is None:
                first_schema = schema
            if schema != EXPECTED_DATA_SCHEMA:
                raise ValueError(
                    f"{path}:{line_no}: top-level schema {schema} != {EXPECTED_DATA_SCHEMA}"
                )
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_no}: id must be a non-empty string")
            if sample_id in ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {sample_id!r}")
            ids.add(sample_id)
            ordered_ids.update(sample_id.encode("utf-8"))
            ordered_ids.update(b"\n")

            images = row.get("images")
            if not isinstance(images, list) or any(not isinstance(item, str) for item in images):
                raise ValueError(f"{path}:{line_no}: images must be a list of strings")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
                raise ValueError(f"{path}:{line_no}: messages must be system/user/assistant")
            roles: list[str] = []
            for message in messages:
                if not isinstance(message, dict) or set(message) != set(EXPECTED_MESSAGE_SCHEMA):
                    raise ValueError(f"{path}:{line_no}: invalid message schema")
                if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                    raise ValueError(f"{path}:{line_no}: message values must be strings")
                roles.append(message["role"])
            if roles != EXPECTED_ROLES:
                raise ValueError(f"{path}:{line_no}: roles {roles} != {EXPECTED_ROLES}")
            if total < EXPECTED_MM_ROWS:
                if len(images) != 1:
                    raise ValueError(f"{path}:{line_no}: MM prefix must have one image")
                if "<image>" not in messages[1]["content"]:
                    raise ValueError(f"{path}:{line_no}: MM user prompt lacks <image>")
                multimodal += 1
            else:
                if images:
                    raise ValueError(f"{path}:{line_no}: text suffix must have images=[]")
                text += 1
            total += 1
    return total, multimodal, text, content.hexdigest(), ordered_ids.hexdigest(), first_schema or []


def _registry_expected() -> dict[str, Any]:
    return {
        DATASET_NAME: {
            "file_name": DATA_FILE_NAME,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def _validate_registry(manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {"path": str(DATA_REGISTRY), "exists": DATA_REGISTRY.is_file()}
    if not DATA_REGISTRY.is_file():
        return ["ROWBAL_REGISTRY_MISSING"], checks
    try:
        registry = _json(DATA_REGISTRY)
    except (OSError, json.JSONDecodeError) as exc:
        return ["ROWBAL_REGISTRY_INVALID"], {**checks, "error": str(exc)}
    expected = _registry_expected()
    checks["actual"] = registry
    checks["expected"] = expected
    if registry != expected:
        blockers.append("ROWBAL_REGISTRY_DRIFT")
    declared_hash = _manifest_value(manifest, "registry_sha256")
    registry_decl = manifest.get("registry")
    if isinstance(registry_decl, dict):
        declared_hash = registry_decl.get("sha256", declared_hash)
    checks["sha256"] = _sha256(DATA_REGISTRY)
    checks["sha256_expected"] = declared_hash
    if not isinstance(declared_hash, str) or checks["sha256"] != declared_hash:
        blockers.append("ROWBAL_REGISTRY_HASH_MISMATCH")
    return blockers, checks


def _validate_report_hash(path: Path, declared_hash: str | None, checks: dict[str, Any], label: str) -> None:
    if declared_hash is not None:
        checks[f"{label}_sha256_expected"] = declared_hash
        checks[f"{label}_sha256_actual"] = _sha256(path)
        if checks[f"{label}_sha256_actual"] != declared_hash:
            raise ValueError(f"{label} hash mismatch")


def _validate_data() -> tuple[list[str], dict[str, Any]]:
    """Validate the immutable builder contract and every row (no model/GPU)."""

    blockers: list[str] = []
    checks: dict[str, Any] = {
        "manifest": str(DATA_MANIFEST),
        "dataset": DATASET_NAME,
        "expected_rows": EXPECTED_ROWS,
        "expected_multimodal_rows": EXPECTED_MM_ROWS,
        "expected_text_rows": EXPECTED_TEXT_ROWS,
    }
    if not DATA_MANIFEST.is_file():
        return ["WAIT_ROWBAL_DATA_MANIFEST"], checks
    try:
        manifest = _json(DATA_MANIFEST)
    except (OSError, json.JSONDecodeError) as exc:
        return ["ROWBAL_DATA_MANIFEST_INVALID"], {**checks, "error": str(exc)}
    if not isinstance(manifest, dict):
        return ["ROWBAL_DATA_MANIFEST_INVALID"], {**checks, "error": "manifest must be an object"}
    checks["manifest_sha256"] = _sha256(DATA_MANIFEST)
    checks["manifest_status"] = manifest.get("status")
    checks["manifest_eligible"] = manifest.get("eligible") is True
    if manifest.get("status") not in {"validated", "frozen"} or not checks["manifest_eligible"]:
        blockers.append("ROWBAL_DATA_MANIFEST_NOT_ELIGIBLE")
    if manifest.get("dataset") != DATASET_NAME:
        blockers.append("ROWBAL_DATASET_NAME_DRIFT")
    declared_file = manifest.get("dataset_file", manifest.get("file", DATA_FILE_NAME))
    if isinstance(manifest.get("data"), dict):
        declared_file = manifest["data"].get("file", declared_file)
    try:
        data_file = _resolve_under(DATA_ROOT, declared_file)
    except (TypeError, ValueError) as exc:
        return [*blockers, "ROWBAL_DATA_FILE_DECLARATION_INVALID"], {**checks, "error": str(exc)}
    checks["data_file"] = str(data_file)
    checks["data_file_expected"] = str(DATA_FILE)
    if data_file != DATA_FILE.resolve():
        blockers.append("ROWBAL_DATA_FILE_NAMESPACE_DRIFT")
    if not data_file.is_file():
        blockers.append("WAIT_ROWBAL_DATA_FILE")
        return blockers, checks

    declared_rows = _int_value(_manifest_value(manifest, "rows", "count"))
    declared_mm = _int_value(_manifest_value(manifest, "multimodal_rows", "mm_rows"))
    declared_text = _int_value(_manifest_value(manifest, "text_rows", "text_only_rows"))
    checks.update({"manifest_rows": declared_rows, "manifest_multimodal_rows": declared_mm, "manifest_text_rows": declared_text})
    if (declared_rows, declared_mm, declared_text) != (EXPECTED_ROWS, EXPECTED_MM_ROWS, EXPECTED_TEXT_ROWS):
        blockers.append("ROWBAL_DATA_MANIFEST_COUNTS_DRIFT")

    schema = manifest.get("schema", manifest.get("dataset_schema"))
    checks["schema"] = schema
    expected_schema = {
        "top_level_fields": EXPECTED_DATA_SCHEMA,
        "top_level_key_signatures": {"id,images,messages": EXPECTED_ROWS},
        "message_fields": EXPECTED_MESSAGE_SCHEMA,
        "message_roles": EXPECTED_ROLES,
        "message_role_sequences": {"system,user,assistant": EXPECTED_ROWS},
        "row_order": "multimodal_prefix_then_text_suffix",
    }
    if schema != expected_schema:
        blockers.append("ROWBAL_DATA_SCHEMA_DECLARATION_DRIFT")

    declared_content = _manifest_value(manifest, "sha256", "content_sha256", "data_sha256")
    checks["content_sha256_expected"] = declared_content
    declared_ordered = _manifest_value(manifest, "ordered_id_sha256", "ordered_ids_sha256")
    checks["ordered_id_sha256_expected"] = declared_ordered
    try:
        total, mm_rows, text_rows, content_hash, ordered_hash, actual_schema = _full_hash_audit(data_file)
    except (OSError, ValueError) as exc:
        blockers.append("ROWBAL_DATA_ROW_AUDIT_FAILED")
        checks["row_audit_error"] = str(exc)
        return blockers, checks
    checks.update(
        {
            "actual_rows": total,
            "actual_multimodal_rows": mm_rows,
            "actual_text_rows": text_rows,
            "content_sha256_actual": content_hash,
            "ordered_id_sha256_actual": ordered_hash,
            "actual_schema": actual_schema,
        }
    )
    if (total, mm_rows, text_rows) != (EXPECTED_ROWS, EXPECTED_MM_ROWS, EXPECTED_TEXT_ROWS):
        blockers.append("ROWBAL_DATA_COUNTS_MISMATCH")
    if not isinstance(declared_content, str) or content_hash != declared_content:
        blockers.append("ROWBAL_DATA_CONTENT_HASH_MISMATCH")
    if not isinstance(declared_ordered, str) or ordered_hash != declared_ordered:
        blockers.append("ROWBAL_DATA_ORDERED_ID_HASH_MISMATCH")

    arrow = manifest.get("arrow_materialization", manifest.get("arrow"))
    checks["arrow_materialization"] = arrow
    arrow_ok = (
        isinstance(arrow, dict)
        and arrow.get("eligible") is True
        and _get_rows_value(arrow) == EXPECTED_ROWS
        and arrow.get("columns") == EXPECTED_DATA_SCHEMA
    )
    if not arrow_ok:
        blockers.append("ROWBAL_ARROW_MATERIALIZATION_GATE_FAILED")

    target_token_mass = manifest.get("target_token_mass")
    target_token_gates = manifest.get("gates")
    target_mm = target_token_mass.get("multimodal") if isinstance(target_token_mass, dict) else None
    target_text = target_token_mass.get("selected_text") if isinstance(target_token_mass, dict) else None
    target_token_expected = {
        "required": True,
        "eligible": True,
        "gate": True,
        "multimodal": {"rows": EXPECTED_MM_ROWS, "total_tokens": EXPECTED_MM_TARGET_TOKENS},
        "selected_text": {"rows": EXPECTED_TEXT_ROWS, "total_tokens": EXPECTED_TEXT_TARGET_TOKENS},
        "text_to_multimodal_target_token_ratio": EXPECTED_TEXT_TO_MM_TARGET_TOKEN_RATIO,
        "row_balance_ratio": EXPECTED_ROW_BALANCE_RATIO,
    }
    target_token_actual = {
        "required": target_token_mass.get("required") if isinstance(target_token_mass, dict) else None,
        "eligible": target_token_mass.get("eligible") if isinstance(target_token_mass, dict) else None,
        "gate": target_token_gates.get("target_token_audit") if isinstance(target_token_gates, dict) else None,
        "multimodal": {
            "rows": target_mm.get("rows") if isinstance(target_mm, dict) else None,
            "total_tokens": target_mm.get("total_tokens") if isinstance(target_mm, dict) else None,
        },
        "selected_text": {
            "rows": target_text.get("rows") if isinstance(target_text, dict) else None,
            "total_tokens": target_text.get("total_tokens") if isinstance(target_text, dict) else None,
        },
        "text_to_multimodal_target_token_ratio": (
            target_token_mass.get("text_to_multimodal_target_token_ratio")
            if isinstance(target_token_mass, dict)
            else None
        ),
        "row_balance_ratio": target_token_mass.get("row_balance_ratio") if isinstance(target_token_mass, dict) else None,
    }
    target_token_ok = (
        target_token_actual["required"] is True
        and target_token_actual["eligible"] is True
        and target_token_actual["gate"] is True
        and _int_value(target_mm.get("rows") if isinstance(target_mm, dict) else None) == EXPECTED_MM_ROWS
        and _int_value(target_mm.get("total_tokens") if isinstance(target_mm, dict) else None)
        == EXPECTED_MM_TARGET_TOKENS
        and _int_value(target_text.get("rows") if isinstance(target_text, dict) else None) == EXPECTED_TEXT_ROWS
        and _int_value(target_text.get("total_tokens") if isinstance(target_text, dict) else None)
        == EXPECTED_TEXT_TARGET_TOKENS
        and _number_equal(
            target_token_actual["text_to_multimodal_target_token_ratio"],
            EXPECTED_TEXT_TO_MM_TARGET_TOKEN_RATIO,
        )
        and _number_equal(target_token_actual["row_balance_ratio"], EXPECTED_ROW_BALANCE_RATIO)
    )
    checks["target_token_mass"] = {
        "expected": target_token_expected,
        "actual": target_token_actual,
        "eligible": target_token_ok,
    }
    if not target_token_ok:
        blockers.append("ROWBAL_TARGET_TOKEN_MASS_GATE_FAILED")

    registry_blockers, registry_checks = _validate_registry(manifest)
    blockers.extend(registry_blockers)
    checks["registry"] = registry_checks

    # The processor audit resolves image paths through the same media root as
    # the training YAML.  Keep this binding explicit: passing the repository
    # root would make ``images/PT/...`` resolve to a nonexistent path and turn
    # every multimodal row into an audit error.
    expected_processor_media_root: Path | None = None
    try:
        train_config = _config_mapping(TRAIN_CONFIG)
        declared_media_dir = train_config.get("media_dir")
        if isinstance(declared_media_dir, str) and declared_media_dir:
            expected_processor_media_root = _resolve(declared_media_dir).resolve()
    except (OSError, ValueError, yaml.YAMLError):
        pass
    checks["processor_media_root_expected"] = (
        str(expected_processor_media_root) if expected_processor_media_root is not None else None
    )

    current_train_config_sha = _sha256(TRAIN_CONFIG) if TRAIN_CONFIG.is_file() else None
    checks["train_config_sha256"] = current_train_config_sha
    for label, default in (("loader_validation", LOADER_REPORT), ("processor_audit", PROCESSOR_AUDIT)):
        try:
            report_path, declared_hash = _report_decl(manifest, label, default)
        except (TypeError, ValueError) as exc:
            blockers.append(f"ROWBAL_{label.upper()}_DECLARATION_INVALID")
            checks[label] = {"error": str(exc)}
            continue
        entry: dict[str, Any] = {"path": str(report_path), "exists": report_path.is_file()}
        checks[label] = entry
        if not report_path.is_file():
            blockers.append(f"WAIT_ROWBAL_{label.upper()}_REPORT")
            continue
        try:
            report = _json(report_path)
            _validate_report_hash(report_path, declared_hash, entry, label)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            blockers.append(f"ROWBAL_{label.upper()}_REPORT_INVALID")
            entry["error"] = str(exc)
            continue
        entry["report"] = report
        report_dataset = report.get("dataset")
        report_rows = _get_rows_value(report)
        report_data_hash = report.get("dataset_sha256", report.get("data_sha256"))
        report_manifest_hash = report.get("manifest_sha256")
        if label == "loader_validation":
            valid = (
                report.get("eligible") is True
                and report.get("raw_full_materialization") is True
                and report.get("model_weights_loaded") is False
                and report.get("optimizer_started") is False
                and report.get("trainer_started") is False
                and report_dataset == DATASET_NAME
                and report_rows == EXPECTED_ROWS
                and report.get("columns") == EXPECTED_DATA_SCHEMA
                and _int_value(report.get("preprocessed_rows")) is not None
                and _int_value(report.get("preprocessed_rows")) >= 4
                and isinstance(report.get("preprocessed_columns"), list)
                and isinstance(report.get("arrow_materialization"), dict)
                and report["arrow_materialization"].get("eligible") is True
                and report["arrow_materialization"].get("rows") == EXPECTED_ROWS
                and report["arrow_materialization"].get("columns") == EXPECTED_DATA_SCHEMA
                and report_data_hash == content_hash
                and report_manifest_hash == checks["manifest_sha256"]
                and isinstance(current_train_config_sha, str)
                and report.get("config_sha256") == current_train_config_sha
            )
        else:
            errors = report.get("errors", report.get("error_count"))
            truncation = report.get("truncated", report.get("truncation_count"))
            source_report = report.get("source_report")
            source_summary = report.get("source_summary")
            source_evidence_root = report.get("source_evidence_root")
            processor_runtime = report.get("config") or report.get("runtime")
            source_report_path: Path | None = None
            source_sidecar_path: Path | None = None
            source_evidence_path: Path | None = None
            source_report_hash_ok = False
            source_sidecar_hash_ok = False
            processor_media_root_actual: Path | None = None
            processor_media_root_ok = False
            try:
                if isinstance(processor_runtime, dict) and isinstance(processor_runtime.get("media_dir"), str):
                    processor_media_root_actual = _resolve(processor_runtime["media_dir"]).resolve()
                    processor_media_root_ok = (
                        expected_processor_media_root is not None
                        and processor_media_root_actual == expected_processor_media_root
                    )
                if isinstance(source_report, str):
                    source_report_path = _resolve_under(DATA_ROOT, source_report)
                if isinstance(source_evidence_root, str):
                    source_evidence_path = _resolve_under(DATA_ROOT, source_evidence_root)
                if isinstance(source_summary, dict) and isinstance(source_summary.get("sidecar"), str):
                    source_sidecar_path = _resolve_under(DATA_ROOT, source_summary["sidecar"])
                source_report_hash_ok = (
                    source_report_path is not None
                    and source_report_path.is_file()
                    and report.get("source_report_sha256") == _sha256(source_report_path)
                )
                source_sidecar_hash_ok = (
                    source_sidecar_path is not None
                    and source_sidecar_path.is_file()
                    and source_summary.get("sidecar_sha256") == _sha256(source_sidecar_path)
                )
            except (TypeError, ValueError, OSError):
                source_report_hash_ok = source_sidecar_hash_ok = False
            valid = (
                report.get("eligible", report.get("training_eligible")) is True
                and report.get("full_pool", report.get("is_full_pool")) is True
                and report.get("training_eligible") is True
                and report.get("zero_errors") is True
                and report.get("zero_truncation") is True
                and report_dataset == DATASET_NAME
                and report_rows == EXPECTED_ROWS
                and report_data_hash == content_hash
                and report_manifest_hash == checks["manifest_sha256"]
                and errors == 0
                and truncation == 0
                and report.get("cutoff_len", EXPECTED_CUTOFF_LEN) == EXPECTED_CUTOFF_LEN
                and processor_media_root_ok
                and source_evidence_path == PROCESSOR_AUDIT_EVIDENCE.resolve()
                and isinstance(source_summary, dict)
                and source_summary.get("count") == EXPECTED_ROWS
                and source_summary.get("errors") == 0
                and source_summary.get("truncated") == 0
                and source_report_hash_ok
                and source_sidecar_hash_ok
            )
            entry["source_evidence_root"] = str(source_evidence_path) if source_evidence_path else None
            entry["source_report"] = str(source_report_path) if source_report_path else None
            entry["source_sidecar"] = str(source_sidecar_path) if source_sidecar_path else None
            entry["source_report_hash_ok"] = source_report_hash_ok
            entry["source_sidecar_hash_ok"] = source_sidecar_hash_ok
            entry["processor_media_root_actual"] = (
                str(processor_media_root_actual) if processor_media_root_actual is not None else None
            )
            entry["processor_media_root_ok"] = processor_media_root_ok
        entry["eligible"] = bool(valid)
        if not valid:
            blockers.append(f"ROWBAL_{label.upper()}_GATE_FAILED")
    checks["eligible"] = not blockers
    return blockers, checks


def _config_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _cache_path(config: dict[str, Any]) -> Path | None:
    value = config.get("tokenized_path")
    if not isinstance(value, str) or not value:
        return None
    return _resolve(value).resolve()


def _cache_check(config_path: Path, *, require_binding: bool = True) -> dict[str, Any]:
    """Check cache metadata and the complete persisted train split.

    ``length_cache_manifest.json`` is only a declared contract.  The actual
    Arrow train split is loaded read-only and every row's persisted ``length``
    is compared with the vectorized Arrow list length of ``input_ids``.
    """

    result: dict[str, Any] = {
        "config": str(config_path),
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "eligible": False,
        "length_column": "length",
        "sampling_strategy": None,
        "actual_train_rows": None,
        "actual_train_columns": None,
        "length_mismatch_count": None,
    }
    if not config_path.is_file():
        result["error"] = "CONFIG_MISSING"
        return result
    try:
        config = _config_mapping(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        result["error"] = f"CONFIG_INVALID: {exc}"
        return result
    cache = _cache_path(config)
    result["cache"] = str(cache) if cache else None
    result["length_column"] = config.get("length_column_name")
    result["sampling_strategy"] = config.get("train_sampling_strategy")
    if cache is None:
        result["error"] = "TOKENIZED_PATH_MISSING"
        return result
    if config.get("length_column_name") != "length":
        result["error"] = "LENGTH_COLUMN_CONFIG_MUST_BE_LENGTH"
        return result
    # A length field is persisted for integrity and optional future diagnosis;
    # random sampling is the frozen PT-exp2 v2 behavior.  It must not become
    # group_by_length merely because the cache contains lengths.
    if config.get("train_sampling_strategy") != "random":
        result["error"] = "SAMPLING_STRATEGY_MUST_REMAIN_RANDOM"
        return result
    if not cache.is_dir():
        result["error"] = "CACHE_MISSING"
        result["build_required"] = True
        return result

    train_info = cache / "train/dataset_info.json"
    length_manifest_path = cache / "length_cache_manifest.json"
    binding_path = cache / CACHE_BINDING_NAME
    result.update(
        {
            "train_dataset_info": str(train_info),
            "length_cache_manifest": str(length_manifest_path),
            "binding_manifest": str(binding_path),
            "build_required": False,
        }
    )
    if not train_info.is_file():
        result["error"] = "CACHE_TRAIN_DATASET_INFO_MISSING"
        return result
    if not length_manifest_path.is_file():
        result["error"] = "CACHE_LENGTH_MANIFEST_MISSING"
        return result
    try:
        info = _json(train_info)
        length_manifest = _json(length_manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"CACHE_METADATA_INVALID: {exc}"
        return result
    if not isinstance(info, dict) or not isinstance(length_manifest, dict):
        result["error"] = "CACHE_METADATA_INVALID"
        return result
    features = info.get("features", {})
    result["columns"] = sorted(features) if isinstance(features, dict) else None
    required_columns = {"input_ids", "attention_mask", "labels", "images", "length"}
    if not isinstance(features, dict) or not required_columns.issubset(features):
        result["error"] = "CACHE_REQUIRED_COLUMNS_MISSING"
        return result
    train_rows = None
    # build_tokenized_cache_with_length writes authoritative rows in stats;
    # dataset_info.json itself intentionally contains only Arrow features.
    stats = length_manifest.get("stats", {})
    if isinstance(stats, dict):
        train_stats = stats.get("train", {})
        if isinstance(train_stats, dict):
            train_rows = _int_value(train_stats.get("rows"))
    result["train_rows"] = train_rows
    if train_rows != EXPECTED_ROWS:
        result["error"] = "CACHE_ROW_COUNT_MISMATCH"
        return result
    result["length_manifest_action"] = length_manifest.get("action")
    result["length_manifest_length_column"] = length_manifest.get("length_column")
    result["length_manifest_input_column"] = length_manifest.get("input_column")
    if length_manifest.get("length_column") != "length" or length_manifest.get("input_column") != "input_ids":
        result["error"] = "CACHE_LENGTH_MANIFEST_BINDING_MISMATCH"
        return result
    if length_manifest.get("output_cache"):
        try:
            if _resolve(length_manifest["output_cache"]).resolve() != cache:
                result["error"] = "CACHE_OUTPUT_PATH_MISMATCH"
                return result
        except (TypeError, ValueError):
            result["error"] = "CACHE_OUTPUT_PATH_INVALID"
            return result

    # Verify the persisted Arrow artifact itself.  Metadata and the length
    # manifest can both be stale or hand-edited, so neither is sufficient for
    # eligibility.  ``list_value_length`` avoids materializing all token IDs;
    # the comparison still visits every row in the train split.
    try:
        import pyarrow.compute as pc
        from datasets import Dataset, DatasetDict, load_from_disk

        loaded = load_from_disk(str(cache))
        if isinstance(loaded, DatasetDict):
            if "train" not in loaded:
                result["error"] = "CACHE_TRAIN_SPLIT_MISSING"
                return result
            train_dataset = loaded["train"]
        elif isinstance(loaded, Dataset):
            train_dataset = loaded
        else:
            result["error"] = "CACHE_DATASET_TYPE_INVALID"
            return result

        actual_rows = len(train_dataset)
        actual_columns = list(train_dataset.column_names)
        result["actual_train_rows"] = actual_rows
        result["actual_train_columns"] = actual_columns
        actual_missing = sorted(required_columns.difference(actual_columns))
        result["actual_required_columns_missing"] = actual_missing
        if actual_rows != EXPECTED_ROWS:
            result["error"] = "CACHE_ACTUAL_ROW_COUNT_MISMATCH"
            return result
        if actual_missing:
            result["error"] = "CACHE_ACTUAL_REQUIRED_COLUMNS_MISSING"
            return result

        declared_lengths = train_dataset.data.column("length")
        actual_lengths = pc.list_value_length(train_dataset.data.column("input_ids"))
        if len(declared_lengths) != actual_rows or len(actual_lengths) != actual_rows:
            result["error"] = "CACHE_ACTUAL_LENGTH_COLUMN_ROW_COUNT_MISMATCH"
            return result
        mismatch_count = sum(
            1
            for declared, actual in zip(declared_lengths, actual_lengths, strict=True)
            if declared.as_py() != actual.as_py()
        )
        result["length_mismatch_count"] = mismatch_count
        result["length_validation"] = {
            "rows": actual_rows,
            "method": "pyarrow.compute.list_value_length",
            "mismatch": mismatch_count,
        }
        if mismatch_count != 0:
            result["error"] = "CACHE_ACTUAL_LENGTH_MISMATCH"
            return result
    except Exception as exc:
        result["error"] = f"CACHE_ACTUAL_TRAIN_SPLIT_VALIDATION_FAILED: {type(exc).__name__}: {exc}"
        return result

    if not require_binding:
        result["eligible"] = True
        return result
    if not binding_path.is_file():
        result["error"] = "CACHE_SOURCE_BINDING_MISSING"
        return result
    try:
        binding = _json(binding_path)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"CACHE_SOURCE_BINDING_INVALID: {exc}"
        return result
    result["binding"] = binding
    source = binding.get("source") if isinstance(binding, dict) else None
    binding_ok = (
        isinstance(binding, dict)
        and binding.get("schema_version") == 1
        and binding.get("dataset") == DATASET_NAME
        and binding.get("config_sha256") == result["config_sha256"]
        and isinstance(source, dict)
        and source.get("manifest_sha256") == (_sha256(DATA_MANIFEST) if DATA_MANIFEST.is_file() else None)
        and source.get("dataset_sha256") == binding.get("dataset_sha256")
        and source.get("ordered_id_sha256") == binding.get("ordered_id_sha256")
        and source.get("rows") == EXPECTED_ROWS
        and binding.get("dataset_sha256") is not None
        and binding.get("ordered_id_sha256") is not None
        and binding.get("rows") == EXPECTED_ROWS
        and binding.get("length_column") == "length"
        and binding.get("sampling_strategy") == "random"
    )
    # Bind against the current immutable data manifest, not just a stale cache
    # sidecar.  The values are checked in _validate_data; here only the hashes
    # need to be compared to avoid a second full 270k-row scan.
    try:
        current_manifest = _json(DATA_MANIFEST)
        current_data_hash = _manifest_value(current_manifest, "sha256", "content_sha256", "data_sha256")
        current_order_hash = _manifest_value(current_manifest, "ordered_id_sha256", "ordered_ids_sha256")
    except (OSError, json.JSONDecodeError):
        current_data_hash = current_order_hash = None
    binding_ok = binding_ok and binding.get("dataset_sha256") == current_data_hash and binding.get("ordered_id_sha256") == current_order_hash
    result["binding_eligible"] = bool(binding_ok)
    if not binding_ok:
        result["error"] = "CACHE_SOURCE_BINDING_MISMATCH"
        return result
    result["eligible"] = True
    return result


def _cache_binding_payload(config_path: Path, cache_check: dict[str, Any], data_checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": DATASET_NAME,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "cache": cache_check.get("cache"),
        "rows": EXPECTED_ROWS,
        "dataset_sha256": data_checks.get("content_sha256_actual"),
        "ordered_id_sha256": data_checks.get("ordered_id_sha256_actual"),
        "length_column": "length",
        "sampling_strategy": "random",
        "source": {
            "manifest": str(DATA_MANIFEST.resolve()),
            "manifest_sha256": data_checks.get("manifest_sha256"),
            "dataset": DATASET_NAME,
            "dataset_sha256": data_checks.get("content_sha256_actual"),
            "ordered_id_sha256": data_checks.get("ordered_id_sha256_actual"),
            "rows": EXPECTED_ROWS,
        },
    }


def _cache_build_command(config_path: Path, args: argparse.Namespace) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "python",
        str(TRAIN_CACHE_BUILDER),
        "--config",
        str(config_path),
        "--batch-size",
        str(args.cache_batch_size),
        "--num-proc",
        str(args.cache_num_proc),
    ]


def _audit_report_paths() -> tuple[Path, Path]:
    """Resolve report destinations from the manifest, with fixed defaults."""

    if not DATA_MANIFEST.is_file():
        return LOADER_REPORT, PROCESSOR_AUDIT
    try:
        manifest = _json(DATA_MANIFEST)
        return (
            _report_decl(manifest, "loader_validation", LOADER_REPORT)[0],
            _report_decl(manifest, "processor_audit", PROCESSOR_AUDIT)[0],
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return LOADER_REPORT, PROCESSOR_AUDIT


def _audit_report_blocker(blocker: str) -> bool:
    return any(
        blocker.startswith(prefix)
        for prefix in (
            "WAIT_ROWBAL_LOADER_VALIDATION_REPORT",
            "ROWBAL_LOADER_VALIDATION_REPORT_INVALID",
            "ROWBAL_LOADER_VALIDATION_GATE_FAILED",
            "WAIT_ROWBAL_PROCESSOR_AUDIT_REPORT",
            "ROWBAL_PROCESSOR_AUDIT_REPORT_INVALID",
            "ROWBAL_PROCESSOR_AUDIT_GATE_FAILED",
        )
    )


def _audit_command(
    loader_output: Path, processor_output: Path, args: argparse.Namespace
) -> list[str]:
    return [
        str(LLAMAFACTORY_PYTHON),
        str(AUDIT_VALIDATOR),
        "--data-root",
        str(DATA_ROOT),
        "--manifest",
        str(DATA_MANIFEST),
        "--data-file",
        str(DATA_FILE),
        "--registry",
        str(DATA_REGISTRY),
        "--config",
        str(TRAIN_CONFIG),
        "--loader-output",
        str(loader_output),
        "--processor-output",
        str(processor_output),
        "--processor-evidence-root",
        str(PROCESSOR_AUDIT_EVIDENCE),
        "--processor-workers",
        str(args.processor_workers),
        "--processor-chunksize",
        str(args.processor_chunksize),
        "--audit-script",
        str(ROOT / "scripts/audit_bricknet_reasoning_tokens.py"),
        "--python",
        str(LLAMAFACTORY_PYTHON),
        "--execute",
    ]


def _prepare_audits(args: argparse.Namespace) -> None:
    data_blockers, data_checks = _validate_data()
    loader_output, processor_output = _audit_report_paths()
    hard_blockers = [blocker for blocker in data_blockers if not _audit_report_blocker(blocker)]
    existing = [
        str(path)
        for path in (loader_output, processor_output, PROCESSOR_AUDIT_EVIDENCE)
        if path.exists()
    ]
    if existing:
        hard_blockers.append("ROWBAL_AUDIT_OUTPUT_MUST_BE_ABSENT")
    command = _audit_command(loader_output, processor_output, args)
    payload = {
        "action": "prepare-audits",
        "dataset": DATASET_NAME,
        "ready": not hard_blockers,
        "blockers": hard_blockers,
        "data": data_checks,
        "report_outputs": [str(loader_output), str(processor_output)],
        "processor_evidence_root": str(PROCESSOR_AUDIT_EVIDENCE),
        "existing_reports": existing,
        "cpu_only": True,
        "model_weights_loaded": False,
        "optimizer_started": False,
        "trainer_started": False,
        "command": shlex.join(command),
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if hard_blockers:
        raise SystemExit("PT-exp2 MM rowbal audits blocked; resolve the reported gates first")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        env.pop(key, None)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    final_blockers, final_checks = _validate_data()
    if final_blockers:
        raise RuntimeError(f"audit validator exited successfully but reports failed the data gate: {final_checks}")
    print(
        json.dumps(
            {
                "event": "rowbal_audits_ready",
                "dataset": DATASET_NAME,
                "report_outputs": [str(loader_output), str(processor_output)],
                "cpu_only": True,
                "executed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _prepare_cache(args: argparse.Namespace) -> None:
    data_blockers, data_checks = _validate_data()
    config = TRAIN_CONFIG
    cache = _cache_check(config, require_binding=True)
    blockers = list(data_blockers)
    if cache.get("eligible"):
        action = "reuse"
    elif cache.get("error") == "CACHE_MISSING":
        action = "build"
    else:
        action = "blocked"
        blockers.append(f"CACHE_NOT_ELIGIBLE:{cache.get('error', 'UNKNOWN')}")
    command = _cache_build_command(config, args)
    payload = {
        "action": "prepare-cache",
        "dataset": DATASET_NAME,
        "ready": not blockers and action in {"reuse", "build"},
        "blockers": blockers,
        "data": data_checks,
        "cache": cache,
        "build_required": action == "build",
        "cpu_only": True,
        "model_weights_loaded": False,
        "trainer_started": False,
        "command": shlex.join(command) if action == "build" else None,
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM rowbal cache preparation blocked; resolve the reported gates first")
    if action == "reuse":
        return
    if action != "build":
        raise SystemExit("PT-exp2 MM rowbal cache is not eligible")
    env = os.environ.copy()
    # The cache builder calls tokenizer/data preprocessing only.  Empty visible
    # devices make accidental CUDA use fail closed while preserving CPU work.
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        env.pop(key, None)
    # Training intentionally keeps bf16=true.  Transformers rejects that value
    # when no GPU is visible, even for a do_train=false parser-only cache build;
    # use a short-lived CPU parser override rather than changing the training
    # YAML or letting the cache builder enter a Trainer.
    with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-rowbal-cache-config-") as temporary:
        cpu_config = Path(temporary) / TRAIN_CONFIG.name
        cpu_mapping = _config_mapping(TRAIN_CONFIG)
        cpu_mapping.update({"use_cpu": True, "bf16": False, "fp16": False})
        cpu_config.write_text(yaml.safe_dump(cpu_mapping, sort_keys=False), encoding="utf-8")
        subprocess.run(_cache_build_command(cpu_config, args), cwd=ROOT, env=env, check=True)
    after = _cache_check(config, require_binding=False)
    if not after.get("eligible"):
        raise RuntimeError(f"cache builder did not produce an eligible cache: {after}")
    binding_path = _resolve(after["cache"]) / CACHE_BINDING_NAME
    if binding_path.exists():
        raise FileExistsError(f"cache binding exists; refusing to overwrite: {binding_path}")
    _write_json(binding_path, _cache_binding_payload(config, after, data_checks))
    final = _cache_check(config, require_binding=True)
    if not final.get("eligible"):
        raise RuntimeError(f"cache source binding failed after build: {final}")
    print(json.dumps({"event": "rowbal_cache_ready", "cache": final, "executed": True}, ensure_ascii=False, indent=2))


def _adapter_weights(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _adapter_ready(path: Path, *, require_training_results: bool = False) -> bool:
    required = (path / "adapter_config.json").is_file() and _adapter_weights(path) is not None
    if require_training_results:
        required = required and (path / "trainer_state.json").is_file() and (path / "train_results.json").is_file()
    return required


def _visible_gpu_selectors() -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [item.strip() for item in visible.split(",") if item.strip()]


def _selected_gpu_processes() -> list[str]:
    """Report compute processes on physical CUDA 1 without blocking preflight."""

    command = [
        "nvidia-smi",
        "-i",
        EXPECTED_GPU,
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gpu_gate() -> tuple[list[str], dict[str, Any]]:
    selectors = _visible_gpu_selectors()
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu": EXPECTED_GPU,
        "gpu_processes": _selected_gpu_processes(),
    }
    if selectors is None:
        blockers.append("SET_EXPLICIT_GPU_1")
    elif selectors != [EXPECTED_GPU]:
        blockers.append("ROWBAL_REQUIRES_EXACTLY_CUDA1")
    if checks["gpu_processes"]:
        blockers.append("CUDA1_COMPUTE_PROCESSES_ACTIVE")
    checks["selectors"] = selectors
    checks["eligible"] = not blockers
    return blockers, checks


def _relative_save_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_train_config(config_path: Path) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {"config": str(config_path), "exists": config_path.is_file()}
    if not config_path.is_file():
        return ["TRAIN_CONFIG_MISSING"], checks
    try:
        config = _config_mapping(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ["TRAIN_CONFIG_INVALID"], {**checks, "error": str(exc)}
    expected = {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "adapter_name_or_path": _relative_save_path(EXPECTED_PARENT_ADAPTER),
        "create_new_adapter": False,
        "dataset": DATASET_NAME,
        "dataset_dir": "data/bricknet_pt_exp2_mm_rowbal_cont3",
        "media_dir": "data",
        "template": "qwen3_5_nothink",
        "enable_thinking": False,
        "cutoff_len": EXPECTED_CUTOFF_LEN,
        "train_on_prompt": False,
        "packing": False,
        "lora_target": "all",
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "per_device_train_batch_size": EXPECTED_MICRO_BATCH,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "learning_rate": 1.0e-5,
        "num_train_epochs": float(EXPECTED_EPOCHS),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "bf16": True,
        "fp16": False,
        "ddp_find_unused_parameters": False,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "save_only_model": False,
        "train_sampling_strategy": "random",
        "length_column_name": "length",
        "dataloader_drop_last": False,
        "do_eval": False,
        "eval_strategy": "no",
        "seed": 42,
    }
    actual = {key: config.get(key) for key in expected}
    checks.update({"expected": expected, "actual": actual})
    for key, expected_value in expected.items():
        # Transformers treats an omitted fp16 field as its false default.  An
        # explicit non-bool value or true must still fail closed.
        if key == "fp16":
            matches = config.get(key) is None or config.get(key) is False
        else:
            matches = actual[key] == expected_value
        if not matches:
            blockers.append(f"TRAIN_CONFIG_DRIFT:{key}")
    if config.get("stage") != "sft" or config.get("do_train") is not True or config.get("finetuning_type") != "lora":
        blockers.append("TRAIN_CONFIG_MODE_DRIFT")
    output_value = config.get("output_dir")
    expected_output = _relative_save_path(TRAIN_OUTPUT)
    checks["output_dir"] = output_value
    checks["output_dir_expected"] = expected_output
    if output_value != expected_output:
        blockers.append("TRAIN_CONFIG_OUTPUT_DRIFT")
    checks["expected_steps_per_epoch"] = (EXPECTED_ROWS + EXPECTED_GLOBAL_BATCH - 1) // EXPECTED_GLOBAL_BATCH
    checks["expected_max_steps"] = EXPECTED_MAX_STEPS
    if checks["expected_steps_per_epoch"] != EXPECTED_CHECKPOINT_STEPS[1]:
        blockers.append("ROWBAL_STEP_ARITHMETIC_INTERNAL_ERROR")
    if config.get("eval_dataset") is not None:
        blockers.append("TRAIN_CONFIG_MUST_NOT_DEFINE_EVAL_DATASET")
    checks["eligible"] = not blockers
    return blockers, checks


def _read_training_args(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Deserialize ``training_args.bin`` in a CPU-only helper process.

    The artifact is a torch pickle containing LlamaFactory's typed arguments;
    loading it with ``map_location=cpu`` verifies schedule/world-size binding
    without loading model weights or constructing a Trainer.
    """

    if not path.is_file():
        return None, "TRAINING_ARGS_MISSING"
    helper = r'''
import json, sys
import torch
value = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
def enum(v):
    return getattr(v, "value", v)
def nproc(v):
    d = getattr(v, "distributed_state", None)
    return getattr(d, "num_processes", None)
attrs = {
    "output_dir": getattr(value, "output_dir", None),
    "per_device_train_batch_size": getattr(value, "per_device_train_batch_size", None),
    "gradient_accumulation_steps": getattr(value, "gradient_accumulation_steps", None),
    "num_train_epochs": getattr(value, "num_train_epochs", None),
    "max_steps": getattr(value, "max_steps", None),
    "learning_rate": getattr(value, "learning_rate", None),
    "lr_scheduler_type": enum(getattr(value, "lr_scheduler_type", None)),
    "warmup_ratio": getattr(value, "warmup_ratio", None),
    "save_strategy": enum(getattr(value, "save_strategy", None)),
    "save_total_limit": getattr(value, "save_total_limit", None),
    "save_only_model": getattr(value, "save_only_model", None),
    "train_sampling_strategy": getattr(value, "train_sampling_strategy", None),
    "length_column_name": getattr(value, "length_column_name", None),
    "eval_strategy": enum(getattr(value, "eval_strategy", None)),
    "do_eval": getattr(value, "do_eval", None),
    "seed": getattr(value, "seed", None),
    "dataloader_drop_last": getattr(value, "dataloader_drop_last", None),
    "n_gpu": getattr(value, "_n_gpu", None),
    "world_size": nproc(value),
}
print(json.dumps(attrs, default=str))
'''
    python = str(LLAMAFACTORY_PYTHON) if LLAMAFACTORY_PYTHON.is_file() else "python"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run(
            [python, "-c", helper, str(path)],
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if not isinstance(payload, dict):
            return None, "TRAINING_ARGS_INVALID"
        return payload, None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as exc:
        return None, f"TRAINING_ARGS_UNREADABLE:{exc}"


def _checkpoint_files(path: Path) -> dict[str, Path]:
    return {
        "adapter_config": path / "adapter_config.json",
        "adapter_weights": _adapter_weights(path) or path / "adapter_model.safetensors",
        "optimizer": path / "optimizer.pt",
        "scheduler": path / "scheduler.pt",
        "rng": path / "rng_state.pth",
        "trainer_state": path / "trainer_state.json",
        "training_args": path / "training_args.bin",
    }


def _float_equal(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == expected


def _checkpoint_gate(
    epoch: int,
    *,
    require_provenance: bool = False,
    output: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if output is None:
        output = TRAIN_OUTPUT
    blockers: list[str] = []
    step = EXPECTED_CHECKPOINT_STEPS[epoch]
    checkpoint = output / f"checkpoint-{step}"
    checks: dict[str, Any] = {
        "epoch": epoch,
        "expected_step": step,
        "expected_max_steps": EXPECTED_MAX_STEPS,
        "checkpoint": str(checkpoint),
        "exists": checkpoint.is_dir(),
    }
    if not checkpoint.is_dir():
        return [f"WAIT_CHECKPOINT_EP{epoch}"], checks
    files = _checkpoint_files(checkpoint)
    checks["files"] = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
        }
        for name, path in files.items()
    }
    missing = [name for name, path in files.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        blockers.append(f"CHECKPOINT_EP{epoch}_FULL_STATE_MISSING:{','.join(missing)}")
        return blockers, checks
    try:
        state = _json(files["trainer_state"])
        adapter_config = _json(files["adapter_config"])
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"CHECKPOINT_EP{epoch}_METADATA_INVALID")
        checks["error"] = str(exc)
        return blockers, checks
    checks["trainer_state"] = {
        key: state.get(key)
        for key in ("epoch", "global_step", "max_steps", "num_train_epochs", "train_batch_size")
    }
    state_ok = (
        state.get("global_step") == step
        and _float_equal(state.get("epoch"), float(epoch))
        and state.get("max_steps") == EXPECTED_MAX_STEPS
        and state.get("num_train_epochs") == EXPECTED_EPOCHS
        and state.get("train_batch_size") == EXPECTED_MICRO_BATCH
    )
    if not state_ok:
        blockers.append(f"CHECKPOINT_EP{epoch}_TRAINER_STATE_DRIFT")
    adapter_ok = (
        adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("r") == 64
        and adapter_config.get("lora_alpha") == 128
        and _float_equal(adapter_config.get("lora_dropout"), 0.0)
        and adapter_config.get("base_model_name_or_path") == "Qwen/Qwen3.5-0.8B"
    )
    checks["adapter_config"] = {
        key: adapter_config.get(key)
        for key in ("peft_type", "r", "lora_alpha", "lora_dropout", "base_model_name_or_path")
    }
    if not adapter_ok:
        blockers.append(f"CHECKPOINT_EP{epoch}_ADAPTER_CONFIG_DRIFT")
    args_payload, args_error = _read_training_args(files["training_args"])
    checks["training_args"] = args_payload
    if args_error:
        checks["training_args_error"] = args_error
        blockers.append(f"CHECKPOINT_EP{epoch}_{args_error.split(':', 1)[0]}")
    else:
        expected_args = {
            "output_dir": _relative_save_path(output),
            "per_device_train_batch_size": EXPECTED_MICRO_BATCH,
            "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
            "num_train_epochs": float(EXPECTED_EPOCHS),
            "max_steps": -1,
            "learning_rate": 1.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "save_strategy": "epoch",
            "save_total_limit": 3,
            "save_only_model": False,
            "train_sampling_strategy": "random",
            "length_column_name": "length",
            "eval_strategy": "no",
            "do_eval": False,
            "seed": 42,
            "dataloader_drop_last": False,
            "n_gpu": 1,
            "world_size": 1,
        }
        checks["training_args_expected"] = expected_args
        for key, expected_value in expected_args.items():
            actual_value = args_payload.get(key)
            if isinstance(expected_value, float):
                ok = _float_equal(actual_value, expected_value)
            else:
                ok = actual_value == expected_value
            if not ok:
                blockers.append(f"CHECKPOINT_EP{epoch}_TRAINING_ARGS_DRIFT:{key}")

    if require_provenance:
        provenance = output.parent / f"{output.name}.run_manifest.json"
        checks["run_manifest"] = str(provenance)
        if not provenance.is_file():
            blockers.append(f"CHECKPOINT_EP{epoch}_RUN_PROVENANCE_MISSING")
        else:
            try:
                run_manifest = _json(provenance)
            except (OSError, json.JSONDecodeError):
                run_manifest = None
            checks["run_manifest_payload"] = run_manifest
            if not isinstance(run_manifest, dict) or run_manifest.get("status") not in {"running", "complete"}:
                blockers.append(f"CHECKPOINT_EP{epoch}_RUN_PROVENANCE_INVALID")
            else:
                expected_provenance = {
                    "config_sha256": _sha256(TRAIN_CONFIG) if TRAIN_CONFIG.is_file() else None,
                    "data_manifest_sha256": _sha256(DATA_MANIFEST) if DATA_MANIFEST.is_file() else None,
                    "dataset": DATASET_NAME,
                    "rows": EXPECTED_ROWS,
                    "dataset_sha256": _manifest_dataset_hash(),
                    "ordered_id_sha256": _manifest_ordered_id_hash(),
                    "parent_adapter_sha256": EXPECTED_PARENT_ADAPTER_SHA256,
                    "gpus": [EXPECTED_GPU],
                    "global_batch_size": EXPECTED_GLOBAL_BATCH,
                }
                for key, expected_value in expected_provenance.items():
                    if run_manifest.get(key) != expected_value:
                        blockers.append(f"CHECKPOINT_EP{epoch}_RUN_PROVENANCE_DRIFT:{key}")
    checks["eligible"] = not blockers
    return blockers, checks


def _manifest_dataset_hash() -> str | None:
    if not DATA_MANIFEST.is_file():
        return None
    try:
        manifest = _json(DATA_MANIFEST)
    except (OSError, json.JSONDecodeError):
        return None
    return _manifest_value(manifest, "sha256", "content_sha256", "data_sha256")


def _manifest_ordered_id_hash() -> str | None:
    if not DATA_MANIFEST.is_file():
        return None
    try:
        manifest = _json(DATA_MANIFEST)
    except (OSError, json.JSONDecodeError):
        return None
    return _manifest_value(manifest, "ordered_id_sha256", "ordered_ids_sha256")


def _run_provenance_path(output: Path | None = None) -> Path:
    if output is None:
        output = TRAIN_OUTPUT
    return output.parent / f"{output.name}.run_manifest.json"


def _run_provenance_payload(data_checks: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "PT-exp2-mm-rowbal-cont3",
        "config": str(TRAIN_CONFIG.resolve()),
        "config_sha256": _sha256(TRAIN_CONFIG),
        "dataset": DATASET_NAME,
        "data_manifest": str(DATA_MANIFEST.resolve()),
        "data_manifest_sha256": data_checks.get("manifest_sha256"),
        "dataset_sha256": data_checks.get("content_sha256_actual"),
        "ordered_id_sha256": data_checks.get("ordered_id_sha256_actual"),
        "parent_adapter": str(EXPECTED_PARENT_ADAPTER.resolve()),
        "parent_adapter_sha256": EXPECTED_PARENT_ADAPTER_SHA256,
        "rows": EXPECTED_ROWS,
        "global_batch_size": EXPECTED_GLOBAL_BATCH,
        "gpus": [EXPECTED_GPU],
        "checkpoint_steps": [EXPECTED_CHECKPOINT_STEPS[epoch] for epoch in (1, 2, 3)],
        "optimizer_scheduler_rng_required": True,
    }


def _training_root_gate(output: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    if output is None:
        output = TRAIN_OUTPUT
    blockers: list[str] = []
    checks: dict[str, Any] = {"output": str(output), "exists": output.is_dir()}
    if not output.is_dir():
        return ["WAIT_ROWBAL_TRAIN_OUTPUT"], checks
    required_root = {
        "adapter_config": output / "adapter_config.json",
        "adapter_weights": _adapter_weights(output) or output / "adapter_model.safetensors",
        "trainer_state": output / "trainer_state.json",
        "train_results": output / "train_results.json",
    }
    checks["root_files"] = {
        key: {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None}
        for key, path in required_root.items()
    }
    missing = [key for key, path in required_root.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        blockers.append(f"ROWBAL_TRAIN_ROOT_INCOMPLETE:{','.join(missing)}")
    checkpoint_names = sorted(path.name for path in output.glob("checkpoint-*") if path.is_dir())
    expected_names = sorted(f"checkpoint-{EXPECTED_CHECKPOINT_STEPS[epoch]}" for epoch in (1, 2, 3))
    checks["checkpoint_names"] = checkpoint_names
    checks["expected_checkpoint_names"] = expected_names
    if checkpoint_names != expected_names:
        blockers.append("ROWBAL_CHECKPOINT_SET_DRIFT")
    if (output / "trainer_state.json").is_file():
        try:
            state = _json(output / "trainer_state.json")
        except (OSError, json.JSONDecodeError) as exc:
            state = None
            checks["root_state_error"] = str(exc)
        checks["root_trainer_state"] = (
            {key: state.get(key) for key in ("epoch", "global_step", "max_steps", "num_train_epochs", "train_batch_size")}
            if isinstance(state, dict)
            else None
        )
        if not isinstance(state, dict) or not (
            state.get("global_step") == EXPECTED_MAX_STEPS
            and _float_equal(state.get("epoch"), float(EXPECTED_EPOCHS))
            and state.get("max_steps") == EXPECTED_MAX_STEPS
            and state.get("num_train_epochs") == EXPECTED_EPOCHS
            and state.get("train_batch_size") == EXPECTED_MICRO_BATCH
        ):
            blockers.append("ROWBAL_ROOT_TRAINER_STATE_DRIFT")
    return blockers, checks


def _posttrain_gate() -> tuple[list[str], dict[str, Any]]:
    data_blockers, data_checks = _validate_data()
    cache_checks = _cache_check(TRAIN_CONFIG, require_binding=True)
    root_blockers, root_checks = _training_root_gate()
    blockers = [*data_blockers, *root_blockers]
    if not cache_checks.get("eligible"):
        blockers.append(f"POSTTRAIN_CACHE_BINDING_FAILED:{cache_checks.get('error', 'UNKNOWN')}")
    checks: dict[str, Any] = {"data": data_checks, "cache": cache_checks, "root": root_checks}
    checkpoint_checks: dict[str, Any] = {}
    for epoch in (1, 2, 3):
        checkpoint_blockers, checkpoint = _checkpoint_gate(epoch, require_provenance=True)
        blockers.extend(checkpoint_blockers)
        checkpoint_checks[f"ep{epoch}"] = checkpoint
    checks["checkpoints"] = checkpoint_checks
    checks["eligible"] = not blockers
    return blockers, checks


def _check_parent_adapter() -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "path": str(EXPECTED_PARENT_ADAPTER),
        "expected_weight_sha256": EXPECTED_PARENT_ADAPTER_SHA256,
        "ready": _adapter_ready(EXPECTED_PARENT_ADAPTER, require_training_results=True),
    }
    if not checks["ready"]:
        blockers.append("WAIT_TEXT8M_PARENT_ADAPTER")
        return blockers, checks
    weights = _adapter_weights(EXPECTED_PARENT_ADAPTER)
    checks["weight"] = str(weights)
    checks["weight_sha256"] = _sha256(weights) if weights else None
    if checks["weight_sha256"] != EXPECTED_PARENT_ADAPTER_SHA256:
        blockers.append("TEXT8M_PARENT_ADAPTER_HASH_DRIFT")
    return blockers, checks


def _check_train(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    data_blockers, data_checks = _validate_data()
    config_blockers, config_checks = _validate_train_config(TRAIN_CONFIG)
    cache_checks = _cache_check(TRAIN_CONFIG, require_binding=True)
    gpu_blockers, gpu_checks = _gpu_gate()
    blockers = [*data_blockers, *config_blockers, *gpu_blockers]
    checks: dict[str, Any] = {
        "data": data_checks,
        "config": config_checks,
        "cache": cache_checks,
        "gpu": gpu_checks,
        "resume_epoch": args.resume_epoch,
    }
    if not cache_checks.get("eligible"):
        blockers.append(f"WAIT_ROWBAL_TOKENIZED_CACHE:{cache_checks.get('error', 'UNKNOWN')}")
    parent_blockers, parent_checks = _check_parent_adapter()
    blockers.extend(parent_blockers)
    checks["parent_adapter"] = parent_checks

    provenance = _run_provenance_path()
    checks["run_provenance"] = str(provenance)
    if args.resume_epoch is None:
        checks["output"] = str(TRAIN_OUTPUT)
        checks["output_exists"] = TRAIN_OUTPUT.exists() or TRAIN_OUTPUT.is_symlink()
        if checks["output_exists"]:
            blockers.append("TRAIN_OUTPUT_MUST_BE_ABSENT")
        if provenance.exists():
            blockers.append("STALE_RUN_PROVENANCE_REVIEW_REQUIRED")
    else:
        if args.resume_epoch not in (1, 2):
            blockers.append("RESUME_ONLY_ACCEPTS_EPOCH1_OR_EPOCH2")
        checks["output"] = str(TRAIN_OUTPUT)
        checks["output_exists"] = TRAIN_OUTPUT.is_dir()
        if not checks["output_exists"]:
            blockers.append("RESUME_OUTPUT_MISSING")
        else:
            root_blockers, root_checks = _training_root_gate(TRAIN_OUTPUT)
            # A partial output is expected for resume; only a complete root is
            # disallowed.  Keep state details in the status payload.
            checks["resume_root"] = root_checks
            if not root_blockers and root_checks.get("checkpoint_names") == sorted(
                f"checkpoint-{EXPECTED_CHECKPOINT_STEPS[epoch]}" for epoch in (1, 2, 3)
            ):
                blockers.append("TRAIN_ALREADY_COMPLETE")
            if args.resume_epoch in (1, 2):
                resume_blockers, resume_checks = _checkpoint_gate(
                    args.resume_epoch, require_provenance=True, output=TRAIN_OUTPUT
                )
                blockers.extend(resume_blockers)
                checks["resume_checkpoint"] = resume_checks
    # A stale cache directory must never be repaired by the training process.
    checks["eligible"] = not blockers
    return blockers, checks


def _validate_val_dataset() -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "path": str(VAL_DATASET),
        "expected_rows": EXPECTED_VAL_SAMPLES,
        "expected_sha256": EXPECTED_VAL_SHA256,
    }
    if not VAL_DATASET.is_file():
        return ["WAIT_PT_VAL512_DATASET"], checks
    checks["sha256"] = _sha256(VAL_DATASET)
    if checks["sha256"] != EXPECTED_VAL_SHA256:
        blockers.append("PT_VAL512_DATASET_HASH_DRIFT")
    try:
        rows = _json(VAL_DATASET)
    except (OSError, json.JSONDecodeError) as exc:
        return [*blockers, "PT_VAL512_DATASET_INVALID"], {**checks, "error": str(exc)}
    if not isinstance(rows, list) or len(rows) != EXPECTED_VAL_SAMPLES:
        blockers.append("PT_VAL512_ROW_COUNT_MISMATCH")
        return blockers, {**checks, "rows": len(rows) if isinstance(rows, list) else None}
    ids: set[str] = set()
    protocol_ok = True
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            protocol_ok = False
            break
        sample_id = row.get("id")
        images = row.get("images")
        messages = row.get("messages")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            protocol_ok = False
            break
        ids.add(sample_id)
        if not isinstance(images, list) or len(images) != 1 or not all(isinstance(v, str) for v in images):
            protocol_ok = False
            break
        if not isinstance(messages, list) or len(messages) != 3:
            protocol_ok = False
            break
        roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
        if roles != EXPECTED_ROLES:
            protocol_ok = False
            break
        user = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(user, str) or "<image>" not in user or "Caption:\n\n\nInventory of parts:" not in user:
            protocol_ok = False
            break
    checks.update({"rows": len(rows), "unique_ids": len(ids), "protocol_ok": protocol_ok})
    if not protocol_ok:
        blockers.append("PT_VAL512_IMAGE_EMPTY_CAPTION_INVENTORY_PROTOCOL_FAILED")
    registry_ok = False
    try:
        registry = _json(VAL_REGISTRY)
        entry = registry.get(VAL_REGISTRY_NAME, {}) if isinstance(registry, dict) else {}
        registry_ok = (
            entry.get("file_name") == "BrickNet-MM_PT_VAL.json"
            and entry.get("formatting") == "sharegpt"
            and entry.get("columns") == {"messages": "messages", "images": "images"}
        )
    except (OSError, json.JSONDecodeError):
        registry_ok = False
    checks["registry_ok"] = registry_ok
    if not registry_ok:
        blockers.append("PT_VAL512_REGISTRY_BINDING_FAILED")
    return blockers, checks


def _validate_predict_config(run: PredictionRun) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    checks: dict[str, Any] = {"config": str(run.config), "exists": run.config.is_file()}
    if not run.config.is_file():
        return ["PREDICT_CONFIG_MISSING"], checks
    try:
        config = _config_mapping(run.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ["PREDICT_CONFIG_INVALID"], {**checks, "error": str(exc)}
    expected = {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "adapter_name_or_path": _relative_save_path(run.checkpoint),
        "eval_dataset": VAL_REGISTRY_NAME,
        "dataset_dir": "data",
        "media_dir": "data",
        "template": "qwen3_5_nothink",
        "enable_thinking": False,
        "cutoff_len": 4096,
        "per_device_eval_batch_size": 1,
        "max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "seed": 42,
        "output_dir": _relative_save_path(run.output),
    }
    actual = {key: config.get(key) for key in expected}
    checks.update({"expected": expected, "actual": actual})
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            blockers.append(f"PREDICT_CONFIG_DRIFT:{key}")
    if config.get("stage") != "sft" or config.get("do_predict") is not True or config.get("predict_with_generate") is not True:
        blockers.append("PREDICT_CONFIG_MODE_DRIFT")
    checks["eligible"] = not blockers
    return blockers, checks


def _check_predict(run_name: str) -> tuple[list[str], dict[str, Any]]:
    run = PREDICTION_RUNS[run_name]
    posttrain_blockers, posttrain_checks = _posttrain_gate()
    val_blockers, val_checks = _validate_val_dataset()
    config_blockers, config_checks = _validate_predict_config(run)
    gpu_blockers, gpu_checks = _gpu_gate()
    blockers = [*posttrain_blockers, *val_blockers, *config_blockers, *gpu_blockers]
    checks: dict[str, Any] = {
        "run": run_name,
        "checkpoint": str(run.checkpoint),
        "posttrain": posttrain_checks,
        "val_dataset": val_checks,
        "config": config_checks,
        "gpu": gpu_checks,
    }
    output = run.output
    checks["output"] = str(output)
    checks["output_exists"] = output.exists() or output.is_symlink()
    if checks["output_exists"]:
        blockers.append("PREDICTION_OUTPUT_MUST_BE_ABSENT")
    predictions = output / "generated_predictions.jsonl"
    checks["prediction_file"] = str(predictions)
    if predictions.exists():
        blockers.append("PREDICTION_ARTIFACT_ALREADY_EXISTS")
    checks["eligible"] = not blockers
    return blockers, checks


def _prediction_dir(run_name: str) -> Path:
    return PREDICTION_RUNS[run_name].output


def _metrics_path(run_name: str) -> Path:
    return BRICKNET_ROOT / "outputs_val/qwen35_08b" / PREDICTION_RUNS[run_name].output_name / "metrics.json"


def _evaluation_dir(run_name: str) -> Path:
    return _metrics_path(run_name).parent


def _alignment_input_path(run_name: str) -> Path:
    return _evaluation_dir(run_name) / "alignment_input.jsonl"


def _alignment_path(run_name: str) -> Path:
    return _evaluation_dir(run_name) / "alignment.jsonl"


def _alignment_manifest_path(run_name: str) -> Path:
    return _evaluation_dir(run_name) / "alignment_manifest.json"


def _normalize_path_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _assistant_reference(row: dict[str, Any], index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"alignment dataset row {index}: messages must be a list")
    references = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(references) != 1 or not isinstance(references[0], str):
        raise ValueError(f"alignment dataset row {index}: expected exactly one assistant reference")
    return references[0]


def _build_alignment_rows(
    prediction_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]] | None = None,
    *,
    expected_samples: int = EXPECTED_VAL_SAMPLES,
) -> list[dict[str, str]]:
    if len(prediction_rows) != expected_samples:
        raise ValueError(f"expected {expected_samples} prediction rows, found {len(prediction_rows)}")
    if len(dataset_rows) != expected_samples:
        raise ValueError(f"expected {expected_samples} alignment dataset rows, found {len(dataset_rows)}")
    if scored_rows is not None and len(scored_rows) != expected_samples:
        raise ValueError(f"expected {expected_samples} scored rows, found {len(scored_rows)}")
    sample_ids: set[str] = set()
    alignment_rows: list[dict[str, str]] = []
    for index, (prediction, dataset_row) in enumerate(zip(prediction_rows, dataset_rows)):
        sample_id = dataset_row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(f"alignment dataset row {index}: missing or duplicate id {sample_id!r}")
        sample_ids.add(sample_id)
        response = prediction.get("predict")
        label = prediction.get("label")
        if not isinstance(response, str) or not isinstance(label, str):
            raise ValueError(f"prediction row {index}: predict and label must both be strings")
        reference = _assistant_reference(dataset_row, index)
        if _normalize_path_text(label) != _normalize_path_text(reference):
            raise ValueError(f"prediction label/reference mismatch at row {index}")
        if scored_rows is not None:
            scored = scored_rows[index]
            collisions = scored.get("collisions")
            if not isinstance(collisions, list):
                raise ValueError(f"scored row {index}: collisions must be a list")
            scored_text = scored.get("text", scored.get("path"))
            if not isinstance(scored_text, str):
                raise ValueError(f"scored row {index}: text/path must be a string")
            if _normalize_path_text(scored_text) != _normalize_path_text(response):
                raise ValueError(f"prediction/scored text mismatch at row {index}")
        alignment_rows.append({"response": response, "label": label})
    return alignment_rows


def _base_evaluation_complete(run_name: str) -> bool:
    output = _evaluation_dir(run_name)
    metrics_path = output / "metrics.json"
    manifest_path = output / "evaluation_manifest.json"
    scored_path = output / "scored.jsonl"
    if not (metrics_path.is_file() and manifest_path.is_file() and scored_path.is_file()):
        return False
    try:
        metrics = _json(metrics_path)
        manifest = _json(manifest_path)
        scored = _jsonl(scored_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("samples") == EXPECTED_VAL_SAMPLES
        and metrics.get("artifacts", {}).get("predictions") == EXPECTED_VAL_SAMPLES
        and metrics.get("structure", {}).get("samples") == EXPECTED_VAL_SAMPLES
        and len(scored) == EXPECTED_VAL_SAMPLES
    )


def _alignment_metrics_complete(metrics: dict[str, Any]) -> bool:
    task_alignment = metrics.get("task_alignment", {})
    condition_generation = metrics.get("condition_generation", {})
    dense_reward = task_alignment.get("dense_reward_mean")
    strict_success = task_alignment.get("strict_success")
    strict_success_rate = task_alignment.get("strict_success_rate")
    return bool(
        task_alignment.get("samples") == EXPECTED_VAL_SAMPLES
        and condition_generation.get("samples") == EXPECTED_VAL_SAMPLES
        and isinstance(dense_reward, (int, float))
        and not isinstance(dense_reward, bool)
        and isinstance(strict_success, int)
        and not isinstance(strict_success, bool)
        and 0 <= strict_success <= EXPECTED_VAL_SAMPLES
        and isinstance(strict_success_rate, (int, float))
        and not isinstance(strict_success_rate, bool)
        and 0.0 <= strict_success_rate <= 1.0
        and task_alignment.get("weights") == REWARD_WEIGHTS
        and task_alignment.get("pose_tolerances")
        == {
            "translation": POSE_TRANSLATION_TOLERANCE,
            "rotation_degrees": POSE_ROTATION_TOLERANCE,
            "success_threshold": POSE_SUCCESS_THRESHOLD,
        }
        and condition_generation.get("dense_reward") == dense_reward
        and condition_generation.get("strict_success_num") == strict_success
        and condition_generation.get("strict_success_rate") == strict_success_rate
    )


def _alignment_identity(run_name: str) -> dict[str, Any] | None:
    output = _evaluation_dir(run_name)
    predictions = _prediction_dir(run_name) / "generated_predictions.jsonl"
    scored = output / "scored.jsonl"
    base_manifest = output / "evaluation_manifest.json"
    required = (predictions, scored, base_manifest, ALIGNMENT_DATASET, MS_SWIFT_EVALUATOR)
    if not all(path.is_file() for path in required):
        return None
    return {
        "run": run_name,
        "expected_samples": EXPECTED_VAL_SAMPLES,
        "predictions": str(predictions.resolve()),
        "predictions_sha256": _sha256(predictions),
        "scored": str(scored.resolve()),
        "scored_sha256": _sha256(scored),
        "alignment_dataset": str(ALIGNMENT_DATASET.resolve()),
        "alignment_dataset_sha256": _sha256(ALIGNMENT_DATASET),
        "alignment_evaluator": str(MS_SWIFT_EVALUATOR.resolve()),
        "alignment_evaluator_sha256": _sha256(MS_SWIFT_EVALUATOR),
        "base_evaluation_manifest": str(base_manifest.resolve()),
        "base_evaluation_manifest_sha256": _sha256(base_manifest),
        "pose_tolerances": {
            "translation": POSE_TRANSLATION_TOLERANCE,
            "rotation_degrees": POSE_ROTATION_TOLERANCE,
            "success_threshold": POSE_SUCCESS_THRESHOLD,
        },
        "reward_weights": REWARD_WEIGHTS,
    }


def _alignment_complete(run_name: str) -> bool:
    output = _evaluation_dir(run_name)
    metrics_path = output / "metrics.json"
    metrics_md = output / "metrics.md"
    alignment_input = _alignment_input_path(run_name)
    alignment = _alignment_path(run_name)
    manifest_path = _alignment_manifest_path(run_name)
    required = (metrics_path, metrics_md, alignment_input, alignment, manifest_path)
    if not all(path.is_file() for path in required):
        return False
    identity = _alignment_identity(run_name)
    if identity is None:
        return False
    try:
        metrics = _json(metrics_path)
        manifest = _json(manifest_path)
        alignment_rows = _jsonl(alignment)
        input_rows = _jsonl(alignment_input)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    artifacts = manifest.get("artifacts", {})
    return bool(
        manifest.get("schema_version") == ALIGNMENT_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and manifest.get("identity") == identity
        and _alignment_metrics_complete(metrics)
        and len(alignment_rows) == EXPECTED_VAL_SAMPLES
        and len(input_rows) == EXPECTED_VAL_SAMPLES
        and artifacts.get("alignment_input_sha256") == _sha256(alignment_input)
        and artifacts.get("alignment_sha256") == _sha256(alignment)
        and artifacts.get("metrics_json_sha256") == _sha256(metrics_path)
        and artifacts.get("metrics_md_sha256") == _sha256(metrics_md)
    )


def _evaluate(run_name: str, args: argparse.Namespace) -> None:
    run = PREDICTION_RUNS[run_name]
    predictions = _prediction_dir(run_name) / "generated_predictions.jsonl"
    text_metrics = _prediction_dir(run_name) / "predict_results.json"
    output = _evaluation_dir(run_name)
    scored = output / "scored.jsonl"
    metrics_json = output / "metrics.json"
    metrics_md = output / "metrics.md"
    alignment_input = _alignment_input_path(run_name)
    alignment = _alignment_path(run_name)
    alignment_manifest = _alignment_manifest_path(run_name)

    posttrain_blockers, posttrain_checks = _posttrain_gate()
    val_blockers, val_checks = _validate_val_dataset()
    config_blockers, config_checks = _validate_predict_config(run)
    gpu_blockers, gpu_checks = _gpu_gate()
    blockers = [*posttrain_blockers, *val_blockers, *config_blockers, *gpu_blockers]
    checks: dict[str, Any] = {
        "run": run_name,
        "predictions": str(predictions),
        "text_metrics": str(text_metrics),
        "output": str(output),
        "posttrain": posttrain_checks,
        "val_dataset": val_checks,
        "config": config_checks,
        "gpu": gpu_checks,
        "base_evaluation_complete": _base_evaluation_complete(run_name),
        "alignment_complete": _alignment_complete(run_name),
    }
    if not predictions.is_file():
        blockers.append("WAIT_PT_ROWBAL_PREDICTIONS")
    if not text_metrics.is_file():
        blockers.append("WAIT_PT_ROWBAL_PREDICT_RESULTS")
    if checks["alignment_complete"]:
        blockers.append("EVALUATION_ALREADY_COMPLETE")
    if predictions.is_file() and ALIGNMENT_DATASET.is_file():
        try:
            prediction_rows = _jsonl(predictions)
            dataset_rows = _jsonl(ALIGNMENT_DATASET)
            _build_alignment_rows(prediction_rows, dataset_rows)
            checks["prediction_reference_preflight"] = {
                "eligible": True,
                "samples": len(prediction_rows),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checks["prediction_reference_preflight"] = {"eligible": False, "error": str(exc)}
            blockers.append("PT_EXP2_MM_ALIGNMENT_PREFLIGHT_FAILED")
    base_command = [
        str(BRICKNET_PYTHON),
        str(EVALUATOR),
        "--predictions",
        str(predictions),
        "--text-metrics",
        str(text_metrics),
        "--input-format",
        "llamafactory",
        "--output-dir",
        str(output),
    ]
    alignment_command = [
        str(BRICKNET_PYTHON),
        str(MS_SWIFT_EVALUATOR),
        "alignment-worker",
        "--results",
        str(alignment_input),
        "--dataset",
        str(ALIGNMENT_DATASET),
        "--scored",
        str(scored),
        "--metrics-json",
        str(metrics_json),
        "--metrics-md",
        str(metrics_md),
        "--output",
        str(alignment),
        "--bricknet-root",
        str(BRICKNET_ROOT),
        "--translation-tolerance",
        str(POSE_TRANSLATION_TOLERANCE),
        "--rotation-tolerance",
        str(POSE_ROTATION_TOLERANCE),
        "--pose-success-threshold",
        str(POSE_SUCCESS_THRESHOLD),
    ]
    payload = {
        "action": "evaluate",
        "run": run_name,
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "commands": {
            "base_evaluation": None if checks["base_evaluation_complete"] else shlex.join(base_command),
            "alignment": shlex.join(alignment_command),
        },
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM rowbal evaluation blocked; resolve the reported gates first")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src")
    if not checks["base_evaluation_complete"]:
        subprocess.run(base_command, cwd=BRICKNET_ROOT, env=env, check=True)
    if not _base_evaluation_complete(run_name):
        raise RuntimeError("base BrickNet evaluation did not produce complete 512-row artifacts")
    prediction_rows = _jsonl(predictions)
    dataset_rows = _jsonl(ALIGNMENT_DATASET)
    scored_rows = _jsonl(scored)
    alignment_rows = _build_alignment_rows(prediction_rows, dataset_rows, scored_rows)
    _write_jsonl(alignment_input, alignment_rows)
    metrics_before_alignment_sha256 = _sha256(metrics_json)
    subprocess.run(alignment_command, cwd=MS_SWIFT_ROOT, env=env, check=True)
    if not (metrics_json.is_file() and metrics_md.is_file() and alignment.is_file()):
        raise RuntimeError("alignment worker did not produce all required artifacts")
    metrics = _json(metrics_json)
    if not _alignment_metrics_complete(metrics):
        raise RuntimeError("alignment worker did not add complete 512-row task metrics")
    alignment_output_rows = _jsonl(alignment)
    if len(alignment_output_rows) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError(
            f"alignment worker produced {len(alignment_output_rows)} rows, expected {EXPECTED_VAL_SAMPLES}"
        )
    identity = _alignment_identity(run_name)
    if identity is None:
        raise RuntimeError("alignment identity inputs disappeared after evaluation")
    _write_json(
        alignment_manifest,
        {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "identity": identity,
            "artifacts": {
                "alignment_input": str(alignment_input.resolve()),
                "alignment_input_sha256": _sha256(alignment_input),
                "alignment": str(alignment.resolve()),
                "alignment_sha256": _sha256(alignment),
                "metrics_json": str(metrics_json.resolve()),
                "metrics_json_sha256": _sha256(metrics_json),
                "metrics_json_before_alignment_sha256": metrics_before_alignment_sha256,
                "metrics_md": str(metrics_md.resolve()),
                "metrics_md_sha256": _sha256(metrics_md),
            },
            "summary": {
                "task_alignment_samples": metrics["task_alignment"]["samples"],
                "condition_generation_samples": metrics["condition_generation"]["samples"],
                "dense_reward_mean": metrics["task_alignment"]["dense_reward_mean"],
                "strict_success": metrics["task_alignment"]["strict_success"],
                "strict_success_rate": metrics["task_alignment"]["strict_success_rate"],
            },
        },
    )
    if not _alignment_complete(run_name):
        raise RuntimeError("alignment manifest failed its post-write completion gate")
    print(f"Complete PT-exp2 MM rowbal evaluation: {metrics_json}")


def _train_command(args: argparse.Namespace) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(TRAIN_CONFIG),
    ]
    if args.resume_epoch in (1, 2):
        command.append(f"resume_from_checkpoint={TRAIN_OUTPUT / f'checkpoint-{EXPECTED_CHECKPOINT_STEPS[args.resume_epoch]}'}")
    return command


def _single_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = EXPECTED_GPU
    # A stale distributed environment could turn this one-GPU run into an
    # unintended torchrun job.  The launcher owns these values.
    for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        env.pop(key, None)
    return env


def _run_train(args: argparse.Namespace) -> None:
    blockers, checks = _check_train(args)
    command = _train_command(args)
    payload = {
        "action": "train",
        "experiment": "PT-exp2-mm-rowbal-cont3",
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "command": shlex.join(command),
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM rowbal training blocked; resolve the reported gates first")
    provenance = _run_provenance_path()
    if args.resume_epoch is None:
        data_checks = checks["data"]
        _write_json(provenance, _run_provenance_payload(data_checks, status="running"))
    env = _single_gpu_env()
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    except BaseException:
        # Keep the running provenance marker so a later explicit resume can
        # prove which config/data/parent/world-size produced the checkpoint.
        raise
    posttrain_blockers, posttrain_checks = _posttrain_gate()
    if posttrain_blockers:
        raise RuntimeError(f"training exited successfully but post-training gate failed: {posttrain_checks}")
    # A successful explicit resume completes the same run manifest as the
    # initial invocation.  Without this write, all three validated
    # checkpoints could exist while the provenance remained permanently
    # ``running``.
    data_checks = checks["data"]
    _write_json(provenance, _run_provenance_payload(data_checks, status="complete"))
    print(json.dumps({"event": "rowbal_train_complete", "checks": posttrain_checks, "executed": True}, ensure_ascii=False, indent=2))


def _run_predict(run_name: str, args: argparse.Namespace) -> None:
    run = PREDICTION_RUNS[run_name]
    blockers, checks = _check_predict(run_name)
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(run.config),
    ]
    payload = {
        "action": "predict",
        "run": run_name,
        "experiment": "PT-exp2-mm-rowbal-cont3",
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "command": shlex.join(command),
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM rowbal prediction blocked; resolve the reported gates first")
    subprocess.run(command, cwd=ROOT, env=_single_gpu_env(), check=True)
    predictions = run.output / "generated_predictions.jsonl"
    if not predictions.is_file():
        raise RuntimeError("prediction exited successfully but generated_predictions.jsonl is missing")
    rows = _jsonl(predictions)
    if len(rows) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError(f"prediction produced {len(rows)} rows, expected {EXPECTED_VAL_SAMPLES}")
    print(json.dumps({"event": "rowbal_prediction_complete", "run": run_name, "rows": len(rows), "executed": True}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("prepare-audits", "prepare-cache", "train", "predict", "evaluate"),
        required=True,
    )
    parser.add_argument("--run", choices=RUN_NAMES, default="ep1", help="prediction/evaluation epoch")
    parser.add_argument("--resume-epoch", type=int, choices=(1, 2), help="resume train from validated epoch-1/2 checkpoint")
    parser.add_argument("--execute", action="store_true", help="allow cache build, train, prediction, or evaluation")
    parser.add_argument("--gpus", nargs="+", metavar="GPU", help="must be exactly 1 for CUDA actions")
    parser.add_argument("--cache-batch-size", type=int, default=10_000)
    parser.add_argument("--cache-num-proc", type=int, default=1)
    parser.add_argument("--processor-workers", type=int, default=4)
    parser.add_argument("--processor-chunksize", type=int, default=8)
    args = parser.parse_args()
    if args.gpus:
        selectors = [selector for value in args.gpus for selector in value.split(",") if selector]
        if len(selectors) != len(set(selectors)):
            parser.error("--gpus contains duplicate CUDA selectors")
        args.gpus = selectors
    if args.cache_batch_size < 1 or args.cache_num_proc < 1:
        parser.error("cache batch/process counts must be positive")
    if args.processor_workers < 1 or args.processor_chunksize < 1:
        parser.error("processor worker/chunksize counts must be positive")
    if args.action in {"prepare-audits", "prepare-cache"} and args.gpus:
        parser.error(f"{args.action} is CPU-only; do not pass --gpus")
    if args.action != "train" and args.resume_epoch is not None:
        parser.error("--resume-epoch is valid only with --action train")
    return args


def main() -> None:
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    if args.action == "prepare-audits":
        _prepare_audits(args)
    elif args.action == "prepare-cache":
        _prepare_cache(args)
    elif args.action == "train":
        _run_train(args)
    elif args.action == "predict":
        _run_predict(args.run, args)
    else:
        _evaluate(args.run, args)


if __name__ == "__main__":
    main()
