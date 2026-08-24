#!/usr/bin/env python3
"""Fail-closed launcher for the isolated PT-exp2 MM v2-no-meta chain.

This entry point intentionally does not mutate or redirect the historical
``launch_bricknet_pt_exp2.py`` runs.  It accepts only mm-e1/e2/e3 and binds
each training run to the immutable v2 projection, its own processor audit,
cache, adapter, prediction, evaluation, and final alias namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/data/jiahao/task/BrickNet")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
MS_SWIFT_ROOT = Path("/data/jiahao/task/ms-swift")
MS_SWIFT_EVALUATOR = (
    MS_SWIFT_ROOT / "examples/train/grpo/plugin/bricknet/evaluate_experiment.py"
)
ALIGNMENT_DATASET = (
    BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
)
CONFIG_ROOT = ROOT / "examples/train_lora"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
V1_ROOT = ROOT / "data/bricknet_pt_exp2/mm"
V2_ROOT = ROOT / "data/bricknet_pt_exp2_mm_v2"
V2_MANIFEST = V2_ROOT / "manifest.v2.json"
V2_LOADER_REPORT = V2_ROOT / "loader_validation_report.json"
V2_AUDIT_ROOT = (
    BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-PT-exp2/reports/token_audit_mm6400_v2"
)
EVALUATOR = BRICKNET_ROOT / "scripts/evaluate_experiment.py"
FINAL_ALIAS = SAVE_ROOT / "PT-exp2-v2"
SELECTION_RECORD = ROOT / "data/bricknet_pt_exp2/gates/PT-exp2-v2-selection.json"
EXPECTED_EVAL_SAMPLES = 512
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
TEXT8M_OUTPUT = (
    "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_"
    "lora64_len6401_nopack"
)
TEXT8M_ADAPTER_SHA256 = "a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd"
SUPPORTED_TRAIN_WORLD_SIZES = (1, 2)
RUN_NAMES = ("mm-e1", "mm-e2", "mm-e3")


@dataclass(frozen=True)
class Run:
    epoch: str
    train_config: str
    predict_config: str
    output: str
    prerequisite_output: str
    eval_name: str


RUNS = {
    "mm-e1": Run(
        epoch="e1",
        train_config="qwen35_08b_bricknet_pt_exp2_mm_e1_v2.yaml",
        predict_config="qwen35_08b_bricknet_pt_exp2_mm_e1_v2_predict.yaml",
        output=(
            "train_PT_exp2_mm_e1_v2_nometa_qwen35_08b_text8m_mm135k_"
            "replay1to1_ep1_bs2_gbs16_lora64_len6400"
        ),
        prerequisite_output=TEXT8M_OUTPUT,
        eval_name="eval_PT_exp2_mm_e1_v2_nometa_ptval_in4096_out4096_p95_t1_k20",
    ),
    "mm-e2": Run(
        epoch="e2",
        train_config="qwen35_08b_bricknet_pt_exp2_mm_e2_v2.yaml",
        predict_config="qwen35_08b_bricknet_pt_exp2_mm_e2_v2_predict.yaml",
        output=(
            "train_PT_exp2_mm_e2_v2_nometa_qwen35_08b_text8m_mm135k_"
            "replay1to1_ep1_bs2_gbs16_lora64_len6400"
        ),
        prerequisite_output=(
            "train_PT_exp2_mm_e1_v2_nometa_qwen35_08b_text8m_mm135k_"
            "replay1to1_ep1_bs2_gbs16_lora64_len6400"
        ),
        eval_name="eval_PT_exp2_mm_e2_v2_nometa_ptval_in4096_out4096_p95_t1_k20",
    ),
    "mm-e3": Run(
        epoch="e3",
        train_config="qwen35_08b_bricknet_pt_exp2_mm_e3_v2.yaml",
        predict_config="qwen35_08b_bricknet_pt_exp2_mm_e3_v2_predict.yaml",
        output=(
            "train_PT_exp2_mm_e3_v2_nometa_qwen35_08b_text8m_mm135k_"
            "replay1to1_ep1_bs2_gbs16_lora64_len6400"
        ),
        prerequisite_output=(
            "train_PT_exp2_mm_e2_v2_nometa_qwen35_08b_text8m_mm135k_"
            "replay1to1_ep1_bs2_gbs16_lora64_len6400"
        ),
        eval_name="eval_PT_exp2_mm_e3_v2_nometa_ptval_in4096_out4096_p95_t1_k20",
    ),
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _adapter_ready(path: Path) -> bool:
    return (
        (path / "adapter_config.json").is_file()
        and any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
        and (path / "trainer_state.json").is_file()
        and (path / "train_results.json").is_file()
    )


def _adapter_weights(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _visible_gpu_selectors() -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [item.strip() for item in visible.split(",") if item.strip()]


def _gpu_processes() -> list[str]:
    selectors = _visible_gpu_selectors()
    if not selectors:
        return []
    processes: list[str] = []
    for selector in selectors:
        command = [
            "nvidia-smi",
            "-i",
            selector,
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        processes.extend(
            f"gpu={selector}, {line.strip()}" for line in result.stdout.splitlines() if line.strip()
        )
    return processes


def _prediction_dir(run_name: str) -> Path:
    return SAVE_ROOT / RUNS[run_name].eval_name


def _metrics_path(run_name: str) -> Path:
    return BRICKNET_ROOT / "outputs_val/qwen35_08b" / RUNS[run_name].eval_name / "metrics.json"


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
        raise ValueError(
            f"alignment dataset row {index}: expected exactly one assistant reference"
        )
    return references[0]


def _build_alignment_rows(
    prediction_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]] | None = None,
    *,
    expected_samples: int = EXPECTED_EVAL_SAMPLES,
) -> list[dict[str, str]]:
    if len(prediction_rows) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} prediction rows, found {len(prediction_rows)}"
        )
    if len(dataset_rows) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} alignment dataset rows, found {len(dataset_rows)}"
        )
    if scored_rows is not None and len(scored_rows) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} scored rows, found {len(scored_rows)}"
        )

    sample_ids: set[str] = set()
    alignment_rows: list[dict[str, str]] = []
    for index, (prediction, dataset_row) in enumerate(zip(prediction_rows, dataset_rows)):
        sample_id = dataset_row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError(
                f"alignment dataset row {index}: missing or duplicate id {sample_id!r}"
            )
        sample_ids.add(sample_id)

        response = prediction.get("predict")
        label = prediction.get("label")
        if not isinstance(response, str) or not isinstance(label, str):
            raise ValueError(
                f"prediction row {index}: predict and label must both be strings"
            )
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
        and manifest.get("samples") == EXPECTED_EVAL_SAMPLES
        and metrics.get("artifacts", {}).get("predictions") == EXPECTED_EVAL_SAMPLES
        and metrics.get("structure", {}).get("samples") == EXPECTED_EVAL_SAMPLES
        and len(scored) == EXPECTED_EVAL_SAMPLES
    )


def _alignment_metrics_complete(metrics: dict[str, Any]) -> bool:
    task_alignment = metrics.get("task_alignment", {})
    condition_generation = metrics.get("condition_generation", {})
    dense_reward = task_alignment.get("dense_reward_mean")
    strict_success = task_alignment.get("strict_success")
    strict_success_rate = task_alignment.get("strict_success_rate")
    return bool(
        task_alignment.get("samples") == EXPECTED_EVAL_SAMPLES
        and condition_generation.get("samples") == EXPECTED_EVAL_SAMPLES
        and isinstance(dense_reward, (int, float))
        and not isinstance(dense_reward, bool)
        and isinstance(strict_success, int)
        and not isinstance(strict_success, bool)
        and 0 <= strict_success <= EXPECTED_EVAL_SAMPLES
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
        "expected_samples": EXPECTED_EVAL_SAMPLES,
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
        and len(alignment_rows) == EXPECTED_EVAL_SAMPLES
        and len(input_rows) == EXPECTED_EVAL_SAMPLES
        and artifacts.get("alignment_input_sha256") == _sha256(alignment_input)
        and artifacts.get("alignment_sha256") == _sha256(alignment)
        and artifacts.get("metrics_json_sha256") == _sha256(metrics_path)
        and artifacts.get("metrics_md_sha256") == _sha256(metrics_md)
    )


def _train_batch_profile(world_size: int) -> dict[str, int]:
    return {
        "per_device_batch_size": 2,
        "world_size": world_size,
        "gradient_accumulation_steps": 8 // world_size,
        "global_batch_size": 16,
    }


def _validate_v2_data(run_name: str) -> tuple[list[str], dict[str, Any]]:
    run = RUNS[run_name]
    blockers: list[str] = []
    checks: dict[str, Any] = {"manifest": str(V2_MANIFEST)}
    if not V2_MANIFEST.is_file():
        return ["WAIT_PT_EXP2_MM_V2_MANIFEST"], checks
    manifest = _json(V2_MANIFEST)
    checks["manifest_sha256"] = _sha256(V2_MANIFEST)
    checks["data_revision"] = manifest.get("data_revision")
    checks["manifest_eligible"] = manifest.get("eligible") is True
    required_gates = (
        "parent_hashes_match",
        "row_counts_match",
        "ordered_ids_match",
        "semantic_projection_match",
        "uniform_top_level_schema",
        "full_arrow_materialization",
    )
    checks["manifest_gates"] = {key: manifest.get("gates", {}).get(key) for key in required_gates}
    if not checks["manifest_eligible"] or not all(checks["manifest_gates"].values()):
        blockers.append("PT_EXP2_MM_V2_MANIFEST_NOT_ELIGIBLE")

    registry = V2_ROOT / "dataset_info.json"
    registry_expected = manifest.get("registry", {}).get("sha256")
    checks["registry"] = str(registry)
    checks["registry_sha256_expected"] = registry_expected
    checks["registry_sha256_actual"] = _sha256(registry) if registry.is_file() else None
    if checks["registry_sha256_actual"] != registry_expected:
        blockers.append("PT_EXP2_MM_V2_REGISTRY_DRIFT")

    info = manifest.get("epochs", {}).get(run.epoch, {})
    dataset = V2_ROOT / str(info.get("file", ""))
    parent_info = info.get("parent", {})
    parent = Path(str(parent_info.get("path", "")))
    checks.update(
        {
            "dataset": str(dataset),
            "dataset_rows": info.get("rows"),
            "dataset_sha256_expected": info.get("sha256"),
            "dataset_sha256_actual": _sha256(dataset) if dataset.is_file() else None,
            "ordered_id_sha256": info.get("ordered_id_sha256"),
            "semantic_projection_sha256": info.get("semantic_projection_sha256"),
            "projection_equivalent": info.get("projection_equivalent"),
            "arrow_materialization": info.get("arrow_materialization"),
            "parent": str(parent),
            "parent_sha256_expected": parent_info.get("sha256"),
            "parent_sha256_actual": _sha256(parent) if parent.is_file() else None,
        }
    )
    if checks["dataset_sha256_actual"] != checks["dataset_sha256_expected"]:
        blockers.append("PT_EXP2_MM_V2_DATASET_DRIFT")
    if checks["parent_sha256_actual"] != checks["parent_sha256_expected"]:
        blockers.append("PT_EXP2_MM_V1_PARENT_DRIFT")
    if info.get("projection_equivalent") is not True:
        blockers.append("PT_EXP2_MM_V2_PROJECTION_NOT_EQUIVALENT")
    if info.get("arrow_materialization", {}).get("eligible") is not True:
        blockers.append("PT_EXP2_MM_V2_ARROW_GATE_FAILED")

    checks["loader_validation"] = str(V2_LOADER_REPORT)
    checks["loader_validation_eligible"] = False
    if V2_LOADER_REPORT.is_file():
        loader_report = _json(V2_LOADER_REPORT)
        loader_result = next(
            (
                item
                for item in loader_report.get("results", [])
                if item.get("dataset") in (info.get("dataset"), [info.get("dataset")])
            ),
            None,
        )
        checks["loader_validation_eligible"] = bool(
            loader_report.get("eligible") is True
            and loader_report.get("model_weights_loaded") is False
            and loader_report.get("optimizer_started") is False
            and loader_report.get("v2_manifest_sha256") == checks["manifest_sha256"]
            and loader_result
            and loader_result.get("raw_full_materialization") is True
            and loader_result.get("eligible") is True
        )
    if not checks["loader_validation_eligible"]:
        blockers.append("WAIT_PT_EXP2_MM_V2_LLAMFACTORY_LOADER_GATE")

    audit_report = V2_AUDIT_ROOT / run.epoch / "BrickNet-MM-Reasoning_token_audit_report.json"
    checks["token_audit"] = str(audit_report)
    checks["token_audit_eligible"] = False
    if audit_report.is_file():
        audit = _json(audit_report)
        audit_dataset = audit.get("datasets", {}).get(info.get("dataset"), {})
        checks["token_audit_eligible"] = bool(
            audit.get("is_full_pool") is True
            and audit.get("zero_errors") is True
            and audit.get("zero_truncation") is True
            and audit.get("training_eligible") is True
            and audit_dataset.get("sha256") == info.get("sha256")
            and audit_dataset.get("count") == info.get("rows")
            and audit_dataset.get("ordered_id_sha256") == info.get("ordered_id_sha256")
        )
    if not checks["token_audit_eligible"]:
        blockers.append("WAIT_PT_EXP2_MM_V2_ZERO_TRUNCATION_AUDIT")
    return blockers, checks


def _validate_train_config(run_name: str, config_path: Path) -> tuple[list[str], dict[str, Any]]:
    run = RUNS[run_name]
    blockers: list[str] = []
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = _json(V2_MANIFEST)
    epoch_info = manifest["epochs"][run.epoch]
    expected = {
        "dataset": epoch_info["dataset"],
        "dataset_dir": "data/bricknet_pt_exp2_mm_v2",
        "media_dir": "data",
        "adapter_name_or_path": str(Path("saves/Qwen3.5-0.8B-Thinking/lora") / run.prerequisite_output),
        "output_dir": str(Path("saves/Qwen3.5-0.8B-Thinking/lora") / run.output),
        "per_device_train_batch_size": 2,
        "cutoff_len": 6400,
        "seed": 42,
    }
    actual = {key: config.get(key) for key in expected}
    if actual != expected:
        blockers.append("PT_EXP2_MM_V2_TRAIN_CONFIG_DRIFT")
    tokenized_path = str(config.get("tokenized_path", ""))
    if "v2-nometa" not in tokenized_path:
        blockers.append("PT_EXP2_MM_V2_CACHE_NAMESPACE_DRIFT")
    return blockers, {"expected": expected, "actual": actual, "tokenized_path": tokenized_path}


def _check_train(run_name: str) -> tuple[list[str], dict[str, Any]]:
    run = RUNS[run_name]
    blockers, checks = _validate_v2_data(run_name)
    config_path = CONFIG_ROOT / run.train_config
    checks["config"] = str(config_path)
    if not config_path.is_file():
        blockers.append("CONFIG_MISSING")
    else:
        config_blockers, config_checks = _validate_train_config(run_name, config_path)
        blockers.extend(config_blockers)
        checks["config_binding"] = config_checks

    selectors = _visible_gpu_selectors()
    world_size = len(selectors) if selectors is not None else None
    checks["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    checks["world_size"] = world_size
    checks["batch_profile"] = (
        _train_batch_profile(world_size) if world_size in SUPPORTED_TRAIN_WORLD_SIZES else None
    )
    if selectors is None:
        blockers.append("SET_EXPLICIT_TRAIN_GPUS")
    elif world_size not in SUPPORTED_TRAIN_WORLD_SIZES:
        blockers.append("PT_EXP2_MM_V2_REQUIRES_1_OR_2_VISIBLE_GPUS")

    prerequisite = SAVE_ROOT / run.prerequisite_output
    checks["prerequisite_adapter"] = str(prerequisite)
    checks["prerequisite_ready"] = _adapter_ready(prerequisite)
    if not checks["prerequisite_ready"]:
        blockers.append("WAIT_PREREQUISITE_ADAPTER")
    if run_name == "mm-e1" and checks["prerequisite_ready"]:
        weights = _adapter_weights(prerequisite)
        checks["text8m_adapter_sha256"] = _sha256(weights) if weights else None
        if checks["text8m_adapter_sha256"] != TEXT8M_ADAPTER_SHA256:
            blockers.append("PT_EXP2_TEXT8M_ADAPTER_DRIFT")

    output = SAVE_ROOT / run.output
    checks["output"] = str(output)
    checks["already_complete"] = _adapter_ready(output)
    if output.exists() and not checks["already_complete"]:
        blockers.append("OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED")
    if checks["already_complete"]:
        blockers.append("RUN_ALREADY_COMPLETE")
    checks["gpu_processes"] = _gpu_processes()
    # User-approved batch policy: report GPU occupancy without blocking.
    return blockers, checks


def _check_predict(run_name: str) -> tuple[list[str], dict[str, Any]]:
    run = RUNS[run_name]
    blockers: list[str] = []
    config_path = CONFIG_ROOT / run.predict_config
    checks: dict[str, Any] = {"config": str(config_path), "config_exists": config_path.is_file()}
    if not config_path.is_file():
        blockers.append("CONFIG_MISSING")
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_adapter = str(Path("saves/Qwen3.5-0.8B-Thinking/lora") / run.output)
        expected_output = str(Path("saves/Qwen3.5-0.8B-Thinking/lora") / run.eval_name)
        checks["config_binding"] = {
            "adapter": config.get("adapter_name_or_path"),
            "output": config.get("output_dir"),
            "eval_dataset": config.get("eval_dataset"),
        }
        if (
            config.get("adapter_name_or_path") != expected_adapter
            or config.get("output_dir") != expected_output
            or config.get("eval_dataset") != "BrickNet-MM-PT-VAL"
        ):
            blockers.append("PT_EXP2_MM_V2_PREDICT_CONFIG_DRIFT")
    adapter = SAVE_ROOT / run.output
    checks["adapter"] = str(adapter)
    checks["adapter_ready"] = _adapter_ready(adapter)
    if not checks["adapter_ready"]:
        blockers.append("WAIT_TRAIN_ADAPTER")
    predictions = _prediction_dir(run_name) / "generated_predictions.jsonl"
    checks["prediction_output"] = str(predictions)
    if predictions.exists():
        blockers.append("PREDICTION_OUTPUT_ALREADY_EXISTS")
    checks["gpu_processes"] = _gpu_processes()
    return blockers, checks


def _run_train_or_predict(args: argparse.Namespace) -> None:
    run = RUNS[args.run]
    if args.action == "train":
        blockers, checks = _check_train(args.run)
        config = CONFIG_ROOT / run.train_config
    else:
        blockers, checks = _check_predict(args.run)
        config = CONFIG_ROOT / run.predict_config
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(config),
    ]
    selectors = _visible_gpu_selectors()
    env = os.environ.copy()
    launch_env: dict[str, str] = {}
    if selectors:
        launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selectors)
    if args.action == "train" and selectors and len(selectors) in SUPPORTED_TRAIN_WORLD_SIZES:
        profile = _train_batch_profile(len(selectors))
        command.append(f"gradient_accumulation_steps={profile['gradient_accumulation_steps']}")
        if len(selectors) > 1:
            launch_env.update({"FORCE_TORCHRUN": "1", "NPROC_PER_NODE": str(len(selectors)), "NNODES": "1"})
            checks["launch_mode"] = "torchrun_ddp"
        else:
            for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES"):
                env.pop(key, None)
            checks["launch_mode"] = "single_process"
    env.update(launch_env)
    display = ["env", *[f"{key}={value}" for key, value in launch_env.items()], *command] if launch_env else command
    payload = {
        "action": args.action,
        "run": args.run,
        "data_revision": "v2-no-meta",
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "command": shlex.join(display),
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM v2 launch blocked; resolve the reported gates first")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _evaluate(args: argparse.Namespace) -> None:
    predictions = _prediction_dir(args.run) / "generated_predictions.jsonl"
    text_metrics = _prediction_dir(args.run) / "predict_results.json"
    output = _evaluation_dir(args.run)
    scored = output / "scored.jsonl"
    metrics_json = output / "metrics.json"
    metrics_md = output / "metrics.md"
    alignment_input = _alignment_input_path(args.run)
    alignment = _alignment_path(args.run)
    alignment_manifest = _alignment_manifest_path(args.run)
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "predictions": str(predictions),
        "text_metrics": str(text_metrics),
        "output": str(output),
        "alignment_dataset": str(ALIGNMENT_DATASET),
        "alignment_evaluator": str(MS_SWIFT_EVALUATOR),
        "base_evaluation_complete": _base_evaluation_complete(args.run),
        "alignment_complete": _alignment_complete(args.run),
    }
    if not predictions.is_file():
        blockers.append("WAIT_512_PREDICTIONS")
    if not ALIGNMENT_DATASET.is_file():
        blockers.append("WAIT_ALIGNMENT_VAL512_DATASET")
    if not MS_SWIFT_EVALUATOR.is_file():
        blockers.append("WAIT_ALIGNMENT_EVALUATOR")
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
            checks["prediction_reference_preflight"] = {
                "eligible": False,
                "error": str(exc),
            }
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
        "run": args.run,
        "data_revision": "v2-no-meta",
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "gpu_processes": _gpu_processes(),
        "commands": {
            "base_evaluation": (
                None if checks["base_evaluation_complete"] else shlex.join(base_command)
            ),
            "alignment": shlex.join(alignment_command),
        },
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 MM v2 evaluation blocked; resolve the reported gates first")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src")
    if not checks["base_evaluation_complete"]:
        subprocess.run(base_command, cwd=BRICKNET_ROOT, env=env, check=True)
    if not _base_evaluation_complete(args.run):
        raise RuntimeError("base BrickNet evaluation did not produce complete 512-row artifacts")

    prediction_rows = _jsonl(predictions)
    dataset_rows = _jsonl(ALIGNMENT_DATASET)
    scored_rows = _jsonl(scored)
    alignment_rows = _build_alignment_rows(
        prediction_rows,
        dataset_rows,
        scored_rows,
    )
    _write_jsonl(alignment_input, alignment_rows)
    metrics_before_alignment_sha256 = _sha256(metrics_json)
    subprocess.run(alignment_command, cwd=MS_SWIFT_ROOT, env=env, check=True)

    if not (metrics_json.is_file() and metrics_md.is_file() and alignment.is_file()):
        raise RuntimeError("alignment worker did not produce all required artifacts")
    metrics = _json(metrics_json)
    if not _alignment_metrics_complete(metrics):
        raise RuntimeError("alignment worker did not add complete 512-row task metrics")
    alignment_output_rows = _jsonl(alignment)
    if len(alignment_output_rows) != EXPECTED_EVAL_SAMPLES:
        raise RuntimeError(
            f"alignment worker produced {len(alignment_output_rows)} rows, "
            f"expected {EXPECTED_EVAL_SAMPLES}"
        )
    identity = _alignment_identity(args.run)
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
    if not _alignment_complete(args.run):
        raise RuntimeError("alignment manifest failed its post-write completion gate")
    print(f"Complete PT-exp2 MM evaluation: {metrics_json}")


def _selection_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["task_alignment"]["strict_success_rate"]),
        float(metrics["task_alignment"]["dense_reward_mean"]),
        float(metrics["structure"]["parsable_and_collision_free_rate"]),
        float(metrics["structure"]["fully_parsable_rate"]),
    )


def _select_final(args: argparse.Namespace) -> None:
    candidates: dict[str, Any] = {}
    blockers: list[str] = []
    for run_name in RUN_NAMES:
        metrics_path = _metrics_path(run_name)
        adapter = SAVE_ROOT / RUNS[run_name].output
        if not metrics_path.is_file():
            blockers.append(f"WAIT_{run_name.upper()}_METRICS")
            continue
        if not _adapter_ready(adapter):
            blockers.append(f"WAIT_{run_name.upper()}_ADAPTER")
            continue
        if not _alignment_complete(run_name):
            blockers.append(f"WAIT_{run_name.upper()}_TASK_ALIGNMENT")
            continue
        metrics = _json(metrics_path)
        candidates[run_name] = {
            "adapter": str(adapter),
            "metrics": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "rank": _selection_tuple(metrics),
        }
    recommended = max(candidates, key=lambda name: tuple(candidates[name]["rank"])) if candidates else None
    if FINAL_ALIAS.exists() or FINAL_ALIAS.is_symlink():
        blockers.append("PT_EXP2_V2_ALIAS_ALREADY_EXISTS")
    if SELECTION_RECORD.exists():
        blockers.append("PT_EXP2_V2_SELECTION_RECORD_ALREADY_EXISTS")
    payload = {
        "action": "select-final",
        "data_revision": "v2-no-meta",
        "ranking": "lexicographic: strict_success_rate, dense_reward_mean, clean_rate, parsable_rate",
        "candidates": candidates,
        "recommended": recommended,
        "alias": str(FINAL_ALIAS),
        "record": str(SELECTION_RECORD),
        "ready": not blockers,
        "blockers": blockers,
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if not args.approve:
        raise SystemExit("--approve is required to freeze the PT-exp2-v2 alias")
    if blockers or recommended is None:
        raise SystemExit("PT-exp2 MM v2 final selection is blocked")
    target = Path(candidates[recommended]["adapter"])
    FINAL_ALIAS.symlink_to(os.path.relpath(target, FINAL_ALIAS.parent), target_is_directory=True)
    SELECTION_RECORD.parent.mkdir(parents=True, exist_ok=True)
    payload.update({"executed": True, "selected": recommended, "created_at": datetime.now(UTC).isoformat()})
    SELECTION_RECORD.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("train", "predict", "evaluate", "select-final"), required=True)
    parser.add_argument("--run", choices=RUN_NAMES, default="mm-e1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--gpus", nargs="+", metavar="GPU")
    args = parser.parse_args()
    if args.gpus:
        selectors = [selector for value in args.gpus for selector in value.split(",") if selector]
        if len(selectors) != len(set(selectors)):
            parser.error("--gpus contains duplicate CUDA selectors")
        args.gpus = selectors
    return args


def main() -> None:
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    if args.action in {"train", "predict"}:
        _run_train_or_predict(args)
    elif args.action == "evaluate":
        _evaluate(args)
    else:
        _select_final(args)


if __name__ == "__main__":
    main()
