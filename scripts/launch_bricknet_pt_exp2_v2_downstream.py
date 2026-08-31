#!/usr/bin/env python3
"""Fail-closed launcher for the isolated PT-exp2-v2 Stage-2 downstream.

Only ``exp4_4`` and ``exp4_7`` are reachable here.  The historical
``launch_bricknet_pt_exp2.py`` entry point and its PT-exp2 YAMLs are not
imported or modified.  This launcher defaults to a read-only dry-run;
execution requires an explicit physical ``--gpus 1`` and, for training, an
explicit ``--pt-exp2-v2-approved``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/data/jiahao/task/BrickNet")
MS_SWIFT_ROOT = Path("/data/jiahao/task/ms-swift")
MS_SWIFT_EVALUATOR = (
    MS_SWIFT_ROOT / "examples/train/grpo/plugin/bricknet/evaluate_experiment.py"
)
ALIGNMENT_DATASET = (
    BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
)
CONFIG_ROOT = ROOT / "examples/train_lora"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
DATASET_REGISTRY = ROOT / "data/dataset_info.json"
SELECTION_RECORD = ROOT / "data/bricknet_pt_exp2/gates/PT-exp2-v2-selection.json"
FINAL_ALIAS = SAVE_ROOT / "PT-exp2-v2"
AUDIT_ROOT = (
    BRICKNET_ROOT
    / "outputs_preprocess/BrickNet-MM-Reasoning/pt_exp2_v2_downstream/reports/token_audit"
)
EVALUATION_ROOT = BRICKNET_ROOT / "outputs_val/qwen35_08b"
EXPECTED_TRAIN_ORDERED_ID_SHA256 = (
    "2d87ff4c3b918f748dde48721cbec66595ccc17317cf728f77e30efc04230dea"
)
EXPECTED_VAL_ORDERED_ID_SHA256 = (
    "908489739b3fff8489877da01b034081f80c5db8c21f357164c7ac38e02a6e51"
)
EXPECTED_EVAL_SAMPLES = 512
ALIGNMENT_SCHEMA_VERSION = "bricknet-stage2-alignment-v1"
POSE_TOLERANCES = {
    "translation": 0.5,
    "rotation_degrees": 5.0,
    "success_threshold": 1.0,
}
REWARD_WEIGHTS = {
    "parse_prefix": 0.20,
    "inventory_f1": 0.20,
    "length_score": 0.10,
    "collision_prefix": 0.20,
    "pose_match": 0.30,
}
EXPECTED_DATASETS = {
    "control_train": {
        "registry_key": "BrickNet-Stage2-NonThinking-Control-10k",
        "audit_key": "NonThinking-Control",
        "path": ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl",
        "count": 10_000,
        "sha256": "4be6a7fb711ba24a658fc3096a8f1e2a9aa3e630c76adb8f04313bf9bff02a15",
        "ordered_id_sha256": EXPECTED_TRAIN_ORDERED_ID_SHA256,
    },
    "lean_train": {
        "registry_key": "BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        "audit_key": "Thinking-Hard-V2-Lean-State",
        "path": ROOT
        / "data/bricknet_stage2_v2/10k/BrickNet-Stage2-ThinkingHard-V2-LeanState.jsonl",
        "count": 10_000,
        "sha256": "b0ee6b1046aaef6290ed7bb4d1b632c0260fbbd65048619c415e8669f9a6bc95",
        "ordered_id_sha256": EXPECTED_TRAIN_ORDERED_ID_SHA256,
    },
    "control_eval": {
        "registry_key": "BrickNet-Stage2-NonThinking-Control-VAL512-Eval",
        "audit_key": "NonThinking-Control-VAL512",
        "path": BRICKNET_ROOT
        / "outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl",
        "count": 512,
        "sha256": "9692d80e13995969937b6cd36780e35905ce244abd0ff0e3f8b5eaa34de76775",
        "ordered_id_sha256": EXPECTED_VAL_ORDERED_ID_SHA256,
    },
    "lean_eval": {
        "registry_key": "BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval",
        "audit_key": "Thinking-Hard-V2-Lean-State-VAL512",
        "path": BRICKNET_ROOT
        / "outputs_preprocess/BrickNet-MM-Reasoning/stage2_v2/validation/datasets/BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval.jsonl",
        "count": 512,
        "sha256": "f102e74a2462e38af0cfcbcd9fd012772c7b3c0bc6fc44f2746430d57eec1009",
        "ordered_id_sha256": EXPECTED_VAL_ORDERED_ID_SHA256,
    },
}


@dataclass(frozen=True)
class Run:
    experiment: str
    variant: str
    train_dataset: str
    eval_dataset: str
    train_file: Path
    eval_file: Path
    train_config: Path
    predict_config: Path
    train_output: Path
    predict_output: Path
    cache_path: Path
    train_audit_key: str
    eval_audit_key: str


RUNS = {
    "exp4_4": Run(
        experiment="exp4_4",
        variant="nonthinking-control",
        train_dataset="BrickNet-Stage2-NonThinking-Control-10k",
        eval_dataset="BrickNet-Stage2-NonThinking-Control-VAL512-Eval",
        train_file=EXPECTED_DATASETS["control_train"]["path"],
        eval_file=EXPECTED_DATASETS["control_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_10k_pt_exp2_v2.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_predict_pt_exp2_v2.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_4_qwen35_08b_PT_exp2_v2_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_4_PT_exp2_v2_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_4-PT_exp2_v2-nonthinking-control-10k-qwen35-08b-len16384",
        train_audit_key="NonThinking-Control",
        eval_audit_key="NonThinking-Control-VAL512",
    ),
    "exp4_7": Run(
        experiment="exp4_7",
        variant="thinking-hard-v2-lean-state",
        train_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        eval_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval",
        train_file=EXPECTED_DATASETS["lean_train"]["path"],
        eval_file=EXPECTED_DATASETS["lean_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_thinking_hard_v2_lean_state_10k_pt_exp2_v2.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_thinking_hard_v2_lean_state_predict_pt_exp2_v2.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_7_qwen35_08b_PT_exp2_v2_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_7_PT_exp2_v2_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_7-PT_exp2_v2-thinking-hard-v2-lean-state-10k-qwen35-08b-len16384",
        train_audit_key="Thinking-Hard-V2-Lean-State",
        eval_audit_key="Thinking-Hard-V2-Lean-State-VAL512",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--action", choices=("train", "predict", "evaluate"), required=True)
    parser.add_argument(
        "--gpus",
        required=True,
        choices=("1",),
        help="The downstream contract is fixed to physical CUDA 1; this flag is mandatory.",
    )
    parser.add_argument(
        "--pt-exp2-v2-approved",
        action="store_true",
        help="Explicit approval required for a PT-exp2-v2 train action.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Invoke the selected command after every gate passes. Default is dry-run.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    return None
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows


def _ordered_id_sha256(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, None
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                sample_id = row.get("id") if isinstance(row, dict) else None
                if not isinstance(sample_id, str):
                    return None, None
                digest.update(sample_id.encode("utf-8"))
                digest.update(b"\n")
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None
    return count, digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _adapter_complete(path: Path) -> bool:
    if not (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any((path / name).is_file() and (path / name).stat().st_size > 0 for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
        ))
        and (path / "trainer_state.json").is_file()
        and (path / "train_results.json").is_file()
        and (path / "all_results.json").is_file()
    ):
        return False
    trainer_state = _load_json(path / "trainer_state.json")
    if trainer_state is None:
        return False
    global_step = trainer_state.get("global_step")
    max_steps = trainer_state.get("max_steps")
    return (
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and global_step > 0
        and max_steps > 0
        and global_step == max_steps
    )


def _gpu1_compute_processes() -> list[str] | None:
    """Return GPU-1 compute rows, or None when occupancy cannot be checked."""
    uuid_command = [
        "nvidia-smi",
        "--id=1",
        "--query-gpu=uuid",
        "--format=csv,noheader",
    ]
    try:
        uuid_result = subprocess.run(uuid_command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    gpu_uuid = next((line.strip() for line in uuid_result.stdout.splitlines() if line.strip()), None)
    if not gpu_uuid:
        return None
    command = [
        "nvidia-smi",
        f"--id={gpu_uuid}",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _validate_selection(checks: dict[str, Any], blockers: list[str]) -> None:
    record = _load_json(SELECTION_RECORD)
    checks["selection_record"] = str(SELECTION_RECORD)
    if record is None:
        blockers.append("WAIT_PT_EXP2_V2_SELECTION_RECORD")
        return

    checks["selection_ready"] = record.get("ready") is True
    checks["selection_executed"] = record.get("executed") is True
    checks["selection_data_revision"] = record.get("data_revision")
    checks["selection_selected"] = record.get("selected")
    checks["selection_recommended"] = record.get("recommended")
    checks["selection_is_mm_e1"] = bool(
        record.get("selected") == "mm-e1"
        and record.get("recommended") == "mm-e1"
    )
    if not checks["selection_ready"]:
        blockers.append("PT_EXP2_V2_SELECTION_NOT_READY")
    if not checks["selection_executed"]:
        blockers.append("PT_EXP2_V2_SELECTION_NOT_EXECUTED")
    if record.get("data_revision") != "v2-no-meta":
        blockers.append("PT_EXP2_V2_SELECTION_REVISION_DRIFT")
    if not checks["selection_is_mm_e1"]:
        blockers.append("PT_EXP2_V2_SELECTION_MUST_BE_MM_E1")

    declared_alias = record.get("alias")
    checks["selection_alias_path_matches"] = bool(
        isinstance(declared_alias, str)
        and Path(declared_alias).expanduser().absolute() == FINAL_ALIAS.absolute()
    )
    if not checks["selection_alias_path_matches"]:
        blockers.append("PT_EXP2_V2_SELECTION_ALIAS_PATH_DRIFT")
    declared_record = record.get("record")
    checks["selection_record_path_matches"] = bool(
        isinstance(declared_record, str)
        and Path(declared_record).expanduser().absolute() == SELECTION_RECORD.absolute()
    )
    if not checks["selection_record_path_matches"]:
        blockers.append("PT_EXP2_V2_SELECTION_RECORD_PATH_DRIFT")

    candidates = record.get("candidates")
    candidate_checks: dict[str, Any] = {}
    if not isinstance(candidates, dict):
        blockers.append("PT_EXP2_V2_SELECTION_CANDIDATES_MISSING")
        candidates = {}
    for name in ("mm-e1", "mm-e2", "mm-e3"):
        candidate = candidates.get(name)
        candidate_checks[name] = {}
        if not isinstance(candidate, dict):
            blockers.append(f"PT_EXP2_V2_SELECTION_{name.upper().replace('-', '_')}_MISSING")
            continue
        metrics = Path(str(candidate.get("metrics", ""))).expanduser()
        adapter = Path(str(candidate.get("adapter", ""))).expanduser()
        actual_metrics_hash = _sha256(metrics)
        metrics_hash_matches = bool(
            actual_metrics_hash
            and actual_metrics_hash == candidate.get("metrics_sha256")
        )
        adapter_ready = _adapter_complete(adapter)
        metrics_payload = _load_json(metrics)
        metrics_complete = bool(
            metrics_payload
            and metrics_payload.get("artifacts", {}).get("predictions") == 512
            and metrics_payload.get("structure", {}).get("samples") == 512
            and metrics_payload.get("task_alignment", {}).get("samples") == 512
        )
        candidate_checks[name] = {
            "metrics": str(metrics),
            "metrics_sha256_record": candidate.get("metrics_sha256"),
            "metrics_sha256_actual": actual_metrics_hash,
            "metrics_hash_matches": metrics_hash_matches,
            "metrics_complete": metrics_complete,
            "adapter": str(adapter),
            "adapter_complete": adapter_ready,
            "adapter_model_sha256": next(
                (
                    _sha256(adapter / weight_name)
                    for weight_name in ("adapter_model.safetensors", "adapter_model.bin")
                    if (adapter / weight_name).is_file()
                ),
                None,
            ),
        }
        if not metrics_hash_matches:
            blockers.append(f"PT_EXP2_V2_{name.upper().replace('-', '_')}_METRICS_HASH_DRIFT")
        if not metrics_complete:
            blockers.append(f"PT_EXP2_V2_{name.upper().replace('-', '_')}_METRICS_INCOMPLETE")
        if not adapter_ready:
            blockers.append(f"PT_EXP2_V2_{name.upper().replace('-', '_')}_ADAPTER_INCOMPLETE")
    checks["selection_candidates"] = candidate_checks

    selected = candidates.get("mm-e1") if isinstance(candidates, dict) else None
    selected_adapter = (
        Path(str(selected.get("adapter", ""))).expanduser()
        if isinstance(selected, dict)
        else None
    )
    checks["alias_is_symlink"] = FINAL_ALIAS.is_symlink()
    checks["alias_target"] = None
    checks["alias_target_matches_selected"] = False
    if FINAL_ALIAS.is_symlink():
        try:
            raw_target = os.readlink(FINAL_ALIAS)
            resolved_target = (FINAL_ALIAS.parent / raw_target).resolve()
            checks["alias_target"] = str(resolved_target)
            checks["alias_target_matches_selected"] = bool(
                selected_adapter and resolved_target == selected_adapter.resolve()
            )
        except OSError:
            pass
    if not checks["alias_is_symlink"]:
        blockers.append("PT_EXP2_V2_ALIAS_NOT_SYMLINK")
    if not checks["alias_target_matches_selected"]:
        blockers.append("PT_EXP2_V2_ALIAS_TARGET_DRIFT")


def _validate_datasets(checks: dict[str, Any], blockers: list[str]) -> None:
    registry = _load_json(DATASET_REGISTRY)
    checks["dataset_registry"] = str(DATASET_REGISTRY)
    checks["datasets"] = {}
    if registry is None:
        blockers.append("WAIT_PT_EXP2_V2_DATASET_REGISTRY")
        return

    for name, spec in EXPECTED_DATASETS.items():
        path = spec["path"]
        actual_count, actual_order_hash = _ordered_id_sha256(path)
        actual_content_hash = _sha256(path)
        entry = registry.get(spec["registry_key"])
        registered = Path(str(entry.get("file_name", ""))).expanduser() if isinstance(entry, dict) else None
        row_checks = {
            "path": str(path),
            "count_expected": spec["count"],
            "count_actual": actual_count,
            "count_matches": actual_count == spec["count"],
            "sha256_expected": spec["sha256"],
            "sha256_actual": actual_content_hash,
            "sha256_matches": actual_content_hash == spec["sha256"],
            "ordered_id_sha256_expected": spec["ordered_id_sha256"],
            "ordered_id_sha256_actual": actual_order_hash,
            "ordered_id_sha256_matches": actual_order_hash == spec["ordered_id_sha256"],
            "registry_key": spec["registry_key"],
            "registry_path": str(registered) if registered else None,
            "registry_matches": bool(registered and _same_path(registered, path)),
        }
        checks["datasets"][name] = row_checks
        for key, label in (
            ("count_matches", "COUNT"),
            ("sha256_matches", "HASH"),
            ("ordered_id_sha256_matches", "ORDERED_IDS"),
            ("registry_matches", "REGISTRY"),
        ):
            if not row_checks[key]:
                blockers.append(f"PT_EXP2_V2_{name.upper()}_{label}_DRIFT")


def _validate_audit(
    report_name: str,
    expected_names: tuple[str, str],
    checks: dict[str, Any],
    blockers: list[str],
) -> None:
    report_path = AUDIT_ROOT / report_name / "BrickNet-MM-Reasoning_token_audit_report.json"
    report = _load_json(report_path)
    report_checks: dict[str, Any] = {"path": str(report_path), "exists": report is not None}
    checks[report_name] = report_checks
    if report is None:
        blockers.append(f"WAIT_PT_EXP2_V2_{report_name.upper()}_TOKEN_AUDIT")
        return

    report_checks.update(
        {
            "is_full_pool": report.get("is_full_pool") is True,
            "paired_order_and_ids": report.get("paired_order_and_ids") is True,
            "zero_errors": report.get("zero_errors") is True,
            "zero_truncation": report.get("zero_truncation") is True,
            "training_eligible": report.get("training_eligible") is True,
            "media_dir": report.get("config", {}).get("media_dir"),
            "media_dir_matches": report.get("config", {}).get("media_dir") == str(BRICKNET_ROOT),
        }
    )
    for key in (
        "is_full_pool",
        "paired_order_and_ids",
        "zero_errors",
        "zero_truncation",
        "training_eligible",
        "media_dir_matches",
    ):
        if not report_checks[key]:
            blockers.append(f"PT_EXP2_V2_{report_name.upper()}_{key.upper()}_FAILED")

    audited: dict[str, Any] = {}
    datasets = report.get("datasets", {})
    for dataset_name in expected_names:
        spec = next(
            item for item in EXPECTED_DATASETS.values() if item["audit_key"] == dataset_name
        )
        item = datasets.get(dataset_name)
        item_checks: dict[str, Any] = {}
        if not isinstance(item, dict):
            blockers.append(f"PT_EXP2_V2_{report_name.upper()}_{dataset_name.upper()}_MISSING")
            audited[dataset_name] = item_checks
            continue
        audited_path = Path(str(item.get("path", ""))).expanduser()
        item_checks.update(
            {
                "path": str(audited_path),
                "path_matches": _same_path(audited_path, spec["path"]),
                "sha256": item.get("sha256"),
                "sha256_matches": item.get("sha256") == spec["sha256"],
                "count": item.get("count"),
                "count_matches": item.get("count") == spec["count"],
                "errors": item.get("errors"),
                "truncated": item.get("truncated"),
                "zero_errors": item.get("errors") == 0,
                "zero_truncation": item.get("truncated") == 0,
                "ordered_id_sha256": item.get("ordered_id_sha256"),
                "ordered_id_sha256_matches": item.get("ordered_id_sha256") == spec["ordered_id_sha256"],
            }
        )
        audited[dataset_name] = item_checks
        for key, label in (
            ("path_matches", "PATH"),
            ("sha256_matches", "HASH"),
            ("count_matches", "COUNT"),
            ("zero_errors", "ERRORS"),
            ("zero_truncation", "TRUNCATION"),
            ("ordered_id_sha256_matches", "ORDERED_IDS"),
        ):
            if not item_checks[key]:
                blockers.append(f"PT_EXP2_V2_{report_name.upper()}_{label}_FAILED")
    report_checks["datasets"] = audited


def _yaml_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key)


def _validate_config(run: Run, action: str, checks: dict[str, Any], blockers: list[str]) -> Path:
    config_path = run.train_config if action == "train" else run.predict_config
    checks["config"] = str(config_path)
    if not config_path.is_file():
        blockers.append("CONFIG_MISSING")
        return config_path
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        config = None
    if not isinstance(config, dict):
        blockers.append("CONFIG_INVALID_YAML")
        return config_path

    expected_common = {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "trust_remote_code": True,
        "finetuning_type": "lora",
        "dataset_dir": "data",
        "media_dir": str(BRICKNET_ROOT),
        "template": "qwen3_5_nothink",
        "enable_thinking": False,
        "cutoff_len": 16384,
    }
    expected = dict(expected_common)
    if action == "train":
        expected.update(
            {
                "adapter_name_or_path": "saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2",
                "stage": "sft",
                "do_train": True,
                "create_new_adapter": True,
                "lora_target": "all",
                "lora_rank": 64,
                "lora_alpha": 128,
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "train_on_prompt": False,
                "packing": False,
                "dataset": run.train_dataset,
                "output_dir": str(run.train_output.relative_to(ROOT)),
                "tokenized_path": str(run.cache_path.relative_to(ROOT)),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 5.0e-5,
                "num_train_epochs": 3.0,
                "ddp_find_unused_parameters": False,
            }
        )
    else:
        expected.update(
            {
                "adapter_name_or_path": (
                    "saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2,"
                    f"{run.train_output.relative_to(ROOT)}"
                ),
                "stage": "sft",
                "do_predict": True,
                "predict_with_generate": True,
                "eval_dataset": run.eval_dataset,
                "output_dir": str(run.predict_output.relative_to(ROOT)),
            }
        )
    mismatches = {
        key: {"expected": value, "actual": _yaml_value(config, key)}
        for key, value in expected.items()
        if _yaml_value(config, key) != value
    }
    checks["config_fields"] = {"expected": expected, "mismatches": mismatches}
    if mismatches:
        blockers.append("CONFIG_CONTRACT_DRIFT")
    if "PT_exp2_v2" not in str(config.get("output_dir", "")) and "PT_exp2_v2" not in str(config.get("tokenized_path", "")):
        blockers.append("CONFIG_PT_EXP2_V2_NAMESPACE_MISSING")
    return config_path


def _prediction_complete(path: Path) -> bool:
    return bool(
        path.is_dir()
        and _line_count(path / "generated_predictions.jsonl") == 512
        and (path / "predict_results.json").is_file()
        and (path / "all_results.json").is_file()
    )


def _evaluation_path(run: Run) -> Path:
    return EVALUATION_ROOT / run.predict_output.name


def _is_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metrics_complete(metrics: dict[str, Any]) -> bool:
    structure = metrics.get("structure", {})
    task_alignment = metrics.get("task_alignment", {})
    condition_generation = metrics.get("condition_generation", {})
    dense_reward = task_alignment.get("dense_reward_mean")
    strict_success = task_alignment.get("strict_success")
    strict_success_rate = task_alignment.get("strict_success_rate")
    return bool(
        structure.get("samples") == EXPECTED_EVAL_SAMPLES
        and task_alignment.get("samples") == EXPECTED_EVAL_SAMPLES
        and condition_generation.get("samples") == EXPECTED_EVAL_SAMPLES
        and metrics.get("artifacts", {}).get("predictions") == EXPECTED_EVAL_SAMPLES
        and metrics.get("artifacts", {}).get("scored") == EXPECTED_EVAL_SAMPLES
        and _is_number(dense_reward)
        and isinstance(strict_success, int)
        and not isinstance(strict_success, bool)
        and 0 <= strict_success <= EXPECTED_EVAL_SAMPLES
        and _is_number(strict_success_rate)
        and 0.0 <= float(strict_success_rate) <= 1.0
        and strict_success_rate == strict_success / EXPECTED_EVAL_SAMPLES
        and task_alignment.get("weights") == REWARD_WEIGHTS
        and task_alignment.get("pose_tolerances") == POSE_TOLERANCES
        and condition_generation.get("dense_reward") == dense_reward
        and condition_generation.get("strict_success_num") == strict_success
        and condition_generation.get("strict_success_rate") == strict_success_rate
    )


def _jsonl_artifact(path: Path) -> dict[str, Any] | None:
    rows = _load_jsonl(path)
    digest = _sha256(path)
    if rows is None or digest is None:
        return None
    return {"path": str(path.resolve()), "sha256": digest, "count": len(rows)}


def _file_artifact(path: Path) -> dict[str, str] | None:
    digest = _sha256(path)
    if digest is None:
        return None
    return {"path": str(path.resolve()), "sha256": digest}


def _canonical_ready(run: Run) -> bool:
    generated = run.predict_output / "generated_predictions.jsonl"
    canonical = run.predict_output / "path_predictions.jsonl"
    report = _load_json(run.predict_output / "trace_extraction_report.json")
    generated_rows = _load_jsonl(generated)
    canonical_rows = _load_jsonl(canonical)
    return bool(
        report
        and generated_rows is not None
        and canonical_rows is not None
        and len(generated_rows) == EXPECTED_EVAL_SAMPLES
        and len(canonical_rows) == EXPECTED_EVAL_SAMPLES
        and report.get("variant") == run.variant
        and report.get("count") == EXPECTED_EVAL_SAMPLES
        and report.get("input_sha256") == _sha256(generated)
        and report.get("output_sha256") == _sha256(canonical)
    )


def _base_evaluation_complete(run: Run) -> bool:
    output = _evaluation_path(run)
    canonical = run.predict_output / "path_predictions.jsonl"
    metrics_path = output / "metrics.json"
    scored = output / "scored.jsonl"
    manifest = _load_json(output / "evaluation_manifest.json")
    metrics = _load_json(metrics_path)
    scored_rows = _load_jsonl(scored)
    if not (manifest and metrics and scored_rows is not None and canonical.is_file()):
        return False
    try:
        manifest_predictions = Path(str(manifest.get("predictions", ""))).resolve()
        manifest_metrics = Path(str(manifest.get("metrics", ""))).resolve()
    except OSError:
        return False
    return bool(
        manifest_predictions == canonical.resolve()
        and manifest_metrics == metrics_path.resolve()
        and manifest.get("predictions_sha256") == _sha256(canonical)
        and manifest.get("status") == "complete"
        and manifest.get("samples") == EXPECTED_EVAL_SAMPLES
        and len(scored_rows) == EXPECTED_EVAL_SAMPLES
        and _metrics_complete(metrics)
    )


def _normalize_path_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _assistant_reference(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    assistants = [
        item.get("content")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    if not assistants or not isinstance(assistants[-1], str):
        return None
    return assistants[-1]


def _close(left: Any, right: Any) -> bool:
    return bool(
        _is_number(left)
        and _is_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _alignment_artifacts_consistent(run: Run) -> bool:
    output = _evaluation_path(run)
    canonical = _load_jsonl(run.predict_output / "path_predictions.jsonl")
    alignment_input = _load_jsonl(output / "alignment_input.jsonl")
    dataset = _load_jsonl(ALIGNMENT_DATASET)
    scored = _load_jsonl(output / "scored.jsonl")
    alignment = _load_jsonl(output / "alignment.jsonl")
    metrics = _load_json(output / "metrics.json")
    if any(item is None for item in (canonical, alignment_input, dataset, scored, alignment, metrics)):
        return False
    assert canonical is not None
    assert alignment_input is not None
    assert dataset is not None
    assert scored is not None
    assert alignment is not None
    assert metrics is not None
    if not all(
        len(rows) == EXPECTED_EVAL_SAMPLES
        for rows in (canonical, alignment_input, dataset, scored, alignment)
    ):
        return False

    score_keys = (
        "parse_prefix",
        "inventory_precision",
        "inventory_recall",
        "inventory_f1",
        "length_score",
        "collision_prefix",
        "pose_match",
        "dense_reward",
    )
    mean_fields = {
        "parse_prefix": "parse_prefix_mean",
        "inventory_precision": "inventory_precision_mean",
        "inventory_recall": "inventory_recall_mean",
        "inventory_f1": "inventory_f1_mean",
        "length_score": "length_score_mean",
        "collision_prefix": "collision_prefix_mean",
        "pose_match": "pose_match_mean",
        "dense_reward": "dense_reward_mean",
    }
    values: dict[str, list[float]] = {key: [] for key in score_keys}
    strict_success = 0
    seen_ids: set[Any] = set()
    for index, (canonical_row, input_row, dataset_row, scored_row, alignment_row) in enumerate(
        zip(canonical, alignment_input, dataset, scored, alignment)
    ):
        prediction = canonical_row.get("predict")
        label = canonical_row.get("label")
        reference = _assistant_reference(dataset_row)
        if not all(isinstance(item, str) for item in (prediction, label, reference)):
            return False
        assert isinstance(prediction, str) and isinstance(label, str) and isinstance(reference, str)
        if _normalize_path_text(label) != _normalize_path_text(reference):
            return False
        if input_row != {"response": prediction, "label": label}:
            return False
        scored_text = scored_row.get("text", scored_row.get("path"))
        collisions = scored_row.get("collisions")
        if (
            scored_row.get("id") != index
            or not isinstance(scored_text, str)
            or _normalize_path_text(scored_text) != _normalize_path_text(prediction)
            or not isinstance(collisions, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in collisions
            )
        ):
            return False
        expected_id = dataset_row.get("id", index)
        if expected_id in seen_ids:
            return False
        seen_ids.add(expected_id)
        if (
            alignment_row.get("index") != index
            or alignment_row.get("id") != expected_id
            or alignment_row.get("collisions") != collisions
        ):
            return False
        for key in score_keys:
            value = alignment_row.get(key)
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                return False
            values[key].append(float(value))
        strict_value = alignment_row.get("strict_success")
        if not isinstance(strict_value, bool):
            return False
        strict_success += int(strict_value)
        expected_dense = sum(
            REWARD_WEIGHTS[key] * float(alignment_row[key])
            for key in REWARD_WEIGHTS
        )
        if not _close(alignment_row["dense_reward"], expected_dense):
            return False

    task_alignment = metrics.get("task_alignment", {})
    for key, metric_key in mean_fields.items():
        mean = math.fsum(values[key]) / EXPECTED_EVAL_SAMPLES
        if not _close(task_alignment.get(metric_key), mean):
            return False
    return bool(
        task_alignment.get("strict_success") == strict_success
        and _close(
            task_alignment.get("strict_success_rate"),
            strict_success / EXPECTED_EVAL_SAMPLES,
        )
    )


def _manifest_sections(
    run: Run,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    output = _evaluation_path(run)
    generated = run.predict_output / "generated_predictions.jsonl"
    canonical = run.predict_output / "path_predictions.jsonl"
    metrics_path = output / "metrics.json"
    metrics = _load_json(metrics_path)
    if metrics is None:
        return None
    identity = {
        "experiment": run.experiment,
        "variant": run.variant,
        "trace_variant": run.variant,
        "expected_samples": EXPECTED_EVAL_SAMPLES,
        "generated_predictions": _jsonl_artifact(generated),
        "canonical_predictions": _jsonl_artifact(canonical),
        "scored": _jsonl_artifact(output / "scored.jsonl"),
        "alignment_dataset": _jsonl_artifact(ALIGNMENT_DATASET),
        "alignment_evaluator": _file_artifact(MS_SWIFT_EVALUATOR),
        "base_evaluation_manifest": _file_artifact(
            output / "evaluation_manifest.json"
        ),
        "reward_weights": REWARD_WEIGHTS,
        "pose_tolerances": POSE_TOLERANCES,
    }
    artifacts = {
        "alignment_input": _jsonl_artifact(output / "alignment_input.jsonl"),
        "alignment": _jsonl_artifact(output / "alignment.jsonl"),
        "metrics_json": _file_artifact(metrics_path),
        "metrics_md": _file_artifact(output / "metrics.md"),
    }
    if any(value is None for value in identity.values()) or any(
        value is None for value in artifacts.values()
    ):
        return None
    summary = {
        "structure_samples": metrics.get("structure", {}).get("samples"),
        "task_alignment_samples": metrics.get("task_alignment", {}).get("samples"),
        "condition_generation_samples": metrics.get("condition_generation", {}).get(
            "samples"
        ),
        "dense_reward_mean": metrics.get("task_alignment", {}).get(
            "dense_reward_mean"
        ),
        "strict_success": metrics.get("task_alignment", {}).get("strict_success"),
        "strict_success_rate": metrics.get("task_alignment", {}).get(
            "strict_success_rate"
        ),
    }
    return identity, artifacts, summary


def _valid_completed_at(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _evaluation_complete(run: Run) -> bool:
    output = _evaluation_path(run)
    manifest = _load_json(output / "alignment_manifest.json")
    metrics = _load_json(output / "metrics.json")
    sections = _manifest_sections(run)
    if not (
        manifest
        and metrics
        and sections
        and _canonical_ready(run)
        and _base_evaluation_complete(run)
        and _alignment_artifacts_consistent(run)
    ):
        return False
    identity, artifacts, summary = sections
    return bool(
        set(manifest) == {
            "schema_version",
            "status",
            "completed_at",
            "freeze",
            "identity",
            "artifacts",
            "summary",
        }
        and manifest.get("schema_version") == ALIGNMENT_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and _valid_completed_at(manifest.get("completed_at"))
        and manifest.get("freeze")
        in (
            {"mode": "verified_numeric_backfill"},
            {"mode": "post_evaluation_freeze"},
        )
        and manifest.get("identity") == identity
        and manifest.get("artifacts") == artifacts
        and manifest.get("summary") == summary
        and _metrics_complete(metrics)
        and identity["generated_predictions"]["count"] == EXPECTED_EVAL_SAMPLES
        and identity["canonical_predictions"]["count"] == EXPECTED_EVAL_SAMPLES
        and identity["scored"]["count"] == EXPECTED_EVAL_SAMPLES
        and identity["alignment_dataset"]["count"] == EXPECTED_EVAL_SAMPLES
        and artifacts["alignment_input"]["count"] == EXPECTED_EVAL_SAMPLES
        and artifacts["alignment"]["count"] == EXPECTED_EVAL_SAMPLES
    )


def _output_state(run: Run, action: str) -> tuple[Path, bool, bool]:
    if action == "train":
        path = run.train_output
        return path, path.exists(), _adapter_complete(path)
    if action == "predict":
        path = run.predict_output
        return path, path.exists(), _prediction_complete(path)
    path = _evaluation_path(run)
    return path, path.exists(), _evaluation_complete(run)


def _conda_run_prefix() -> list[str]:
    return ["conda", "run", "-n", "llamafactory", "--no-capture-output"]


def _command(run: Run, action: str, execute: bool) -> list[str]:
    if action == "evaluate":
        command = [
            *_conda_run_prefix(),
            "python",
            "scripts/evaluate_bricknet_stage2.py",
            "--experiment",
            run.experiment,
        ]
        if execute:
            command.append("--execute")
        return command
    config = run.train_config if action == "train" else run.predict_config
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(config.relative_to(ROOT)),
    ]
    if action == "train":
        # Keep the single-visible-GPU contract explicit at the command line
        # as well as in the YAML, so inherited/default launcher settings
        # cannot silently change the effective accumulation.
        command.append("gradient_accumulation_steps=16")
    return command


def main() -> None:
    args = parse_args()
    run = RUNS[args.run]
    blockers: list[str] = []
    checks: dict[str, Any] = {"run": run.experiment, "action": args.action, "gpus": args.gpus}

    _validate_selection(checks, blockers)
    _validate_datasets(checks, blockers)
    _validate_audit(
        "train10k",
        ("NonThinking-Control", "Thinking-Hard-V2-Lean-State"),
        checks,
        blockers,
    )
    _validate_audit(
        "eval_val512",
        ("NonThinking-Control-VAL512", "Thinking-Hard-V2-Lean-State-VAL512"),
        checks,
        blockers,
    )
    _validate_config(run, args.action, checks, blockers)

    output, output_exists, output_complete = _output_state(run, args.action)
    checks["output"] = str(output)
    checks["output_exists"] = output_exists
    checks["output_complete"] = output_complete
    checks["output_action_safe_noop"] = output_complete
    if output_exists and not output_complete:
        blockers.append("OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED")

    if args.action == "train":
        checks["pt_exp2_v2_approved"] = args.pt_exp2_v2_approved
        if not args.pt_exp2_v2_approved:
            blockers.append("PT_EXP2_V2_APPROVAL_REQUIRED_FOR_TRAIN")
    if args.action in ("train", "predict", "evaluate"):
        gpu_processes = _gpu1_compute_processes()
        checks["gpu1_compute_processes"] = gpu_processes
        checks["gpu1_occupancy_check_passed"] = gpu_processes == []
        if gpu_processes is None:
            blockers.append("GPU1_OCCUPANCY_UNKNOWN_FAIL_CLOSED")
        elif gpu_processes:
            blockers.append("GPU1_COMPUTE_OCCUPIED_NO_AUTO_KILL")
    if args.action == "predict":
        checks["train_adapter"] = str(run.train_output)
        checks["train_adapter_complete"] = _adapter_complete(run.train_output)
        if not checks["train_adapter_complete"]:
            blockers.append("WAIT_FINAL_TRAIN_ADAPTER")
    if args.action == "evaluate":
        checks["prediction_output"] = str(run.predict_output)
        checks["prediction_output_complete"] = _prediction_complete(run.predict_output)
        if not checks["prediction_output_complete"]:
            blockers.append("WAIT_FINAL_PREDICTION")

    command = _command(run, args.action, args.execute)
    result = {
        "experiment": run.experiment,
        "action": args.action,
        "mode": "execute" if args.execute else "dry-run",
        "gpus": args.gpus,
        "ready": not blockers,
        "already_complete": output_complete,
        "checks": checks,
        "blockers": blockers,
        "command": shlex.join(command),
        "executed": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2-v2 downstream launch blocked; resolve the reported gates first")
    if output_complete:
        print(f"Already complete; safe no-op: {output}")
        return

    env = os.environ.copy()
    for name in (
        "FORCE_TORCHRUN",
        "NPROC_PER_NODE",
        "NNODES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
    ):
        env.pop(name, None)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.action == "train" and not _adapter_complete(run.train_output):
        raise SystemExit("Training command returned without a complete adapter")
    if args.action == "predict" and not _prediction_complete(run.predict_output):
        raise SystemExit("Prediction command returned without 512 complete rows")
    if args.action == "evaluate" and not _evaluation_complete(run):
        raise SystemExit("Evaluation command returned without a complete VAL512 artifact")


if __name__ == "__main__":
    main()
