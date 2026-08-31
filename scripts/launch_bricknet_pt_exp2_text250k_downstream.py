#!/usr/bin/env python3
"""Fail-closed launcher for the isolated Text250k-only Stage-2 downstream.

The two exposed runs start from the frozen text8m 250k-step adapter itself.
No PT-exp2 alias and no MM e1/e2/e3 adapter is accepted by this launcher.
The default is a read-only dry-run.  Execution is restricted to physical
CUDA device 1; training additionally requires ``--text250k-approved``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
TEXT250K_ADAPTER = SAVE_ROOT / (
    "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_"
    "lora64_len6401_nopack"
)
TEXT8M_ADAPTER = TEXT250K_ADAPTER
TEXT250K_ADAPTER_MODEL_SHA256 = (
    "a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd"
)
TEXT250K_ADAPTER_CONFIG_SHA256 = (
    "f3d911f759aa05a05a7eaf0192f3a1b537dd816f524b5b99e7bfbeb22c39446a"
)
TEXT250K_TRAINER_STATE_SHA256 = (
    "dd55f2ea0c42e3d9f562f78b5a5fd74d3d8a017ec3ab1e7c50d9956adfead997"
)
TEXT250K_TRAIN_RESULTS_SHA256 = (
    "b07662baf32da06765e43d46c518403560dfa9775eff2b8ec05c096ca2c6e7b4"
)
EXPECTED_TEXT250K_ADAPTER_HASHES = {
    "adapter_model.safetensors": TEXT250K_ADAPTER_MODEL_SHA256,
    "adapter_config.json": TEXT250K_ADAPTER_CONFIG_SHA256,
    "trainer_state.json": TEXT250K_TRAINER_STATE_SHA256,
    "train_results.json": TEXT250K_TRAIN_RESULTS_SHA256,
}
EXPECTED_TEXT250K_GLOBAL_STEP = 250_000
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
    "exp4_4_2": Run(
        experiment="exp4_4_2",
        variant="nonthinking-control",
        train_dataset="BrickNet-Stage2-NonThinking-Control-10k",
        eval_dataset="BrickNet-Stage2-NonThinking-Control-VAL512-Eval",
        train_file=EXPECTED_DATASETS["control_train"]["path"],
        eval_file=EXPECTED_DATASETS["control_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_2_nonthinking_control_10k_pt_exp2_text250k.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_2_nonthinking_control_predict_pt_exp2_text250k.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_4_2_qwen35_08b_PT_exp2_text250k_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_4_2_PT_exp2_text250k_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_4_2-PT_exp2_text250k-nonthinking-control-10k-qwen35-08b-len16384",
        train_audit_key="NonThinking-Control",
        eval_audit_key="NonThinking-Control-VAL512",
    ),
    "exp4_7_2": Run(
        experiment="exp4_7_2",
        variant="thinking-hard-v2-lean-state",
        train_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        eval_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval",
        train_file=EXPECTED_DATASETS["lean_train"]["path"],
        eval_file=EXPECTED_DATASETS["lean_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_2_thinking_hard_v2_lean_state_10k_pt_exp2_text250k.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_2_thinking_hard_v2_lean_state_predict_pt_exp2_text250k.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_7_2_qwen35_08b_PT_exp2_text250k_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_7_2_PT_exp2_text250k_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_7_2-PT_exp2_text250k-thinking-hard-v2-lean-state-10k-qwen35-08b-len16384",
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
        help="This downstream contract is fixed to physical CUDA 1.",
    )
    parser.add_argument(
        "--text250k-approved",
        action="store_true",
        help="Explicit approval required to execute a Text250k train action.",
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


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
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return rows


def _line_count(path: Path) -> int | None:
    rows = _load_jsonl(path)
    return len(rows) if rows is not None else None


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
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    return None, None
                digest.update(row["id"].encode("utf-8"))
                digest.update(b"\n")
                count += 1
    except (OSError, UnicodeDecodeError, ValueError):
        return None, None
    return count, digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _adapter_complete(path: Path) -> bool:
    required = (
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
        path / "trainer_state.json",
        path / "train_results.json",
        path / "all_results.json",
    )
    if not path.is_dir() or not all(item.is_file() and item.stat().st_size > 0 for item in required):
        return False
    state = _load_json(path / "trainer_state.json")
    if state is None:
        return False
    global_step = state.get("global_step")
    max_steps = state.get("max_steps")
    return bool(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and global_step > 0
        and max_steps > 0
        and global_step == max_steps
    )


def _prediction_complete(path: Path) -> bool:
    return bool(
        path.is_dir()
        and _line_count(path / "generated_predictions.jsonl") == EXPECTED_EVAL_SAMPLES
        and (path / "predict_results.json").is_file()
        and (path / "all_results.json").is_file()
    )


def _gpu1_compute_processes() -> list[str] | None:
    """Return physical GPU-1 compute rows, or None when the check is unknown."""
    try:
        uuid_result = subprocess.run(
            ["nvidia-smi", "--id=1", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    gpu_uuid = next((line.strip() for line in uuid_result.stdout.splitlines() if line.strip()), None)
    if not gpu_uuid:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_uuid}",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _validate_text250k_adapter(checks: dict[str, Any], blockers: list[str]) -> None:
    checks["text250k_adapter"] = str(TEXT250K_ADAPTER)
    checks["text250k_adapter_is_dir"] = TEXT250K_ADAPTER.is_dir()
    checks["text250k_adapter_is_symlink"] = TEXT250K_ADAPTER.is_symlink()
    if not checks["text250k_adapter_is_dir"]:
        blockers.append("TEXT250K_ADAPTER_DIRECTORY_MISSING")
    if checks["text250k_adapter_is_symlink"]:
        blockers.append("TEXT250K_ADAPTER_ALIAS_SYMLINK_FORBIDDEN")
    checks["text250k_adapter_path_exact"] = TEXT250K_ADAPTER.name == (
        "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack"
    )
    if not checks["text250k_adapter_path_exact"]:
        blockers.append("TEXT250K_ADAPTER_PATH_CONTRACT_DRIFT")
    checks["text250k_adapter_files"] = {}
    for name, expected in EXPECTED_TEXT250K_ADAPTER_HASHES.items():
        actual = _sha256(TEXT250K_ADAPTER / name)
        matches = actual == expected
        checks["text250k_adapter_files"][name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": matches,
        }
        if not matches:
            blockers.append(f"TEXT250K_{name.upper().replace('.', '_')}_HASH_DRIFT")
    state = _load_json(TEXT250K_ADAPTER / "trainer_state.json")
    global_step = state.get("global_step") if state else None
    max_steps = state.get("max_steps") if state else None
    checks["text250k_global_step"] = global_step
    checks["text250k_max_steps"] = max_steps
    checks["text250k_steps_match"] = bool(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and global_step == EXPECTED_TEXT250K_GLOBAL_STEP
        and max_steps == EXPECTED_TEXT250K_GLOBAL_STEP
    )
    if not checks["text250k_steps_match"]:
        blockers.append("TEXT250K_TRAINER_STATE_GLOBAL_MAX_DRIFT")


def _validate_datasets(checks: dict[str, Any], blockers: list[str]) -> None:
    registry = _load_json(DATASET_REGISTRY)
    checks["dataset_registry"] = str(DATASET_REGISTRY)
    checks["datasets"] = {}
    if registry is None:
        blockers.append("WAIT_TEXT250K_DATASET_REGISTRY")
        return
    for name, spec in EXPECTED_DATASETS.items():
        path = spec["path"]
        count, ordered_hash = _ordered_id_sha256(path)
        content_hash = _sha256(path)
        entry = registry.get(spec["registry_key"])
        registered = (
            Path(str(entry.get("file_name", ""))).expanduser()
            if isinstance(entry, dict)
            else None
        )
        row = {
            "path": str(path),
            "count_expected": spec["count"],
            "count_actual": count,
            "count_matches": count == spec["count"],
            "sha256_expected": spec["sha256"],
            "sha256_actual": content_hash,
            "sha256_matches": content_hash == spec["sha256"],
            "ordered_id_sha256_expected": spec["ordered_id_sha256"],
            "ordered_id_sha256_actual": ordered_hash,
            "ordered_id_sha256_matches": ordered_hash == spec["ordered_id_sha256"],
            "registry_key": spec["registry_key"],
            "registry_path": str(registered) if registered else None,
            "registry_matches": bool(registered and _same_path(registered, path)),
        }
        checks["datasets"][name] = row
        for key, suffix in (
            ("count_matches", "COUNT"),
            ("sha256_matches", "HASH"),
            ("ordered_id_sha256_matches", "ORDERED_IDS"),
            ("registry_matches", "REGISTRY"),
        ):
            if not row[key]:
                blockers.append(f"TEXT250K_{name.upper()}_{suffix}_DRIFT")


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
        blockers.append(f"WAIT_TEXT250K_{report_name.upper()}_TOKEN_AUDIT")
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
            blockers.append(f"TEXT250K_{report_name.upper()}_{key.upper()}_FAILED")
    audited: dict[str, Any] = {}
    datasets = report.get("datasets", {})
    for dataset_name in expected_names:
        spec = next(item for item in EXPECTED_DATASETS.values() if item["audit_key"] == dataset_name)
        item = datasets.get(dataset_name)
        if not isinstance(item, dict):
            audited[dataset_name] = {}
            blockers.append(f"TEXT250K_{report_name.upper()}_{dataset_name.upper()}_MISSING")
            continue
        audited_path = Path(str(item.get("path", ""))).expanduser()
        row = {
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
        audited[dataset_name] = row
        for key, suffix in (
            ("path_matches", "PATH"),
            ("sha256_matches", "HASH"),
            ("count_matches", "COUNT"),
            ("zero_errors", "ERRORS"),
            ("zero_truncation", "TRUNCATION"),
            ("ordered_id_sha256_matches", "ORDERED_IDS"),
        ):
            if not row[key]:
                blockers.append(f"TEXT250K_{report_name.upper()}_{suffix}_FAILED")
    report_checks["datasets"] = audited


_FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(?i)(?:^|[/\\])pt[-_]?exp2[-_]?v(?:1|2|3)(?:[/\\,\s]|$)"),
    re.compile(r"(?i)(?:^|[/\\])pt[-_]?exp2(?:[/\\,\s]|$)"),
    re.compile(r"(?i)(?:^|[^a-z0-9])mm[-_ ]?e[123](?:[^a-z0-9]|$)"),
    re.compile(r"(?i)pt[-_]?exp2[-_]?mm[-_]?e[123](?:[^a-z0-9]|$)"),
)


def _forbidden_binding_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [pattern.pattern for pattern in _FORBIDDEN_PATH_PATTERNS if pattern.search(value)]


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
    common = {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "trust_remote_code": True,
        "finetuning_type": "lora",
        "dataset_dir": "data",
        "media_dir": str(BRICKNET_ROOT),
        "template": "qwen3_5_nothink",
        "enable_thinking": False,
        "cutoff_len": 16384,
    }
    expected = dict(common)
    if action == "train":
        expected.update(
            {
                "adapter_name_or_path": str(TEXT250K_ADAPTER.relative_to(ROOT)),
                "flash_attn": "auto",
                "disable_gradient_checkpointing": False,
                "stage": "sft",
                "do_train": True,
                "create_new_adapter": True,
                "lora_target": "all",
                "lora_rank": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.0,
                "freeze_vision_tower": True,
                "freeze_multi_modal_projector": True,
                "train_on_prompt": False,
                "packing": False,
                "dataset": run.train_dataset,
                "output_dir": str(run.train_output.relative_to(ROOT)),
                "tokenized_path": str(run.cache_path.relative_to(ROOT)),
                "preprocessing_num_workers": 16,
                "dataloader_num_workers": 4,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 5.0e-5,
                "num_train_epochs": 3.0,
                "lr_scheduler_type": "cosine",
                "warmup_steps": 0,
                "max_grad_norm": 1.0,
                "optim": "adamw_torch",
                "bf16": True,
                "ddp_timeout": 180000000,
                "ddp_find_unused_parameters": False,
                "seed": 42,
            }
        )
    else:
        expected.update(
            {
                "flash_attn": "auto",
                "stage": "sft",
                "do_predict": True,
                "predict_with_generate": True,
                "eval_dataset": run.eval_dataset,
                "output_dir": str(run.predict_output.relative_to(ROOT)),
                "preprocessing_num_workers": 16,
                "per_device_eval_batch_size": 1,
                "seed": 42,
                "max_new_tokens": 16384,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
            }
        )
        expected["adapter_name_or_path"] = (
            f"{TEXT250K_ADAPTER.relative_to(ROOT)},{run.train_output.relative_to(ROOT)}"
        )
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    checks["config_fields"] = {"expected": expected, "mismatches": mismatches}
    if mismatches:
        blockers.append("CONFIG_CONTRACT_DRIFT")
    binding_fields = (
        "adapter_name_or_path",
        "output_dir",
        "tokenized_path",
    )
    forbidden = {
        key: _forbidden_binding_tokens(config.get(key))
        for key in binding_fields
        if _forbidden_binding_tokens(config.get(key))
    }
    checks["forbidden_binding_tokens"] = forbidden
    if forbidden:
        blockers.append("FORBIDDEN_PT_EXP2_ALIAS_OR_MM_ADAPTER")
    return config_path


def _evaluation_path(run: Run) -> Path:
    return EVALUATION_ROOT / run.predict_output.name


def _is_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(left: Any, right: Any) -> bool:
    return bool(
        _is_number(left)
        and _is_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
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
    return {"path": str(path.resolve()), "sha256": digest} if digest else None


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
    manifest = _load_json(output / "evaluation_manifest.json")
    metrics = _load_json(metrics_path)
    scored_rows = _load_jsonl(output / "scored.jsonl")
    if not (manifest and metrics and scored_rows is not None and canonical.is_file()):
        return False
    return bool(
        Path(str(manifest.get("predictions", ""))).expanduser().resolve() == canonical.resolve()
        and Path(str(manifest.get("metrics", ""))).expanduser().resolve() == metrics_path.resolve()
        and manifest.get("predictions_sha256") == _sha256(canonical)
        and manifest.get("status") == "complete"
        and manifest.get("samples") == EXPECTED_EVAL_SAMPLES
        and len(scored_rows) == EXPECTED_EVAL_SAMPLES
        and _metrics_complete(metrics)
    )


def _assistant_reference(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    assistants = [
        item.get("content")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    return assistants[-1] if assistants and isinstance(assistants[-1], str) else None


def _normalize_path_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _alignment_artifacts_consistent(run: Run) -> bool:
    output = _evaluation_path(run)
    canonical = _load_jsonl(run.predict_output / "path_predictions.jsonl")
    alignment_input = _load_jsonl(output / "alignment_input.jsonl")
    dataset = _load_jsonl(ALIGNMENT_DATASET)
    scored = _load_jsonl(output / "scored.jsonl")
    alignment = _load_jsonl(output / "alignment.jsonl")
    metrics = _load_json(output / "metrics.json")
    if any(value is None for value in (canonical, alignment_input, dataset, scored, alignment, metrics)):
        return False
    assert canonical is not None and alignment_input is not None and dataset is not None
    assert scored is not None and alignment is not None and metrics is not None
    if not all(len(rows) == EXPECTED_EVAL_SAMPLES for rows in (canonical, alignment_input, dataset, scored, alignment)):
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
    mean_fields = {key: f"{key}_mean" for key in score_keys}
    values: dict[str, list[float]] = {key: [] for key in score_keys}
    strict_success = 0
    seen_ids: set[Any] = set()
    for index, (prediction_row, input_row, dataset_row, scored_row, alignment_row) in enumerate(
        zip(canonical, alignment_input, dataset, scored, alignment)
    ):
        prediction = prediction_row.get("predict")
        label = prediction_row.get("label")
        reference = _assistant_reference(dataset_row)
        if not all(isinstance(value, str) for value in (prediction, label, reference)):
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
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in collisions)
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
        expected_dense = sum(REWARD_WEIGHTS[key] * float(alignment_row[key]) for key in REWARD_WEIGHTS)
        if not _close(alignment_row["dense_reward"], expected_dense):
            return False
    task_alignment = metrics.get("task_alignment", {})
    for key, metric_key in mean_fields.items():
        if not _close(task_alignment.get(metric_key), math.fsum(values[key]) / EXPECTED_EVAL_SAMPLES):
            return False
    return bool(
        task_alignment.get("strict_success") == strict_success
        and _close(task_alignment.get("strict_success_rate"), strict_success / EXPECTED_EVAL_SAMPLES)
    )


def _manifest_sections(run: Run) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    output = _evaluation_path(run)
    metrics = _load_json(output / "metrics.json")
    if metrics is None:
        return None
    identity = {
        "experiment": run.experiment,
        "variant": run.variant,
        "trace_variant": run.variant,
        "expected_samples": EXPECTED_EVAL_SAMPLES,
        "generated_predictions": _jsonl_artifact(run.predict_output / "generated_predictions.jsonl"),
        "canonical_predictions": _jsonl_artifact(run.predict_output / "path_predictions.jsonl"),
        "scored": _jsonl_artifact(output / "scored.jsonl"),
        "alignment_dataset": _jsonl_artifact(ALIGNMENT_DATASET),
        "alignment_evaluator": _file_artifact(MS_SWIFT_EVALUATOR),
        "base_evaluation_manifest": _file_artifact(output / "evaluation_manifest.json"),
        "reward_weights": REWARD_WEIGHTS,
        "pose_tolerances": POSE_TOLERANCES,
    }
    artifacts = {
        "alignment_input": _jsonl_artifact(output / "alignment_input.jsonl"),
        "alignment": _jsonl_artifact(output / "alignment.jsonl"),
        "metrics_json": _file_artifact(output / "metrics.json"),
        "metrics_md": _file_artifact(output / "metrics.md"),
    }
    if any(value is None for value in (*identity.values(), *artifacts.values())):
        return None
    summary = {
        "structure_samples": metrics.get("structure", {}).get("samples"),
        "task_alignment_samples": metrics.get("task_alignment", {}).get("samples"),
        "condition_generation_samples": metrics.get("condition_generation", {}).get("samples"),
        "dense_reward_mean": metrics.get("task_alignment", {}).get("dense_reward_mean"),
        "strict_success": metrics.get("task_alignment", {}).get("strict_success"),
        "strict_success_rate": metrics.get("task_alignment", {}).get("strict_success_rate"),
    }
    return identity, artifacts, summary


def _valid_completed_at(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _evaluation_complete(run: Run) -> bool:
    output = _evaluation_path(run)
    manifest = _load_json(output / "alignment_manifest.json")
    sections = _manifest_sections(run)
    if not (
        manifest
        and sections
        and _canonical_ready(run)
        and _base_evaluation_complete(run)
        and _alignment_artifacts_consistent(run)
    ):
        return False
    identity, artifacts, summary = sections
    return bool(
        set(manifest) == {"schema_version", "status", "completed_at", "freeze", "identity", "artifacts", "summary"}
        and manifest.get("schema_version") == ALIGNMENT_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and _valid_completed_at(manifest.get("completed_at"))
        and manifest.get("freeze") in ({"mode": "verified_numeric_backfill"}, {"mode": "post_evaluation_freeze"})
        and manifest.get("identity") == identity
        and manifest.get("artifacts") == artifacts
        and manifest.get("summary") == summary
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


def _command(run: Run, action: str, execute: bool = False) -> list[str]:
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
        *_conda_run_prefix(),
        "llamafactory-cli",
        "train",
        str(config.relative_to(ROOT)),
    ]
    if action == "train":
        command.append("gradient_accumulation_steps=16")
    return command


def main() -> None:
    args = parse_args()
    run = RUNS[args.run]
    blockers: list[str] = []
    checks: dict[str, Any] = {"run": run.experiment, "action": args.action, "gpus": args.gpus}
    _validate_text250k_adapter(checks, blockers)
    _validate_datasets(checks, blockers)
    _validate_audit("train10k", ("NonThinking-Control", "Thinking-Hard-V2-Lean-State"), checks, blockers)
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
        checks["text250k_approved"] = args.text250k_approved
        if not args.text250k_approved:
            blockers.append("TEXT250K_APPROVAL_REQUIRED_FOR_TRAIN")
    gpu_processes = _gpu1_compute_processes()
    checks["gpu1_compute_processes"] = gpu_processes
    checks["gpu1_occupancy_check_passed"] = gpu_processes == []
    if gpu_processes is None:
        blockers.append("GPU1_OCCUPANCY_UNKNOWN_FAIL_CLOSED")
    elif gpu_processes:
        blockers.append("GPU1_COMPUTE_OCCUPIED_NO_AUTO_KILL")
    if args.action in ("predict", "evaluate"):
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
        raise SystemExit("Text250k-only downstream launch blocked; resolve the reported gates first")
    if output_complete:
        print(f"Already complete; safe no-op: {output}")
        return
    env = os.environ.copy()
    for name in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        env.pop(name, None)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.action == "train" and not _adapter_complete(run.train_output):
        raise SystemExit("Training command returned without a complete adapter")
    if args.action == "predict" and not _prediction_complete(run.predict_output):
        raise SystemExit("Prediction command returned without 512 complete rows")
    if args.action == "evaluate" and not _evaluation_complete(run):
        raise SystemExit("Evaluation command returned without a complete VAL512 alignment manifest")


if __name__ == "__main__":
    main()
