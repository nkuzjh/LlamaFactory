#!/usr/bin/env python3
"""Fail-closed launcher for the rowbal-cont3 ep3 downstream arms.

``exp4_4_3`` and ``exp4_7_3`` are deliberately separate from the Text250k
downstream launcher.  Both arms bind the exact rowbal-cont3
``checkpoint-50646`` endpoint, then train a fresh Stage-2 LoRA adapter on one
of the two canonical 10k datasets.  The endpoint is a user-approved
sensitivity arm; it must not be silently replaced by the rowbal point-estimate
recommendation (``checkpoint-33764``) or by an alias.

The default is a read-only dry-run.  Training execution additionally requires
``--rowbal-ep3-approved``.  The downstream arms are fixed to physical CUDA 0
and reject an occupied or unknown GPU state; they never kill processes.  No
training, prediction, or evaluation subprocess is invoked without
``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/data/jiahao/task/BrickNet")
CONFIG_ROOT = ROOT / "examples/train_lora"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
DATASET_REGISTRY = ROOT / "data/dataset_info.json"
EVALUATION_ROOT = BRICKNET_ROOT / "outputs_val/qwen35_08b"
# The rowbal parent was trained on physical GPU 1, but these downstream arms
# intentionally run on physical GPU 0. Keep the identities separate so a
# downstream device migration cannot rewrite historical parent provenance.
EXPECTED_GPU = "0"
ROWBAL_PARENT_EXPECTED_GPU = "1"
EXPECTED_VAL_SAMPLES = 512
MIN_FREE_GIB = 40
EXPECTED_SFT_STEPS_PER_EPOCH = 625
EXPECTED_SFT_MAX_STEPS = 1_875

ROWBAL_PARENT_OUTPUT_NAME = (
    "train_PT_exp2_mm_rowbal_cont3_qwen35_08b_text8m250k_mm135051_text135051_ep3_bs2_gbs16_lora64_len6400"
)
ROWBAL_PARENT_OUTPUT = SAVE_ROOT / ROWBAL_PARENT_OUTPUT_NAME
ROWBAL_PARENT = ROWBAL_PARENT_OUTPUT / "checkpoint-50646"
ROWBAL_RUN_MANIFEST = Path(str(ROWBAL_PARENT_OUTPUT) + ".run_manifest.json")
ROWBAL_PARENT_RELATIVE = str(ROWBAL_PARENT.relative_to(ROOT))
ROWBAL_RUN_MANIFEST_SHA256 = "0aa2efbbb31f921e3640ed45b2a44f48354b141bfbd5acdb3ee81f61aa01d349"
ROWBAL_PARENT_FILE_HASHES = {
    "adapter_config.json": "d7f51df4e20760c1dae948a39abda8223fb91ae64fe1bb150c8074af4f92e3f8",
    "adapter_model.safetensors": "caaffbb3d8c1fc1dcd79f77b4762f8aebede9a627aab5f97598edfed77cdad12",
    "optimizer.pt": "6502f63de70a2f58816c0f82637aa14a92f4e755b2194525c67942d6616f2b0f",
    "scheduler.pt": "1fc60bbd6b8f8d39446cf362de872b1a2643b0fba6e230e1470d977d5d67805d",
    "rng_state.pth": "f4a9f217e852f439efa6bd32fde98d6867f11aa6ea13ddc021ba10af6a0b0934",
    "trainer_state.json": "c47293315d4582439ff95aa7cacd2c0cf3fd3a2c2a6208005e97de7400a0d56b",
    "training_args.bin": "a235e923311db022391f634b434e9c19c6f4f68bf4b35a75f661007f2f4b5c9a",
}

ROWBAL_DATA_ROOT = ROOT / "data/bricknet_pt_exp2_mm_rowbal_cont3"
ROWBAL_DATA_MANIFEST = ROWBAL_DATA_ROOT / "manifest.v1.json"
ROWBAL_DATA_FILE = ROWBAL_DATA_ROOT / "PT-exp2-mm-rowbal-cont3.jsonl"
ROWBAL_DATA_MANIFEST_SHA256 = "b2d77a977ded33ba27bdd0fa3985c2a46dee1eb176500741521d06e8b73bc100"
ROWBAL_DATASET_SHA256 = "ac55229c93c548008051f138b6587676f020d7f54c0426208c49163b8d285d84"
ROWBAL_ORDERED_ID_SHA256 = "2bc8bb9381ac397135fed382fe18abd753692948137d54d65b709a99bf4c838b"

STAGE2_DATASETS = {
    "control_train": {
        "registry_key": "BrickNet-Stage2-NonThinking-Control-10k",
        "path": ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl",
        "count": 10_000,
        "sha256": "4be6a7fb711ba24a658fc3096a8f1e2a9aa3e630c76adb8f04313bf9bff02a15",
        "ordered_id_sha256": "2d87ff4c3b918f748dde48721cbec66595ccc17317cf728f77e30efc04230dea",
    },
    "lean_train": {
        "registry_key": "BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        "path": ROOT / "data/bricknet_stage2_v2/10k/BrickNet-Stage2-ThinkingHard-V2-LeanState.jsonl",
        "count": 10_000,
        "sha256": "b0ee6b1046aaef6290ed7bb4d1b632c0260fbbd65048619c415e8669f9a6bc95",
        "ordered_id_sha256": "2d87ff4c3b918f748dde48721cbec66595ccc17317cf728f77e30efc04230dea",
    },
    "control_eval": {
        "registry_key": "BrickNet-Stage2-NonThinking-Control-VAL512-Eval",
        "path": BRICKNET_ROOT
        / "outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl",
        "count": 512,
        "sha256": "9692d80e13995969937b6cd36780e35905ce244abd0ff0e3f8b5eaa34de76775",
        "ordered_id_sha256": "908489739b3fff8489877da01b034081f80c5db8c21f357164c7ac38e02a6e51",
    },
    "lean_eval": {
        "registry_key": "BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval",
        "path": BRICKNET_ROOT
        / "outputs_preprocess/BrickNet-MM-Reasoning/stage2_v2/validation/datasets/BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval.jsonl",
        "count": 512,
        "sha256": "f102e74a2462e38af0cfcbcd9fd012772c7b3c0bc6fc44f2746430d57eec1009",
        "ordered_id_sha256": "908489739b3fff8489877da01b034081f80c5db8c21f357164c7ac38e02a6e51",
    },
}

EXPECTED_CONFIG_SHA256 = {
    (
        "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml"
    ): "a5971ef5c8ef281db459ab6492f59f9add3f4bfd0ada56208352d9ecd7c7b966",
    (
        "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml"
    ): "756a45f13a32e2b5d7c98007c699313266a3f1467d6dc0818855ca857363315b",
    (
        "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml"
    ): "eb51a07b75c106e658218612bebff6347b1140a2165a3693c9d8d26127dcaf82",
    (
        "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml"
    ): "51560c987238e05ccb24c00bfe247280481786afb234e08f5f5f1f34a44d6420",
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
    "exp4_4_3": Run(
        experiment="exp4_4_3",
        variant="nonthinking-control",
        train_dataset="BrickNet-Stage2-NonThinking-Control-10k",
        eval_dataset="BrickNet-Stage2-NonThinking-Control-VAL512-Eval",
        train_file=STAGE2_DATASETS["control_train"]["path"],
        eval_file=STAGE2_DATASETS["control_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_4_3_qwen35_08b_PT_exp2_mm_rowbal_cont3_ep3_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_4_3_PT_exp2_mm_rowbal_cont3_ep3_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_4_3-PT_exp2_mm_rowbal_cont3-ep3-nonthinking-control-10k-qwen35-08b-len16384",
        train_audit_key="NonThinking-Control",
        eval_audit_key="NonThinking-Control-VAL512",
    ),
    "exp4_7_3": Run(
        experiment="exp4_7_3",
        variant="thinking-hard-v2-lean-state",
        train_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        eval_dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval",
        train_file=STAGE2_DATASETS["lean_train"]["path"],
        eval_file=STAGE2_DATASETS["lean_eval"]["path"],
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_7_3_qwen35_08b_PT_exp2_mm_rowbal_cont3_ep3_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_7_3_PT_exp2_mm_rowbal_cont3_ep3_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20",
        cache_path=ROOT
        / ".llamafactory_cache/tokenized_dataset/exp4_7_3-PT_exp2_mm_rowbal_cont3-ep3-thinking-hard-v2-lean-state-10k-qwen35-08b-len16384",
        train_audit_key="Thinking-Hard-V2-Lean-State",
        eval_audit_key="Thinking-Hard-V2-Lean-State-VAL512",
    ),
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load launcher helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the existing strict Stage-2 artifact/evaluator checks without
# modifying the shared launchers.  These imports are read-only and do not run
# their ``main`` functions.
_STAGE2_SHARED = _load_module(
    ROOT / "scripts/launch_bricknet_pt_exp2_text250k_downstream.py",
    "_bricknet_stage2_text250k_shared_for_rowbal_downstream",
)
_ROWBAL_SHARED = _load_module(
    ROOT / "scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py",
    "_bricknet_rowbal_parent_shared_for_downstream",
)


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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--action", choices=("train", "predict", "evaluate"), required=True)
    parser.add_argument(
        "--gpus",
        required=True,
        choices=(EXPECTED_GPU,),
        help="This downstream contract is fixed to physical CUDA 0.",
    )
    parser.add_argument(
        "--rowbal-ep3-approved",
        action="store_true",
        help="Explicit approval required to execute a rowbal ep3 train action.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Invoke the selected command after all gates pass. Default is dry-run.",
    )
    return parser.parse_args()


def _validate_rowbal_parent(checks: dict[str, Any], blockers: list[str]) -> None:
    checks["rowbal_parent"] = str(ROWBAL_PARENT)
    checks["rowbal_parent_relative"] = ROWBAL_PARENT_RELATIVE
    checks["rowbal_parent_is_dir"] = ROWBAL_PARENT.is_dir()
    checks["rowbal_parent_output_is_symlink"] = ROWBAL_PARENT_OUTPUT.is_symlink()
    checks["rowbal_parent_is_symlink"] = ROWBAL_PARENT.is_symlink()
    if not checks["rowbal_parent_is_dir"]:
        blockers.append("ROWBAL_EP3_PARENT_CHECKPOINT_MISSING")
    if checks["rowbal_parent_output_is_symlink"]:
        blockers.append("ROWBAL_EP3_PARENT_OUTPUT_ALIAS_SYMLINK_FORBIDDEN")
    if checks["rowbal_parent_is_symlink"]:
        blockers.append("ROWBAL_EP3_PARENT_CHECKPOINT_SYMLINK_FORBIDDEN")

    file_checks: dict[str, Any] = {}
    for name, expected in ROWBAL_PARENT_FILE_HASHES.items():
        actual = _sha256(ROWBAL_PARENT / name)
        row = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": actual == expected,
            "nonempty": (ROWBAL_PARENT / name).is_file() and (ROWBAL_PARENT / name).stat().st_size > 0,
        }
        file_checks[name] = row
        if not row["sha256_matches"]:
            blockers.append(f"ROWBAL_EP3_PARENT_{name.upper().replace('.', '_')}_HASH_DRIFT")
    checks["rowbal_parent_files"] = file_checks

    state = _load_json(ROWBAL_PARENT / "trainer_state.json")
    checks["rowbal_parent_trainer_state"] = {
        "epoch": state.get("epoch") if state else None,
        "global_step": state.get("global_step") if state else None,
        "max_steps": state.get("max_steps") if state else None,
        "num_train_epochs": state.get("num_train_epochs") if state else None,
        "train_batch_size": state.get("train_batch_size") if state else None,
    }
    state_ok = bool(
        state
        and state.get("epoch") == 3.0
        and state.get("global_step") == 50_646
        and state.get("max_steps") == 50_646
        and state.get("num_train_epochs") == 3
        and state.get("train_batch_size") == 2
    )
    checks["rowbal_parent_trainer_state_ok"] = state_ok
    if not state_ok:
        blockers.append("ROWBAL_EP3_PARENT_TRAINER_STATE_DRIFT")

    manifest = _load_json(ROWBAL_RUN_MANIFEST)
    manifest_hash = _sha256(ROWBAL_RUN_MANIFEST)
    checks["rowbal_run_manifest"] = {
        "path": str(ROWBAL_RUN_MANIFEST),
        "expected_sha256": ROWBAL_RUN_MANIFEST_SHA256,
        "actual_sha256": manifest_hash,
        "sha256_matches": manifest_hash == ROWBAL_RUN_MANIFEST_SHA256,
        "payload": manifest,
    }
    if manifest is None:
        blockers.append("ROWBAL_RUN_MANIFEST_MISSING_OR_INVALID")
        return
    if manifest_hash != ROWBAL_RUN_MANIFEST_SHA256:
        blockers.append("ROWBAL_RUN_MANIFEST_HASH_DRIFT")
    expected_manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "PT-exp2-mm-rowbal-cont3",
        "config": str(CONFIG_ROOT / "qwen35_08b_bricknet_pt_exp2_mm_rowbal_cont3.yaml"),
        "config_sha256": "830ecb6d04c5ccf55e750d599755ab360d88f309caba434b447b993c4c988529",
        "dataset": "BrickNet-PT-exp2-mm-rowbal-cont3",
        "data_manifest": str(ROWBAL_DATA_MANIFEST),
        "data_manifest_sha256": ROWBAL_DATA_MANIFEST_SHA256,
        "dataset_sha256": ROWBAL_DATASET_SHA256,
        "ordered_id_sha256": ROWBAL_ORDERED_ID_SHA256,
        "rows": 270_102,
        "global_batch_size": 16,
        # This is the GPU used by the already-completed parent run, not the
        # physical device selected for the downstream action.
        "gpus": [ROWBAL_PARENT_EXPECTED_GPU],
        "checkpoint_steps": [16_882, 33_764, 50_646],
        "optimizer_scheduler_rng_required": True,
    }
    manifest_checks = {}
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        matches = actual == expected
        manifest_checks[key] = {"expected": expected, "actual": actual, "matches": matches}
        if not matches:
            blockers.append(f"ROWBAL_RUN_MANIFEST_{key.upper()}_DRIFT")
    parent_decl = manifest.get("parent_adapter")
    parent_hash = manifest.get("parent_adapter_sha256")
    expected_parent = str(
        SAVE_ROOT / "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack"
    )
    parent_ok = parent_decl == expected_parent and parent_hash == (
        "a5ec2be5d96a8beb54a4b53e0b1816626fae3cd2cd5da717917804204f5685bd"
    )
    manifest_checks["parent_adapter"] = {
        "expected": expected_parent,
        "actual": parent_decl,
        "matches": parent_ok,
        "sha256": parent_hash,
    }
    if not parent_ok:
        blockers.append("ROWBAL_RUN_MANIFEST_PARENT_PROVENANCE_DRIFT")
    checks["rowbal_run_manifest_fields"] = manifest_checks


def _validate_rowbal_data(checks: dict[str, Any], blockers: list[str]) -> None:
    """Run the immutable full-pool rowbal data/provenance validator read-only."""
    try:
        rowbal_blockers, rowbal_checks = _ROWBAL_SHARED._validate_data()
    except Exception as exc:  # validation failures are blockers, never bypasses
        blockers.append(f"ROWBAL_DATA_VALIDATION_EXCEPTION:{type(exc).__name__}")
        checks["rowbal_data"] = {"exception": str(exc)}
        return
    checks["rowbal_data"] = rowbal_checks
    blockers.extend(rowbal_blockers)

    # The parent run manifest and the data validator must converge on the
    # same bytes.  Checking only the hashes declared inside manifest.v1 would
    # allow a rewritten manifest plus rewritten declarations to look
    # internally consistent.  These three values are the frozen provenance
    # anchors recorded with the completed rowbal run.
    actual_anchors = {
        "manifest_sha256": rowbal_checks.get("manifest_sha256"),
        "content_sha256_actual": rowbal_checks.get("content_sha256_actual"),
        "ordered_id_sha256_actual": rowbal_checks.get("ordered_id_sha256_actual"),
    }
    expected_anchors = {
        "manifest_sha256": ROWBAL_DATA_MANIFEST_SHA256,
        "content_sha256_actual": ROWBAL_DATASET_SHA256,
        "ordered_id_sha256_actual": ROWBAL_ORDERED_ID_SHA256,
    }
    checks["rowbal_data_actual_anchors"] = {
        key: {
            "expected": expected_anchors[key],
            "actual": actual_anchors[key],
            "matches": actual_anchors[key] == expected_anchors[key],
        }
        for key in expected_anchors
    }
    for key, expected in expected_anchors.items():
        if actual_anchors[key] != expected:
            blockers.append(f"ROWBAL_DATA_ACTUAL_{key.upper()}_DRIFT")


def _validate_stage2_data(checks: dict[str, Any], blockers: list[str]) -> None:
    try:
        dataset_checks: dict[str, Any] = {}
        dataset_blockers: list[str] = []
        _STAGE2_SHARED._validate_datasets(dataset_checks, dataset_blockers)
        checks["stage2_datasets"] = dataset_checks
        blockers.extend(f"STAGE2_{item}" for item in dataset_blockers)
        for report_name, expected_names in (
            ("train10k", ("NonThinking-Control", "Thinking-Hard-V2-Lean-State")),
            ("eval_val512", ("NonThinking-Control-VAL512", "Thinking-Hard-V2-Lean-State-VAL512")),
        ):
            report_blockers: list[str] = []
            report_checks: dict[str, Any] = {}
            _STAGE2_SHARED._validate_audit(report_name, expected_names, report_checks, report_blockers)
            checks[f"stage2_{report_name}"] = report_checks
            blockers.extend(f"STAGE2_{item}" for item in report_blockers)
    except Exception as exc:
        blockers.append(f"STAGE2_DATA_VALIDATION_EXCEPTION:{type(exc).__name__}")
        checks["stage2_data_exception"] = str(exc)


def _expected_config(run: Run, action: str) -> dict[str, Any]:
    common = {
        "model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "trust_remote_code": True,
        "flash_attn": "auto",
        "stage": "sft",
        "finetuning_type": "lora",
        "dataset_dir": "data",
        "media_dir": str(BRICKNET_ROOT),
        "template": "qwen3_5_nothink",
        "enable_thinking": False,
        "cutoff_len": 16_384,
    }
    if action == "train":
        return {
            **common,
            "adapter_name_or_path": ROWBAL_PARENT_RELATIVE,
            "disable_gradient_checkpointing": False,
            "do_train": True,
            "create_new_adapter": True,
            "lora_target": "all",
            "lora_rank": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "freeze_vision_tower": True,
            "freeze_multi_modal_projector": True,
            "dataset": run.train_dataset,
            "train_on_prompt": False,
            "packing": False,
            "preprocessing_num_workers": 16,
            "dataloader_num_workers": 4,
            "tokenized_path": _relative(run.cache_path),
            "image_max_pixels": 589_824,
            "image_min_pixels": 1024,
            "video_max_pixels": 65_536,
            "video_min_pixels": 256,
            "output_dir": _relative(run.train_output),
            "logging_steps": 10,
            "save_steps": 250,
            "plot_loss": True,
            "report_to": "none",
            "include_num_input_tokens_seen": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5.0e-5,
            "num_train_epochs": 3.0,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "optim": "adamw_torch",
            "bf16": True,
            "ddp_timeout": 180_000_000,
            "ddp_find_unused_parameters": False,
            "seed": 42,
        }
    return {
        **common,
        "adapter_name_or_path": f"{ROWBAL_PARENT_RELATIVE},{_relative(run.train_output)}",
        "do_predict": True,
        "predict_with_generate": True,
        "eval_dataset": run.eval_dataset,
        "preprocessing_num_workers": 16,
        "per_device_eval_batch_size": 1,
        "seed": 42,
        "image_max_pixels": 589_824,
        "image_min_pixels": 1024,
        "video_max_pixels": 65_536,
        "video_min_pixels": 256,
        "output_dir": _relative(run.predict_output),
        "report_to": "none",
        "max_new_tokens": 16_384,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
    }


def _validate_config(run: Run, action: str, checks: dict[str, Any], blockers: list[str]) -> None:
    path = run.train_config if action == "train" else run.predict_config
    checks["config"] = str(path)
    if not path.is_file():
        blockers.append("CONFIG_MISSING")
        return
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        config = None
    if not isinstance(config, dict):
        blockers.append("CONFIG_INVALID_YAML")
        return
    expected_hash = EXPECTED_CONFIG_SHA256.get(path.name)
    actual_hash = _sha256(path)
    checks["config_sha256"] = {
        "expected": expected_hash,
        "actual": actual_hash,
        "matches": actual_hash == expected_hash,
    }
    if expected_hash is None or actual_hash != expected_hash:
        blockers.append("CONFIG_FILE_HASH_DRIFT")
    expected = _expected_config(run, action)
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    checks["config_fields"] = {"expected": expected, "mismatches": mismatches}
    if mismatches:
        blockers.append("CONFIG_CONTRACT_DRIFT")
    binding = config.get("adapter_name_or_path")
    if not isinstance(binding, str) or ROWBAL_PARENT_RELATIVE not in binding:
        blockers.append("ROWBAL_EP3_PARENT_BINDING_MISSING")
    if isinstance(binding, str) and any(
        token in binding.lower() for token in ("pt-exp2-v2", "pt_exp2_v2", "mm_e1", "mm_e2", "mm_e3")
    ):
        blockers.append("FORBIDDEN_PT_EXP2_ALIAS_OR_OLD_MM_ADAPTER")


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
    return bool(
        isinstance(state.get("global_step"), int)
        and not isinstance(state.get("global_step"), bool)
        and isinstance(state.get("max_steps"), int)
        and not isinstance(state.get("max_steps"), bool)
        and state.get("global_step") == EXPECTED_SFT_MAX_STEPS
        and state.get("max_steps") == EXPECTED_SFT_MAX_STEPS
        and isinstance(state.get("num_train_epochs"), (int, float))
        and not isinstance(state.get("num_train_epochs"), bool)
        and math.isclose(float(state.get("num_train_epochs")), 3.0, rel_tol=0.0, abs_tol=1e-12)
        and isinstance(state.get("epoch"), (int, float))
        and not isinstance(state.get("epoch"), bool)
        and math.isclose(float(state.get("epoch")), 3.0, rel_tol=0.0, abs_tol=1e-6)
        and state.get("train_batch_size") == 1
    )


def _prediction_complete(path: Path) -> bool:
    return _STAGE2_SHARED._prediction_complete(path)


def _evaluation_path(run: Run) -> Path:
    return EVALUATION_ROOT / run.predict_output.name


def _evaluation_complete(run: Run) -> bool:
    return _STAGE2_SHARED._evaluation_complete(run)


def _gpu0_compute_processes() -> list[str] | None:
    """Return physical GPU-0 compute rows, or None when unknown.

    Keep this query local to the CUDA-0 launcher.  The shared Stage-2 helper
    is intentionally hard-coded for physical GPU 1 and must not be reused for
    this migrated downstream contract.
    """
    try:
        uuid_result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"],
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


def _output_state(run: Run, action: str) -> tuple[Path, bool, bool]:
    if action == "train":
        path = run.train_output
        return path, path.exists(), _adapter_complete(path)
    if action == "predict":
        path = run.predict_output
        return path, path.exists(), _prediction_complete(path)
    path = _evaluation_path(run)
    return path, path.exists(), _evaluation_complete(run)


def _command(run: Run, action: str, execute: bool = False) -> list[str]:
    return _STAGE2_SHARED._command(run, action, execute)


def _disk_gate(checks: dict[str, Any], blockers: list[str]) -> None:
    try:
        usage = shutil.disk_usage(ROOT)
    except OSError as exc:
        checks["disk"] = {"error": str(exc)}
        blockers.append("DISK_USAGE_UNKNOWN_FAIL_CLOSED")
        return
    free_gib = usage.free / (1024**3)
    checks["disk"] = {"free_gib": free_gib, "minimum_free_gib": MIN_FREE_GIB}
    if free_gib < MIN_FREE_GIB:
        blockers.append("DISK_FREE_SPACE_BELOW_40_GIB")


def _single_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = EXPECTED_GPU
    for name in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        env.pop(name, None)
    return env


def main() -> None:
    args = parse_args()
    run = RUNS[args.run]
    blockers: list[str] = []
    checks: dict[str, Any] = {
        "run": run.experiment,
        "action": args.action,
        "gpus": args.gpus,
        "execution_authorized": bool(args.execute),
    }

    _validate_rowbal_parent(checks, blockers)
    _validate_rowbal_data(checks, blockers)
    _validate_stage2_data(checks, blockers)
    _validate_config(run, args.action, checks, blockers)
    _disk_gate(checks, blockers)

    output, output_exists, output_complete = _output_state(run, args.action)
    checks["output"] = str(output)
    checks["output_exists"] = output_exists
    checks["output_complete"] = output_complete
    checks["output_action_safe_noop"] = output_complete
    if output_exists and not output_complete:
        blockers.append("OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED")

    if args.action == "train":
        checks["rowbal_ep3_approved"] = args.rowbal_ep3_approved
        if not args.rowbal_ep3_approved:
            blockers.append("ROWBAL_EP3_APPROVAL_REQUIRED_FOR_TRAIN")
    else:
        checks["train_adapter"] = str(run.train_output)
        checks["train_adapter_complete"] = _adapter_complete(run.train_output)
        if not checks["train_adapter_complete"]:
            blockers.append("WAIT_FINAL_SFT_TRAIN_ADAPTER")
    if args.action == "evaluate":
        checks["prediction_output"] = str(run.predict_output)
        checks["prediction_output_complete"] = _prediction_complete(run.predict_output)
        if not checks["prediction_output_complete"]:
            blockers.append("WAIT_FINAL_SFT_PREDICTION")

    gpu_processes = _gpu0_compute_processes()
    checks["gpu0_compute_processes"] = gpu_processes
    checks["gpu0_occupancy_check_passed"] = gpu_processes == []
    if gpu_processes is None:
        blockers.append("GPU0_OCCUPANCY_UNKNOWN_FAIL_CLOSED")
    elif gpu_processes:
        blockers.append("GPU0_COMPUTE_OCCUPIED_NO_AUTO_KILL")

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
        raise SystemExit("rowbal-cont3 downstream launch blocked; resolve the reported gates first")
    if output_complete:
        print(f"Already complete; safe no-op: {output}")
        return

    subprocess.run(command, cwd=ROOT, env=_single_gpu_env(), check=True)
    if args.action == "train" and not _adapter_complete(run.train_output):
        raise SystemExit("Training command returned without a complete SFT adapter")
    if args.action == "predict" and not _prediction_complete(run.predict_output):
        raise SystemExit("Prediction command returned without 512 complete rows")
    if args.action == "evaluate" and not _evaluation_complete(run):
        raise SystemExit("Evaluation command returned without a complete VAL512 alignment manifest")


if __name__ == "__main__":
    main()
