#!/usr/bin/env python3
"""Gate-protected launcher and selector for the BrickNet PT-exp2 branch.

The default is a read-only status/dry run.  Training, prediction, evaluation,
50k materialization, alias creation, and scale approval all require explicit
``--execute``; final selection and scale approvals additionally require
``--approve``.  A fresh text8m run requires an absent output root.  Resuming it
requires both ``--resume-from-checkpoint`` and ``--resume-approved`` and accepts
only a validated full-state checkpoint inside that root.  No PT-exp2 VAL511
train or validation run exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


# Load the reviewed text8m contract by its adjacent path.  Importing it as
# ``scripts.*`` is not reliable because scripts/ is intentionally not a Python
# package, while importing it by basename fails when this launcher itself is
# loaded through importlib (as the focused tests do).
_TEXT8M_CONTRACT_PATH = Path(__file__).resolve().with_name("launch_bricknet_pt_exp2_unconditional.py")
_TEXT8M_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_bricknet_pt_exp2_text8m_contract", _TEXT8M_CONTRACT_PATH
)
if _TEXT8M_CONTRACT_SPEC is None or _TEXT8M_CONTRACT_SPEC.loader is None:
    raise ImportError(f"cannot load text8m contract: {_TEXT8M_CONTRACT_PATH}")
_text8m_contract = importlib.util.module_from_spec(_TEXT8M_CONTRACT_SPEC)
sys.modules.setdefault(_TEXT8M_CONTRACT_SPEC.name, _text8m_contract)
_TEXT8M_CONTRACT_SPEC.loader.exec_module(_text8m_contract)


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
CONFIG_ROOT = ROOT / "examples/train_lora"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
DATA_ROOT = ROOT / "data/bricknet_pt_exp2"
GATE_ROOT = DATA_ROOT / "gates"
TEXT_MANIFEST = DATA_ROOT / "text8m/manifest.json"
TEXT_VIEW = DATA_ROOT / "text8m_train"
TEXT_COUNT_FAILURE = DATA_ROOT / "text8m.building/count_gate_failure.json"
MM_MANIFEST = DATA_ROOT / "mm/manifest.json"
MM_TOKEN_AUDIT_ROOT = BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-PT-exp2/reports/token_audit_mm6400"
MM_EPOCH_EXPECTED = {
    "e1": {"rows": 150_668, "text_replay_rows": 15_617, "text_replay_target_tokens": 26_285_287},
    "e2": {"rows": 150_637, "text_replay_rows": 15_586, "text_replay_target_tokens": 26_285_922},
    "e3": {"rows": 150_718, "text_replay_rows": 15_667, "text_replay_target_tokens": 26_284_707},
}
FINAL_ALIAS = SAVE_ROOT / "PT-exp2"
STAGE2_PREP = BRICKNET_ROOT / "data_preprocess/prepare_bricknet_stage2_sft.py"
EVALUATOR = BRICKNET_ROOT / "scripts/evaluate_experiment.py"
SUPPORTED_TRAIN_WORLD_SIZES = (1, 2)
LENGTH_CACHE_BUILDER = ROOT / "scripts/build_tokenized_cache_with_length.py"
DEFAULT_CACHE_BATCH_SIZE = 10_000
DEFAULT_CACHE_NUM_PROC = 4

# The root filesystem on the training host is intentionally small.  Keep the
# complete text8m runtime (including temporary files created by datasets and
# the model stack) on the data disk.  These are fixed, non-secret paths so
# they are safe to show in the read-only launcher report.
TEXT8M_DATA_DISK_ROOT = Path("/data/jiahao")
TEXT8M_HF_CACHE_ROOT = TEXT8M_DATA_DISK_ROOT / ".cache/huggingface"
TEXT8M_XDG_CACHE_ROOT = TEXT8M_DATA_DISK_ROOT / ".cache"
TEXT8M_TMP_ROOT = TEXT8M_DATA_DISK_ROOT / "tmp/pt-exp2-text8m"
TEXT8M_RUNTIME_PATHS = {
    "HF_HOME": TEXT8M_HF_CACHE_ROOT,
    "HF_HUB_CACHE": TEXT8M_HF_CACHE_ROOT / "hub",
    # Keep the legacy spelling pinned too; different installed HF versions
    # consult different aliases.
    "HUGGINGFACE_HUB_CACHE": TEXT8M_HF_CACHE_ROOT / "hub",
    "HF_DATASETS_CACHE": TEXT8M_HF_CACHE_ROOT / "datasets",
    "HF_ASSETS_CACHE": TEXT8M_HF_CACHE_ROOT / "assets",
    "HF_MODULES_CACHE": TEXT8M_HF_CACHE_ROOT / "modules",
    "TRANSFORMERS_CACHE": TEXT8M_HF_CACHE_ROOT / "hub",
    "XDG_CACHE_HOME": TEXT8M_XDG_CACHE_ROOT,
    "TRITON_CACHE_DIR": TEXT8M_XDG_CACHE_ROOT / "triton",
    "TORCH_EXTENSIONS_DIR": TEXT8M_XDG_CACHE_ROOT / "torch_extensions_pt_exp2",
    "TORCH_HOME": TEXT8M_XDG_CACHE_ROOT / "torch",
    "NUMBA_CACHE_DIR": TEXT8M_XDG_CACHE_ROOT / "numba",
    "TMPDIR": TEXT8M_TMP_ROOT,
    "TMP": TEXT8M_TMP_ROOT,
    "TEMP": TEXT8M_TMP_ROOT,
}
TEXT8M_RUNTIME_FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
TEXT8M_RUNTIME_ENV_KEYS = (*TEXT8M_RUNTIME_PATHS, *TEXT8M_RUNTIME_FLAGS)
TEXT8M_EXPECTED_STEPS = _text8m_contract.EXPECTED_TRAIN_STEPS
TEXT8M_ADAPTER_CONFIG_CONTRACT = {
    "peft_type": "LORA",
    "r": 64,
    "lora_alpha": 128,
    "base_model_name_or_path": _text8m_contract.BASE_MODEL,
}
TEXT8M_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-([1-9][0-9]*)$")
TEXT8M_RESUME_STATE_FILES = (
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)


@dataclass(frozen=True)
class Run:
    config: str
    output: str
    kind: str
    prerequisite: str | None = None
    dataset: str | None = None
    eval_name: str | None = None


RUNS: dict[str, Run] = {
    "text8m": Run(
        "qwen35_08b_bricknet_pt_exp2_text8m.yaml",
        "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack",
        "text",
    ),
    "mm-e1": Run(
        "qwen35_08b_bricknet_pt_exp2_mm_e1.yaml",
        "train_PT_exp2_mm_e1_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400",
        "mm",
        prerequisite="text8m",
        dataset="PT-exp2-mm-e1.jsonl",
        eval_name="eval_PT_exp2_mm_e1_ptval_in4096_out4096_p95_t1_k20",
    ),
    "mm-e2": Run(
        "qwen35_08b_bricknet_pt_exp2_mm_e2.yaml",
        "train_PT_exp2_mm_e2_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400",
        "mm",
        prerequisite="mm-e1",
        dataset="PT-exp2-mm-e2.jsonl",
        eval_name="eval_PT_exp2_mm_e2_ptval_in4096_out4096_p95_t1_k20",
    ),
    "mm-e3": Run(
        "qwen35_08b_bricknet_pt_exp2_mm_e3.yaml",
        "train_PT_exp2_mm_e3_qwen35_08b_text8m_mm135k_replay1to1_ep1_bs2_gbs16_lora64_len6400",
        "mm",
        prerequisite="mm-e2",
        dataset="PT-exp2-mm-e3.jsonl",
        eval_name="eval_PT_exp2_mm_e3_ptval_in4096_out4096_p95_t1_k20",
    ),
    "exp4_4": Run(
        "qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_10k_pt_exp2.yaml",
        "train_exp4_4_qwen35_08b_PT_exp2_stage2_nonthinking_control_10k_ep3_bs1_gbs16_lora64_len16384",
        "downstream",
        dataset="/home/jiahao/task/LlamaFactory/data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl",
        eval_name="eval_exp4_4_PT_exp2_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
    ),
    "exp4_5": Run(
        "qwen35_08b_bricknet_stage2_exp4_5_nonthinking_control_50k_pt_exp2.yaml",
        "train_exp4_5_qwen35_08b_PT_exp2_stage2_nonthinking_control_50k_ep3_bs1_gbs16_lora64_len16384",
        "downstream",
        prerequisite="exp4_4",
        dataset="/home/jiahao/task/LlamaFactory/data/bricknet_stage2/50k/BrickNet-Stage2-NonThinking-Control.jsonl",
        eval_name="eval_exp4_5_PT_exp2_nonthinking_control_50k_val512_in16384_out16384_p95_t1_k20",
    ),
    "exp4_6": Run(
        "qwen35_08b_bricknet_stage2_exp4_6_nonthinking_control_all_pt_exp2.yaml",
        "train_exp4_6_qwen35_08b_PT_exp2_stage2_nonthinking_control_all66456_ep3_bs1_gbs16_lora64_len16384",
        "downstream",
        prerequisite="exp4_5",
        dataset="/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Reasoning/datasets/BrickNet-MM-NonThinking-Control.jsonl",
        eval_name="eval_exp4_6_PT_exp2_nonthinking_control_all66456_val512_in16384_out16384_p95_t1_k20",
    ),
}


PREDICT_CONFIGS = {
    "mm-e1": "qwen35_08b_bricknet_pt_exp2_mm_e1_predict.yaml",
    "mm-e2": "qwen35_08b_bricknet_pt_exp2_mm_e2_predict.yaml",
    "mm-e3": "qwen35_08b_bricknet_pt_exp2_mm_e3_predict.yaml",
    "exp4_4": "qwen35_08b_bricknet_stage2_exp4_4_nonthinking_control_predict_pt_exp2.yaml",
    "exp4_5": "qwen35_08b_bricknet_stage2_exp4_5_nonthinking_control_predict_pt_exp2.yaml",
    "exp4_6": "qwen35_08b_bricknet_stage2_exp4_6_nonthinking_control_predict_pt_exp2.yaml",
}

TRAIN_RUNS = {"text8m", "mm-e1", "mm-e2", "mm-e3", "exp4_4", "exp4_5", "exp4_6"}
TRAIN_BATCH_TARGETS = {
    "text8m": {"per_device_batch_size": 16, "global_batch_size": 32},
    "mm-e1": {"per_device_batch_size": 2, "global_batch_size": 16},
    "mm-e2": {"per_device_batch_size": 2, "global_batch_size": 16},
    "mm-e3": {"per_device_batch_size": 2, "global_batch_size": 16},
    "exp4_4": {"per_device_batch_size": 1, "global_batch_size": 16},
    "exp4_5": {"per_device_batch_size": 1, "global_batch_size": 16},
    "exp4_6": {"per_device_batch_size": 1, "global_batch_size": 16},
}


def _train_batch_profile(run_name: str, world_size: int) -> dict[str, int]:
    target = TRAIN_BATCH_TARGETS[run_name]
    divisor = target["per_device_batch_size"] * world_size
    if target["global_batch_size"] % divisor:
        raise ValueError(f"global batch for {run_name} is not divisible by world size {world_size}")
    return {
        **target,
        "world_size": world_size,
        "gradient_accumulation_steps": target["global_batch_size"] // divisor,
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalize_target_modules(value: Any) -> tuple[list[str] | None, str | None]:
    """Normalize a YAML/JSON target-module value while retaining duplicates."""
    if isinstance(value, str):
        items: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return None, "TARGET_MODULES_NOT_LIST_OR_COMMA_STRING"
    if not all(isinstance(item, str) for item in items):
        return None, "TARGET_MODULES_CONTAINS_NON_STRING"
    normalized = [item.strip() for item in items]
    if not normalized or any(not item for item in normalized):
        return None, "TARGET_MODULES_CONTAINS_EMPTY_NAME"
    if len(normalized) != len(set(normalized)):
        return normalized, "TARGET_MODULES_CONTAINS_DUPLICATE_NAME"
    return normalized, None


def _text8m_target_modules_contract(
    config_path: Path | None = None,
) -> tuple[dict[str, Any], frozenset[str], list[str]]:
    """Read the exact text8m target set from the frozen training YAML."""
    if config_path is None:
        config_path = CONFIG_ROOT / RUNS["text8m"].config
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        loaded = None
    config = loaded if isinstance(loaded, dict) else {}
    raw = config.get("lora_target")
    normalized, parse_error = _normalize_target_modules(raw)
    expected = frozenset(normalized or ())
    eligible = bool(normalized and parse_error is None and len(normalized) == len(expected))
    blockers = [] if eligible else ["TEXT8M_TARGET_MODULES_CONFIG_DRIFT"]
    checks = {
        "config": str(config_path),
        "yaml_key": "lora_target",
        "yaml_value": raw,
        "normalized": sorted(expected),
        "parse_error": parse_error,
        "eligible": eligible,
    }
    return checks, expected, blockers


def _target_modules_check(
    value: Any,
    expected: frozenset[str],
    *,
    expected_eligible: bool,
) -> dict[str, Any]:
    """Compare adapter target modules by exact normalized set and cardinality."""
    normalized, parse_error = _normalize_target_modules(value)
    actual = frozenset(normalized or ())
    eligible = bool(
        expected_eligible
        and normalized is not None
        and parse_error is None
        and len(normalized) == len(expected)
        and actual == expected
    )
    return {
        "value": value,
        "normalized": sorted(actual),
        "parse_error": parse_error,
        "expected": sorted(expected),
        "eligible": eligible,
    }


def _strict_file_check(path: Path) -> dict[str, Any]:
    """Describe one path without following symlinks for completion gates."""
    try:
        metadata = path.lstat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "symlink": False,
            "regular_file": False,
            "size": None,
            "nonempty": False,
            "eligible": False,
        }

    symlink = stat.S_ISLNK(metadata.st_mode)
    regular_file = stat.S_ISREG(metadata.st_mode)
    size = metadata.st_size if regular_file else None
    nonempty = regular_file and size > 0
    return {
        "path": str(path),
        "exists": True,
        "symlink": symlink,
        "regular_file": regular_file,
        "size": size,
        "nonempty": nonempty,
        "eligible": regular_file and not symlink and nonempty,
    }


def _strict_json_object(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read a non-symlink, non-empty regular JSON object and its file check."""
    file_check = _strict_file_check(path)
    value: dict[str, Any] | None = None
    parse_error: str | None = None
    if file_check["eligible"]:
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_error = type(exc).__name__
        else:
            if isinstance(decoded, dict):
                value = decoded
            else:
                parse_error = "JSON_NOT_OBJECT"
    file_check["valid_json_object"] = value is not None
    if parse_error is not None:
        file_check["parse_error"] = parse_error
    return value, file_check


def _finite_number(value: Any) -> bool:
    """Return whether a JSON number is finite, excluding bool-as-int."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _text8m_completion_check(path: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Validate the semantic 250k endpoint, not just PEFT file presence."""
    try:
        output_metadata = path.lstat()
    except OSError:
        output_metadata = None
    output_is_symlink = bool(output_metadata and stat.S_ISLNK(output_metadata.st_mode))
    output_is_directory = bool(output_metadata and stat.S_ISDIR(output_metadata.st_mode))

    adapter_config_path = path / "adapter_config.json"
    adapter_config, adapter_config_file = _strict_json_object(adapter_config_path)
    target_contract, expected_target_modules, target_contract_blockers = _text8m_target_modules_contract(config_path)
    target_modules = _target_modules_check(
        adapter_config.get("target_modules") if adapter_config else None,
        expected_target_modules,
        expected_eligible=not target_contract_blockers,
    )
    adapter_contract_ok = bool(
        adapter_config is not None
        and all(adapter_config.get(key) == expected for key, expected in TEXT8M_ADAPTER_CONFIG_CONTRACT.items())
        and target_modules["eligible"]
    )

    adapter_weight_candidates = {
        name: _strict_file_check(path / name) for name in ("adapter_model.safetensors", "adapter_model.bin")
    }
    present_weight_files = [name for name, item in adapter_weight_candidates.items() if item["exists"]]
    eligible_weight_files = [name for name, item in adapter_weight_candidates.items() if item["eligible"]]
    adapter_weights_ok = len(present_weight_files) == 1 and len(eligible_weight_files) == 1

    trainer_state_path = path / "trainer_state.json"
    trainer_state, trainer_state_file = _strict_json_object(trainer_state_path)
    global_step = trainer_state.get("global_step") if trainer_state else None
    max_steps = trainer_state.get("max_steps") if trainer_state else None
    trainer_state_ok = bool(
        trainer_state is not None
        and isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and isinstance(max_steps, int)
        and not isinstance(max_steps, bool)
        and global_step == TEXT8M_EXPECTED_STEPS
        and max_steps == TEXT8M_EXPECTED_STEPS
    )
    train_results_path = path / "train_results.json"
    train_results, train_results_file = _strict_json_object(train_results_path)
    train_loss = train_results.get("train_loss") if train_results else None
    train_loss_ok = _finite_number(train_loss)

    checks = {
        "path": str(path),
        "output_directory": {
            "exists": output_metadata is not None,
            "symlink": output_is_symlink,
            "real_directory": output_is_directory and not output_is_symlink,
            "eligible": output_is_directory and not output_is_symlink,
        },
        "adapter_config": {
            **adapter_config_file,
            "peft_type": adapter_config.get("peft_type") if adapter_config else None,
            "r": adapter_config.get("r") if adapter_config else None,
            "lora_alpha": adapter_config.get("lora_alpha") if adapter_config else None,
            "base_model_name_or_path": (
                adapter_config.get("base_model_name_or_path") if adapter_config else None
            ),
            "expected": TEXT8M_ADAPTER_CONFIG_CONTRACT.copy(),
            "contract_eligible": adapter_contract_ok,
            "target_modules_contract": target_contract,
            "target_modules": target_modules,
        },
        "adapter_weights": {
            "candidates": adapter_weight_candidates,
            "present": bool(present_weight_files),
            "files": present_weight_files,
            "eligible_files": eligible_weight_files,
            "exactly_one_non_symlink_nonempty_file": adapter_weights_ok,
            "eligible": adapter_weights_ok,
        },
        "trainer_state": {
            **trainer_state_file,
            "global_step": global_step,
            "max_steps": max_steps,
            "expected_steps": TEXT8M_EXPECTED_STEPS,
            "ok": trainer_state_ok,
        },
        "train_results": {
            **train_results_file,
            "train_loss": train_loss,
            "finite_train_loss": train_loss_ok,
            "ok": bool(train_results is not None and train_loss_ok),
        },
    }
    checks["eligible"] = bool(
        checks["output_directory"]["eligible"]
        and adapter_config_file["eligible"]
        and adapter_contract_ok
        and not target_contract_blockers
        and adapter_weights_ok
        and trainer_state_file["eligible"]
        and trainer_state_ok
        and train_results_file["eligible"]
        and train_results is not None
        and train_loss_ok
    )
    return checks


def _text8m_base_contract_check(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Bind text8m training to the reviewed base snapshot and tokenizer."""
    blockers: list[str] = []
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        loaded = None
    config = loaded if isinstance(loaded, dict) else {}
    configured_model = config.get("model_name_or_path")
    configured_revision = config.get("model_revision")
    config_check = {
        "path": str(config_path),
        "model_name_or_path": configured_model,
        "expected_model_name_or_path": _text8m_contract.BASE_MODEL,
        "model_revision": configured_revision,
        "expected_model_revision": _text8m_contract.BASE_REVISION,
        "eligible": (
            configured_model == _text8m_contract.BASE_MODEL and configured_revision == _text8m_contract.BASE_REVISION
        ),
    }
    if configured_model != _text8m_contract.BASE_MODEL:
        blockers.append("TEXT8M_BASE_MODEL_CONFIG_DRIFT")
    if configured_revision != _text8m_contract.BASE_REVISION:
        blockers.append("TEXT8M_BASE_REVISION_CONFIG_DRIFT")

    snapshot, snapshot_blockers = _text8m_contract._check_snapshot()
    tokenizer, tokenizer_blockers = _text8m_contract._check_tokenizer_semantics()
    blockers.extend(snapshot_blockers)
    blockers.extend(tokenizer_blockers)
    checks = {
        "contract_source": str(Path(_text8m_contract.__file__).resolve()),
        "config": config_check,
        "snapshot": snapshot,
        "tokenizer_semantics": tokenizer,
        "eligible": not blockers,
    }
    return checks, list(dict.fromkeys(blockers))


def _text8m_output_contract_check(config_path: Path, output: Path) -> tuple[dict[str, Any], list[str]]:
    """Keep both fresh and resumed text8m launches on the one reviewed root."""
    blockers: list[str] = []
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        loaded = None
    config = loaded if isinstance(loaded, dict) else {}
    configured_output = config.get("output_dir")
    configured_path: Path | None = None
    if isinstance(configured_output, str) and configured_output:
        configured_path = Path(configured_output).expanduser()
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
        configured_path = Path(os.path.abspath(configured_path))
    expected_path = Path(os.path.abspath(output))
    output_ok = configured_path == expected_path
    overwrite_value = config.get("overwrite_output_dir", False)
    overwrite_ok = overwrite_value is False
    configured_resume = config.get("resume_from_checkpoint")
    embedded_resume_ok = configured_resume is None
    if not output_ok:
        blockers.append("TEXT8M_OUTPUT_DIR_CONFIG_DRIFT")
    if not overwrite_ok:
        blockers.append("TEXT8M_OVERWRITE_FORBIDDEN")
    if not embedded_resume_ok:
        blockers.append("TEXT8M_CONFIG_EMBEDDED_RESUME_FORBIDDEN")
    checks = {
        "config": str(config_path),
        "configured_output_dir": configured_output,
        "configured_output_resolved": str(configured_path) if configured_path else None,
        "expected_output_dir": str(expected_path),
        "output_dir_eligible": output_ok,
        "overwrite_output_dir": overwrite_value,
        "overwrite_forbidden": True,
        "resume_from_checkpoint_in_config": configured_resume,
        "embedded_resume_forbidden": True,
        "eligible": not blockers,
    }
    return checks, blockers


def _resume_state_file_check(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    symlink = path.is_symlink()
    size = path.stat().st_size if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "symlink": symlink,
        "size": size,
        "eligible": bool(exists and not symlink and size is not None and size > 0),
    }


def _text8m_resume_checkpoint_check(
    checkpoint_value: str | Path,
    *,
    output: Path,
    world_size: int | None,
    config_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one explicitly named, in-root, full-state text8m checkpoint."""
    blockers: list[str] = []
    raw = str(checkpoint_value)
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    requested = Path(os.path.abspath(requested))
    expected_output = Path(os.path.abspath(output))
    match = TEXT8M_CHECKPOINT_PATTERN.fullmatch(requested.name)
    step = int(match.group(1)) if match else None
    direct_child = requested.parent == expected_output
    checks: dict[str, Any] = {
        "requested": raw,
        "checkpoint": str(requested),
        "expected_output": str(expected_output),
        "direct_checkpoint_child": direct_child,
        "checkpoint_name_valid": match is not None,
        "checkpoint_step": step,
        "expected_max_steps": TEXT8M_EXPECTED_STEPS,
        "world_size": world_size,
    }
    if not direct_child:
        blockers.append("TEXT8M_RESUME_CHECKPOINT_OUTSIDE_EXPECTED_OUTPUT")
    if match is None:
        blockers.append("TEXT8M_RESUME_CHECKPOINT_NAME_INVALID")
    elif not (0 < step < TEXT8M_EXPECTED_STEPS):
        blockers.append("TEXT8M_RESUME_CHECKPOINT_STEP_OUT_OF_RANGE")

    output_is_dir = expected_output.is_dir()
    output_is_symlink = expected_output.is_symlink()
    checkpoint_is_dir = requested.is_dir()
    checkpoint_is_symlink = requested.is_symlink()
    checks["output_directory"] = {
        "exists": output_is_dir,
        "symlink": output_is_symlink,
    }
    checks["checkpoint_directory"] = {
        "exists": checkpoint_is_dir,
        "symlink": checkpoint_is_symlink,
    }
    if not output_is_dir:
        blockers.append("TEXT8M_RESUME_OUTPUT_MISSING")
    elif output_is_symlink:
        blockers.append("TEXT8M_RESUME_OUTPUT_SYMLINK_FORBIDDEN")
    if not checkpoint_is_dir:
        blockers.append("TEXT8M_RESUME_CHECKPOINT_MISSING")
    elif checkpoint_is_symlink:
        blockers.append("TEXT8M_RESUME_CHECKPOINT_SYMLINK_FORBIDDEN")

    # Resolve only after rejecting symlinked roots.  This second containment
    # check catches a direct-looking path whose ancestors escape the run root.
    resolved_contained = False
    if output_is_dir and checkpoint_is_dir and not output_is_symlink and not checkpoint_is_symlink:
        try:
            resolved_contained = requested.resolve(strict=True).parent == expected_output.resolve(strict=True)
        except OSError:
            resolved_contained = False
    checks["resolved_direct_checkpoint_child"] = resolved_contained
    if checkpoint_is_dir and not resolved_contained:
        blockers.append("TEXT8M_RESUME_CHECKPOINT_RESOLVES_OUTSIDE_OUTPUT")

    if not checkpoint_is_dir:
        checks["eligible"] = False
        return checks, list(dict.fromkeys(blockers))

    state_files = {name: _resume_state_file_check(requested / name) for name in TEXT8M_RESUME_STATE_FILES}
    adapter_weight_candidates = {
        name: _resume_state_file_check(requested / name) for name in ("adapter_model.safetensors", "adapter_model.bin")
    }
    present_adapter_weights = [
        name for name, item in adapter_weight_candidates.items() if item["exists"] or item["symlink"]
    ]
    eligible_adapter_weights = [name for name, item in adapter_weight_candidates.items() if item["eligible"]]
    state_files["adapter_weights"] = {
        "candidates": adapter_weight_candidates,
        "present": present_adapter_weights,
        "selected": eligible_adapter_weights[0] if len(eligible_adapter_weights) == 1 else None,
        "eligible": len(present_adapter_weights) == 1 and len(eligible_adapter_weights) == 1,
    }
    checks["state_files"] = state_files
    missing = [name for name in TEXT8M_RESUME_STATE_FILES if not state_files[name]["eligible"]]
    if not state_files["adapter_weights"]["eligible"]:
        missing.append("adapter_weights")
    if missing:
        blockers.append(f"TEXT8M_RESUME_FULL_STATE_MISSING:{','.join(missing)}")

    adapter_config = _json_object(requested / "adapter_config.json")
    target_contract, expected_target_modules, target_contract_blockers = _text8m_target_modules_contract(config_path)
    target_modules = _target_modules_check(
        adapter_config.get("target_modules") if adapter_config else None,
        expected_target_modules,
        expected_eligible=not target_contract_blockers,
    )
    checks["adapter_config"] = {
        "valid_json_object": adapter_config is not None,
        "peft_type": adapter_config.get("peft_type") if adapter_config else None,
        "r": adapter_config.get("r") if adapter_config else None,
        "lora_alpha": adapter_config.get("lora_alpha") if adapter_config else None,
        "base_model_name_or_path": (adapter_config.get("base_model_name_or_path") if adapter_config else None),
        "target_modules_contract": target_contract,
        "target_modules": target_modules,
    }
    adapter_contract_ok = bool(
        adapter_config is not None
        and adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("r") == 64
        and adapter_config.get("lora_alpha") == 128
        and adapter_config.get("base_model_name_or_path") == _text8m_contract.BASE_MODEL
        and target_modules["eligible"]
        and not target_contract_blockers
    )
    checks["adapter_config"]["eligible"] = adapter_contract_ok
    if not adapter_contract_ok:
        blockers.append("TEXT8M_RESUME_ADAPTER_CONFIG_DRIFT")
    if not target_modules["eligible"] and not target_contract_blockers:
        blockers.append("TEXT8M_RESUME_TARGET_MODULES_DRIFT")
    if target_contract_blockers:
        blockers.extend(target_contract_blockers)

    trainer_state = _json_object(requested / "trainer_state.json")
    state_step = trainer_state.get("global_step") if trainer_state else None
    state_max_steps = trainer_state.get("max_steps") if trainer_state else None
    trainer_state_ok = bool(
        trainer_state is not None
        and state_step == step
        and state_max_steps == TEXT8M_EXPECTED_STEPS
        and isinstance(state_step, int)
        and not isinstance(state_step, bool)
    )
    checks["trainer_state"] = {
        "valid_json_object": trainer_state is not None,
        "global_step": state_step,
        "path_step": step,
        "max_steps": state_max_steps,
        "eligible": trainer_state_ok,
    }
    if not trainer_state_ok:
        blockers.append("TEXT8M_RESUME_TRAINER_STATE_DRIFT")

    if world_size == 1:
        expected_rng_names = ["rng_state.pth"]
    elif world_size == 2:
        expected_rng_names = ["rng_state_0.pth", "rng_state_1.pth"]
    else:
        expected_rng_names = []
        blockers.append("TEXT8M_RESUME_WORLD_SIZE_UNSUPPORTED")
    actual_rng_names = sorted(
        path.name for path in requested.glob("rng_state*.pth") if path.is_file() or path.is_symlink()
    )
    rng_files = {name: _resume_state_file_check(requested / name) for name in expected_rng_names}
    rng_ok = bool(
        expected_rng_names
        and actual_rng_names == expected_rng_names
        and all(item["eligible"] for item in rng_files.values())
    )
    checks["rng_state"] = {
        "expected_files": expected_rng_names,
        "actual_files": actual_rng_names,
        "files": rng_files,
        "eligible": rng_ok,
    }
    if not rng_ok:
        blockers.append("TEXT8M_RESUME_RNG_STATE_MISMATCH")

    checks["eligible"] = not blockers
    return checks, list(dict.fromkeys(blockers))


def _visible_gpu_selectors() -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [item.strip() for item in visible.split(",") if item.strip()]


def _selected_gpu_uuids() -> set[str] | None:
    selectors = _visible_gpu_selectors()
    if selectors is None:
        return None
    if not selectors:
        return set()

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Preserve the previous conservative behavior if physical GPU indices
        # cannot be resolved: inspect processes on every GPU.
        return None

    index_to_uuid = {}
    gpu_uuids = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            continue
        index, gpu_uuid = fields
        index_to_uuid[index] = gpu_uuid
        gpu_uuids.append(gpu_uuid)

    selected = set()
    for selector in selectors:
        if selector in index_to_uuid:
            selected.add(index_to_uuid[selector])
            continue
        selected.update(gpu_uuid for gpu_uuid in gpu_uuids if gpu_uuid.startswith(selector))
    return selected


def _gpu_processes() -> list[str]:
    selected_gpu_uuids = _selected_gpu_uuids()
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            continue
        gpu_uuid, pid, used_memory, process_name = fields
        if selected_gpu_uuids is not None and gpu_uuid not in selected_gpu_uuids:
            continue
        processes.append(f"{pid}, {used_memory}, {process_name}")
    return processes


def _prediction_dir(run: str) -> Path:
    return SAVE_ROOT / RUNS[run].eval_name


def _metrics_path(run: str) -> Path:
    return BRICKNET_ROOT / "outputs_val/qwen35_08b" / RUNS[run].eval_name / "metrics.json"


def _load_length_cache_builder() -> Any:
    spec = importlib.util.spec_from_file_location("_bricknet_tokenized_cache_with_length", LENGTH_CACHE_BUILDER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load length-cache builder: {LENGTH_CACHE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is contained by *root*, including root itself."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _text8m_runtime_env(*, create: bool = False) -> tuple[dict[str, str], dict[str, Any]]:
    """Build the text8m environment and optionally create its data-disk dirs.

    The non-creating path is used while producing the default dry-run report.
    It only resolves/stat-checks the fixed paths.  The creating path is called
    after all launch blockers have cleared, creates the directories, then
    resolves them again to catch a symlink/race that would redirect writes.
    """
    data_root = TEXT8M_DATA_DISK_ROOT.resolve(strict=False)
    data_mount = Path("/data").resolve(strict=False)
    configured_paths = {key: Path(value) for key, value in TEXT8M_RUNTIME_PATHS.items()}

    def inspect_paths() -> tuple[dict[str, Path], dict[str, dict[str, Any]], list[str]]:
        resolved_paths: dict[str, Path] = {}
        path_checks: dict[str, dict[str, Any]] = {}
        blockers: list[str] = []
        for key, configured in configured_paths.items():
            try:
                resolved = configured.expanduser().resolve(strict=False)
                resolve_error = None
            except (OSError, RuntimeError) as exc:
                resolved = configured
                resolve_error = type(exc).__name__
            resolved_paths[key] = resolved
            under_data = _path_is_within(resolved, data_root)
            under_mount = _path_is_within(resolved, data_mount)
            is_directory = configured.is_dir()
            check = {
                "configured": str(configured),
                "resolved": str(resolved),
                "under_data_disk": under_data,
                "under_data_mount": under_mount,
                "exists": configured.exists(),
                "directory": is_directory,
                "resolve_error": resolve_error,
                "eligible": resolve_error is None and under_data and under_mount,
            }
            path_checks[key] = check
            if not check["eligible"]:
                blockers.append(f"TEXT8M_RUNTIME_PATH_INVALID:{key}")
        return resolved_paths, path_checks, blockers

    resolved_paths, path_checks, blockers = inspect_paths()
    checks: dict[str, Any] = {
        "eligible": not blockers,
        "created": False,
        "data_disk_root": str(data_root),
        "data_mount": str(data_mount),
        "paths": path_checks,
        "effective_env": {
            **{key: str(value) for key, value in configured_paths.items()},
            **TEXT8M_RUNTIME_FLAGS,
        },
        "blockers": blockers,
    }
    if blockers:
        if create:
            raise RuntimeError(f"text8m runtime paths are unsafe: {blockers}")
        return os.environ.copy(), checks

    if create:
        # Resolve and validate before every write.  Deduplicating the paths
        # avoids redundant mkdir calls for HF_HOME and its subdirectories.
        for path in dict.fromkeys(resolved_paths.values()):
            path.mkdir(parents=True, exist_ok=True)
        resolved_paths, post_create_paths, post_create_blockers = inspect_paths()
        checks["post_create_paths"] = post_create_paths
        checks["created"] = True
        checks["eligible"] = not post_create_blockers
        checks["blockers"] = post_create_blockers
        if post_create_blockers:
            raise RuntimeError(f"text8m runtime paths changed after mkdir: {post_create_blockers}")

    # Use the verified, canonical paths for child processes.  In the current
    # layout these equal the configured paths; canonicalizing also prevents a
    # benign in-data symlink from making later checks ambiguous.
    child_env = os.environ.copy()
    child_env.update({key: str(resolved_paths[key]) for key in configured_paths})
    child_env.update(TEXT8M_RUNTIME_FLAGS)
    checks["effective_env"] = {
        **{key: child_env[key] for key in configured_paths},
        **TEXT8M_RUNTIME_FLAGS,
    }
    return child_env, checks


def _length_cache_check(config_path: Path, *, strict_text8m: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    length_column = config.get("length_column_name")
    if not length_column:
        return {
            "mode": "standard",
            "enabled": False,
            "eligible": True,
            "build_required": False,
        }

    tokenized_path = config.get("tokenized_path")
    if not tokenized_path:
        return {
            "mode": "with_length",
            "enabled": True,
            "eligible": False,
            "build_required": False,
            "error": "TOKENIZED_PATH_MISSING",
        }
    cache_path = Path(tokenized_path)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    info_path = cache_path / "train/dataset_info.json"
    result = {
        "mode": "with_length",
        "enabled": True,
        "eligible": False,
        "build_required": not cache_path.exists(),
        "path": str(cache_path),
        "length_column": length_column,
        "dataset_info": str(info_path),
    }
    if not cache_path.exists():
        result["error"] = "TOKENIZED_CACHE_MISSING"
        return result
    if strict_text8m:
        try:
            builder = _load_length_cache_builder()
            strict = builder.check_cache(
                cache_path,
                str(length_column),
                "input_ids",
                config_path=config_path,
                require_manifest=True,
                full_length_validation=True,
            )
        except Exception as exc:
            result["error"] = f"TOKENIZED_CACHE_STRICT_CHECK_FAILED:{type(exc).__name__}:{exc}"
            return result
        strict["mode"] = "with_length"
        strict["enabled"] = True
        strict["build_required"] = False
        strict["strict_text8m_contract"] = True
        if strict.get("train_rows") != 7_698_261:
            strict["eligible"] = False
            strict["error"] = "TOKENIZED_CACHE_ROW_COUNT_MISMATCH"
            strict["expected_train_rows"] = 7_698_261
        return strict
    if not info_path.is_file():
        result["error"] = "TOKENIZED_TRAIN_DATASET_INFO_MISSING"
        return result
    try:
        info = _json(info_path)
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"TOKENIZED_TRAIN_DATASET_INFO_INVALID: {exc}"
        return result
    features = info.get("features", {})
    result["columns"] = sorted(features)
    result["train_rows"] = info.get("splits", {}).get("train", {}).get("num_examples")
    if length_column not in features:
        result["error"] = "TOKENIZED_LENGTH_COLUMN_MISSING"
        return result
    result["eligible"] = True
    return result


def _length_cache_build_command(config_path: Path, args: argparse.Namespace | None = None) -> list[str]:
    batch_size = args.cache_batch_size if args is not None else DEFAULT_CACHE_BATCH_SIZE
    num_proc = args.cache_num_proc if args is not None else DEFAULT_CACHE_NUM_PROC
    return [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "python",
        str(LENGTH_CACHE_BUILDER),
        "--config",
        str(config_path),
        "--batch-size",
        str(batch_size),
        "--num-proc",
        str(num_proc),
    ]


def _prepare_length_cache(
    config_path: Path,
    args: argparse.Namespace,
    *,
    strict_text8m: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    before = _length_cache_check(config_path, strict_text8m=strict_text8m)
    result = {"before": before, "executed": False}
    if not before["enabled"] or before["eligible"]:
        result["action"] = "standard" if not before["enabled"] else "reuse"
        result["after"] = before
        return result
    if not before["build_required"]:
        raise RuntimeError(
            "configured tokenized cache exists but is missing a valid length column; "
            "migrate it to a different output path with build_tokenized_cache_with_length.py, "
            "then update tokenized_path"
        )

    command = _length_cache_build_command(config_path, args)
    result.update({"action": "build", "command": shlex.join(command)})
    print(json.dumps({"event": "prepare_length_cache", **result}, ensure_ascii=False, indent=2), flush=True)
    # The caller supplies the already validated text8m runtime environment.
    # Keep a copy here only to remove launcher/distributed control variables;
    # all HF/cache/offline values remain identical for cache construction and
    # the eventual training subprocess.
    cache_env = dict(env) if env is not None else os.environ.copy()
    for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        cache_env.pop(key, None)
    subprocess.run(command, cwd=ROOT, env=cache_env, check=True)
    after = _length_cache_check(config_path, strict_text8m=strict_text8m)
    result.update({"executed": True, "after": after})
    if not after["eligible"]:
        raise RuntimeError(f"with-length cache failed post-build validation: {after}")
    print(json.dumps({"event": "length_cache_ready", **result}, ensure_ascii=False, indent=2), flush=True)
    return result


def _text8m_prelaunch_cache_check(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Revalidate the strict cache contract in the final launch window."""
    try:
        cache = _length_cache_check(config_path, strict_text8m=True)
    except Exception as exc:
        cache = {
            "mode": "with_length",
            "enabled": True,
            "eligible": False,
            "error": f"TOKENIZED_CACHE_PRELAUNCH_CHECK_FAILED:{type(exc).__name__}:{exc}",
        }
    blockers: list[str] = []
    if not cache.get("eligible"):
        blockers.append("TEXT8M_PRELAUNCH_CACHE_INVALID")
    return cache, blockers


def _text8m_prelaunch_output_check(
    config_path: Path,
    output: Path,
    *,
    validated_resume: str | None,
    world_size: int | None,
) -> tuple[dict[str, Any], list[str]]:
    """Recheck the fresh/resume output contract immediately before launch."""
    output_contract, blockers = _text8m_output_contract_check(config_path, output)
    checks: dict[str, Any] = {
        "config_contract": output_contract,
        "mode": "resume" if validated_resume is not None else "fresh",
        "output": str(output),
    }
    try:
        output_metadata = output.lstat()
    except OSError:
        output_metadata = None

    if validated_resume is None:
        output_absent = output_metadata is None
        checks["fresh_output"] = {
            "entry_exists": output_metadata is not None,
            "symlink": bool(output_metadata and stat.S_ISLNK(output_metadata.st_mode)),
            "absent": output_absent,
        }
        if not output_absent:
            blockers.append("TEXT8M_FRESH_OUTPUT_APPEARED")
    else:
        resume_checks, resume_blockers = _text8m_resume_checkpoint_check(
            validated_resume,
            output=output,
            world_size=world_size,
            config_path=config_path,
        )
        checks["resume"] = resume_checks
        blockers.extend(resume_blockers)

    checks["eligible"] = not blockers
    return checks, list(dict.fromkeys(blockers))


def _check_common(run_name: str, action: str) -> tuple[list[str], dict[str, Any]]:
    run = RUNS[run_name]
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    config_name = PREDICT_CONFIGS[run_name] if action == "predict" else run.config
    config = CONFIG_ROOT / config_name
    checks["config"] = str(config)
    checks["config_exists"] = config.is_file()
    if not config.is_file():
        blockers.append("CONFIG_MISSING")

    if action == "train":
        if run.kind == "text":
            base_contract, base_blockers = _text8m_base_contract_check(config)
            checks["text8m_base_contract"] = base_contract
            blockers.extend(base_blockers)
        selectors = _visible_gpu_selectors()
        world_size = len(selectors) if selectors is not None else None
        checks["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        checks["distributed_world_size"] = world_size
        checks["supported_world_sizes"] = list(SUPPORTED_TRAIN_WORLD_SIZES)
        checks["batch_profile"] = (
            _train_batch_profile(run_name, world_size) if world_size in SUPPORTED_TRAIN_WORLD_SIZES else None
        )
        resolved_uuids = _selected_gpu_uuids()
        checks["resolved_gpu_count"] = len(resolved_uuids) if resolved_uuids is not None else None
        if selectors is None:
            blockers.append("SET_EXPLICIT_TRAIN_GPUS")
        elif world_size not in SUPPORTED_TRAIN_WORLD_SIZES:
            blockers.append("PT_EXP2_REQUIRES_1_OR_2_VISIBLE_GPUS")
        elif resolved_uuids is not None and len(resolved_uuids) != world_size:
            blockers.append("TRAIN_GPU_SELECTOR_UNRESOLVED_OR_DUPLICATE")
        checks["length_cache"] = _length_cache_check(config, strict_text8m=run.kind == "text")
        if checks["length_cache"]["enabled"]:
            checks["length_cache"]["prepare_command"] = shlex.join(_length_cache_build_command(config))
        if not checks["length_cache"]["eligible"] and not checks["length_cache"]["build_required"]:
            blockers.append("TOKENIZED_LENGTH_CACHE_MISSING_OR_INVALID")
        output = SAVE_ROOT / run.output
        checks["output"] = str(output)
        if run.kind == "text":
            output_contract, output_blockers = _text8m_output_contract_check(config, output)
            checks["text8m_output_contract"] = output_contract
            blockers.extend(output_blockers)
            completion = _text8m_completion_check(output, config)
            checks["text8m_completion"] = completion
            checks["already_complete"] = completion["eligible"]
        else:
            checks["already_complete"] = _adapter_ready(output)
        output_entry_exists = output.exists() or (run.kind == "text" and output.is_symlink())
        checks["output_entry_exists"] = output_entry_exists
        if output_entry_exists and not checks["already_complete"]:
            blockers.append("OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED")
        if checks["already_complete"]:
            blockers.append("RUN_ALREADY_COMPLETE")
        if run.kind == "text":
            if not TEXT_MANIFEST.is_file():
                if TEXT_COUNT_FAILURE.is_file():
                    failure = _json(TEXT_COUNT_FAILURE)
                    checks["text_count_failure"] = str(TEXT_COUNT_FAILURE)
                    checks["text_actual_unique_rows"] = failure.get("actual_unique_rows")
                    checks["text_expected_unique_rows"] = failure.get("expected_unique_rows")
                    blockers.append("WAIT_TEXT8M_FINALIZE_EXISTING_AND_AUDIT")
                else:
                    blockers.append("WAIT_TEXT8M_CORPUS_BUILD_AND_AUDIT")
            else:
                manifest = _json(TEXT_MANIFEST)
                checks["text_rows"] = manifest.get("stats", {}).get("unique_rows")
                audit = manifest.get("audit", {})
                checks["text_audit_eligible"] = audit.get("eligible") is True
                checks["text_collision_policy"] = audit.get("collision_audit_policy")
                checks["text_collision_failure_count"] = audit.get("collision_failure_count")
                if checks["text_rows"] != 7_698_261 or not checks["text_audit_eligible"]:
                    blockers.append("WAIT_TEXT8M_AUDIT_GATE")
            if not TEXT_VIEW.is_dir():
                blockers.append("WAIT_TEXT8M_TRAIN_VIEW")
        elif run.kind == "mm":
            mm_manifest = _json(MM_MANIFEST) if MM_MANIFEST.is_file() else {}
            if not mm_manifest.get("eligible"):
                blockers.append("WAIT_PT_EXP2_MM_DATA_GATE")
            elif not (MM_MANIFEST.parent / str(run.dataset)).is_file():
                blockers.append("PT_EXP2_MM_DATASET_MISSING")
            epoch = run_name.removeprefix("mm-")
            epoch_manifest = mm_manifest.get("epochs", {}).get(epoch, {})
            checks["mm_epoch"] = epoch
            checks["mm_dataset_rows"] = epoch_manifest.get("rows")
            checks["mm_replay_rows"] = epoch_manifest.get("text_replay_rows")
            checks["mm_replay_target_tokens"] = epoch_manifest.get("text_replay_target_tokens")
            checks["mm_replay_mm_ratio"] = epoch_manifest.get("text_to_mm_target_token_ratio")
            expected = MM_EPOCH_EXPECTED[epoch]
            checks["mm_expected"] = expected
            if (
                epoch_manifest.get("file") != run.dataset
                or epoch_manifest.get("multimodal_rows") != 135_051
                or epoch_manifest.get("multimodal_target_tokens") != 26_285_148
                or any(epoch_manifest.get(key) != value for key, value in expected.items())
            ):
                blockers.append("PT_EXP2_MM_EPOCH_MANIFEST_DRIFT")
            audit_report = MM_TOKEN_AUDIT_ROOT / epoch / "BrickNet-MM-Reasoning_token_audit_report.json"
            checks["mm_token_audit"] = str(audit_report)
            checks["mm_token_audit_eligible"] = (
                audit_report.is_file() and _json(audit_report).get("training_eligible") is True
            )
            if not checks["mm_token_audit_eligible"]:
                blockers.append("WAIT_PT_EXP2_MM_ZERO_TRUNCATION_AUDIT")
            previous = RUNS[run.prerequisite]
            if not _adapter_ready(SAVE_ROOT / previous.output):
                blockers.append(f"WAIT_{run.prerequisite.upper().replace('-', '_')}_ADAPTER")
        else:
            checks["final_alias"] = str(FINAL_ALIAS)
            checks["final_alias_ready"] = _adapter_ready(FINAL_ALIAS)
            if not checks["final_alias_ready"]:
                blockers.append("WAIT_PT_EXP2_FINAL_ALIAS")
            dataset = Path(str(run.dataset))
            checks["dataset"] = str(dataset)
            checks["dataset_exists"] = dataset.is_file()
            if not dataset.is_file():
                blockers.append("WAIT_DOWNSTREAM_DATASET_MATERIALIZATION")
            if run_name in {"exp4_5", "exp4_6"}:
                gate = GATE_ROOT / f"{run.prerequisite}-approved.json"
                checks["scale_gate"] = str(gate)
                if not gate.is_file():
                    blockers.append(f"WAIT_{run.prerequisite.upper()}_HUMAN_GATE")
    else:
        if not _adapter_ready(SAVE_ROOT / run.output):
            blockers.append("WAIT_TRAIN_ADAPTER")
        predictions = _prediction_dir(run_name) / "generated_predictions.jsonl"
        checks["prediction_output"] = str(predictions)
        if predictions.exists():
            blockers.append("PREDICTION_OUTPUT_ALREADY_EXISTS")

    gpu_processes = _gpu_processes()
    checks["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    checks["gpu_processes"] = gpu_processes
    # Temporarily disabled: report selected-GPU processes without blocking launch.
    # if gpu_processes:
    #     blockers.append("WAIT_GPU_AVAILABLE")
    return blockers, checks


def _run_train_or_predict(args: argparse.Namespace) -> None:
    if args.action == "train" and args.run not in TRAIN_RUNS:
        raise SystemExit(f"{args.run} has no training configuration")
    blockers, checks = _check_common(args.run, args.action)
    config_name = PREDICT_CONFIGS[args.run] if args.action == "predict" else RUNS[args.run].config
    config_path = CONFIG_ROOT / config_name
    resume_value = getattr(args, "resume_from_checkpoint", None)
    resume_approved = bool(getattr(args, "resume_approved", False))
    resume_requested = resume_value is not None or resume_approved
    validated_resume: str | None = None
    if resume_requested:
        resume_checks: dict[str, Any] = {
            "requested": str(resume_value) if resume_value is not None else None,
            "approved": resume_approved,
            "eligible": False,
        }
        resume_blockers: list[str] = []
        if args.action != "train" or args.run != "text8m":
            resume_blockers.append("TEXT8M_RESUME_FLAGS_SCOPE_INVALID")
        elif resume_value is None or not resume_approved:
            resume_blockers.append("TEXT8M_RESUME_REQUIRES_CHECKPOINT_AND_APPROVAL")
        else:
            selectors = _visible_gpu_selectors()
            world_size = len(selectors) if selectors is not None else None
            resume_checks, resume_blockers = _text8m_resume_checkpoint_check(
                resume_value,
                output=SAVE_ROOT / RUNS["text8m"].output,
                world_size=world_size,
                config_path=config_path,
            )
            resume_checks["approved"] = True
            if not resume_blockers:
                validated_resume = resume_checks["checkpoint"]
                # An incomplete output is required for an explicit resume.  It
                # is safe to remove this generic blocker only after every
                # checkpoint/full-state/containment check succeeds.
                blockers = [blocker for blocker in blockers if blocker != "OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED"]
        blockers.extend(resume_blockers)
        resume_checks["eligible"] = not resume_blockers
        checks["text8m_resume"] = resume_checks
    elif args.action == "train" and args.run == "text8m":
        checks["text8m_resume"] = {
            "requested": None,
            "approved": False,
            "eligible": False,
            "mode": "fresh_output_must_be_absent",
        }
    text8m_train = args.action == "train" and args.run == "text8m"
    runtime_env: dict[str, str] | None = None
    runtime_checks: dict[str, Any] | None = None
    if text8m_train:
        # This is deliberately the non-creating probe.  It makes the dry-run
        # report complete while keeping the default invocation read-only.
        runtime_env, runtime_checks = _text8m_runtime_env(create=False)
        checks["text8m_runtime_env"] = runtime_checks
        if not runtime_checks["eligible"]:
            blockers.extend(runtime_checks["blockers"])
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(config_path),
    ]
    if args.action == "train" and args.run == "text8m":
        # Keep overwrite disabled at both the reviewed YAML gate and the final
        # CLI precedence layer.  LlamaFactory may otherwise auto-detect an
        # existing checkpoint, which this launcher deliberately forbids.
        command.append("overwrite_output_dir=false")
        if validated_resume is not None:
            command.append(f"resume_from_checkpoint={validated_resume}")
    selectors = _visible_gpu_selectors()
    if args.action == "train" and selectors and len(selectors) in SUPPORTED_TRAIN_WORLD_SIZES:
        profile = _train_batch_profile(args.run, len(selectors))
        command.append(f"gradient_accumulation_steps={profile['gradient_accumulation_steps']}")
    env = runtime_env if runtime_env is not None else os.environ.copy()
    launch_env: dict[str, str] = {}
    if selectors:
        launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selectors)
    if args.action == "train" and selectors and len(selectors) > 1:
        launch_env.update(
            {
                "FORCE_TORCHRUN": "1",
                "NPROC_PER_NODE": str(len(selectors)),
                "NNODES": "1",
            }
        )
        checks["launch_mode"] = "torchrun_ddp"
    elif args.action == "train" and selectors:
        for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES"):
            env.pop(key, None)
        checks["launch_mode"] = "single_process"
    env.update(launch_env)
    display_command = command
    display_env: dict[str, str] = {}
    if text8m_train and runtime_checks is not None:
        display_env.update(runtime_checks["effective_env"])
    display_env.update(launch_env)
    if display_env:
        display_command = ["env", *[f"{key}={value}" for key, value in display_env.items()], *command]
    checks["launch_env"] = launch_env
    result = {
        "action": args.action,
        "run": args.run,
        "ready": not blockers,
        "blockers": blockers,
        "checks": checks,
        "command": shlex.join(display_command),
        "executed": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 launch blocked; resolve the reported gates first")
    if text8m_train:
        # Directory creation is intentionally after the blocker gate.  The
        # helper revalidates every resolved path after mkdir and returns the
        # exact environment that is then shared with cache construction and
        # training (apart from cache-builder removal of DDP-only variables).
        runtime_env, runtime_checks = _text8m_runtime_env(create=True)
        runtime_env.update(launch_env)
        checks["text8m_runtime_env"] = runtime_checks
        print(
            json.dumps(
                {
                    "event": "text8m_runtime_environment_ready",
                    "checks": runtime_checks,
                    "env": runtime_checks["effective_env"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        env = runtime_env
    if args.action == "train":
        _prepare_length_cache(
            config_path,
            args,
            strict_text8m=RUNS[args.run].kind == "text",
            env=env,
        )
        if RUNS[args.run].kind == "text":
            cache_checks, cache_blockers = _text8m_prelaunch_cache_check(config_path)
            print(
                json.dumps(
                    {
                        "event": "text8m_prelaunch_cache_validation",
                        "config": str(config_path),
                        "require_manifest": True,
                        "full_length_validation": True,
                        "checks": cache_checks,
                        "blockers": cache_blockers,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            if cache_blockers:
                raise SystemExit("text8m tokenized cache changed while preparing the launch")
            base_contract, base_blockers = _text8m_base_contract_check(config_path)
            print(
                json.dumps(
                    {
                        "event": "text8m_prelaunch_base_validation",
                        **base_contract,
                        "blockers": base_blockers,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            if base_blockers:
                raise SystemExit("text8m base contract changed while preparing the launch")
            selectors = _visible_gpu_selectors()
            world_size = len(selectors) if selectors is not None else None
            output_checks, output_blockers = _text8m_prelaunch_output_check(
                config_path,
                SAVE_ROOT / RUNS["text8m"].output,
                validated_resume=validated_resume,
                world_size=world_size,
            )
            print(
                json.dumps(
                    {
                        "event": "text8m_prelaunch_output_validation",
                        **output_checks,
                        "blockers": output_blockers,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            if output_blockers:
                raise SystemExit("text8m output contract changed while preparing the launch")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.action == "train" and RUNS[args.run].kind == "text":
        completion = _text8m_completion_check(SAVE_ROOT / RUNS[args.run].output, config_path)
        print(
            json.dumps(
                {"event": "text8m_completion_validation", **completion},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if not completion["eligible"]:
            raise SystemExit("text8m training exited without a valid 250000-step endpoint")


def _evaluate(args: argparse.Namespace) -> None:
    if args.run not in PREDICT_CONFIGS:
        raise SystemExit(f"{args.run} has no prediction/evaluation run")
    predictions = _prediction_dir(args.run) / "generated_predictions.jsonl"
    text_metrics = _prediction_dir(args.run) / "predict_results.json"
    output = _metrics_path(args.run).parent
    blockers = []
    if not predictions.is_file():
        blockers.append("WAIT_512_PREDICTIONS")
    if _metrics_path(args.run).is_file():
        blockers.append("EVALUATION_ALREADY_COMPLETE")
    gpu_processes = _gpu_processes()
    # Temporarily disabled: GPU occupancy no longer blocks image-metric evaluation.
    # if gpu_processes:
    #     blockers.append("WAIT_GPU_AVAILABLE_FOR_IMAGE_METRICS")
    command = [
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
    print(
        json.dumps(
            {
                "action": "evaluate",
                "run": args.run,
                "ready": not blockers,
                "blockers": blockers,
                "gpu_processes": gpu_processes,
                "command": shlex.join(command),
                "executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute:
        return
    if blockers:
        raise SystemExit("PT-exp2 evaluation blocked; resolve the reported gates first")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src")
    subprocess.run(command, cwd=BRICKNET_ROOT, env=env, check=True)


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
    for run_name in ("mm-e1", "mm-e2", "mm-e3"):
        metrics_path = _metrics_path(run_name)
        adapter = SAVE_ROOT / RUNS[run_name].output
        if not metrics_path.is_file():
            blockers.append(f"WAIT_{run_name.upper()}_METRICS")
            continue
        if not _adapter_ready(adapter):
            blockers.append(f"WAIT_{run_name.upper()}_ADAPTER")
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
        blockers.append("PT_EXP2_ALIAS_ALREADY_EXISTS")
    payload = {
        "action": "select-final",
        "ranking": "lexicographic: strict_success_rate, dense_reward_mean, clean_rate, parsable_rate",
        "candidates": candidates,
        "recommended": recommended,
        "alias": str(FINAL_ALIAS),
        "ready": not blockers,
        "blockers": blockers,
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if not args.approve:
        raise SystemExit("--approve is required to freeze the PT-exp2 alias")
    if blockers or recommended is None:
        raise SystemExit("PT-exp2 final selection is blocked")
    target = Path(candidates[recommended]["adapter"])
    FINAL_ALIAS.symlink_to(os.path.relpath(target, FINAL_ALIAS.parent), target_is_directory=True)
    GATE_ROOT.mkdir(parents=True, exist_ok=True)
    payload.update({"executed": True, "selected": recommended, "created_at": datetime.now(UTC).isoformat()})
    (GATE_ROOT / "PT-exp2-selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _materialize(args: argparse.Namespace) -> None:
    if args.run != "exp4_5":
        raise SystemExit("only exp4_5 requires materializing a new 50k dataset")
    gate = GATE_ROOT / "exp4_4-approved.json"
    output = Path(str(RUNS[args.run].dataset))
    blockers = []
    if not gate.is_file():
        blockers.append("WAIT_EXP4_4_HUMAN_GATE")
    if output.exists():
        blockers.append("50K_DATASET_ALREADY_EXISTS")
    command = [str(BRICKNET_PYTHON), str(STAGE2_PREP), "materialize", "--scale", "50k"]
    print(
        json.dumps(
            {
                "action": "materialize",
                "run": args.run,
                "ready": not blockers,
                "blockers": blockers,
                "command": shlex.join(command),
                "executed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute:
        return
    if blockers:
        raise SystemExit("50k materialization is blocked")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src")
    subprocess.run(command, cwd=BRICKNET_ROOT, env=env, check=True)


def _approve_scale(args: argparse.Namespace) -> None:
    if args.run not in {"exp4_4", "exp4_5"}:
        raise SystemExit("scale approval is only defined for exp4_4 and exp4_5")
    metrics = _metrics_path(args.run)
    gate = GATE_ROOT / f"{args.run}-approved.json"
    blockers = []
    if not metrics.is_file():
        blockers.append("WAIT_COMPLETE_METRICS")
    if gate.exists():
        blockers.append("APPROVAL_ALREADY_EXISTS")
    payload = {
        "action": "approve-scale",
        "run": args.run,
        "metrics": str(metrics),
        "metrics_sha256": _sha256(metrics) if metrics.is_file() else None,
        "decision_rule": "manual review that the preceding scale shows useful gain; no unfrozen numeric threshold",
        "ready": not blockers,
        "blockers": blockers,
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if not args.approve:
        raise SystemExit("--approve is required for a scale gate")
    if blockers:
        raise SystemExit("scale approval is blocked")
    GATE_ROOT.mkdir(parents=True, exist_ok=True)
    payload.update({"executed": True, "approved_at": datetime.now(UTC).isoformat()})
    gate.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("train", "predict", "evaluate", "select-final", "materialize", "approve-scale"),
        required=True,
    )
    parser.add_argument("--run", choices=tuple(RUNS), default="text8m")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH",
        help=("resume text8m only from an explicitly named checkpoint-N directly under its expected output directory"),
    )
    parser.add_argument(
        "--resume-approved",
        action="store_true",
        help="confirm explicit review of the full-state text8m resume checkpoint",
    )
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=DEFAULT_CACHE_BATCH_SIZE,
        help="rows per map batch when automatically building a with-length cache",
    )
    parser.add_argument(
        "--cache-num-proc",
        type=int,
        default=DEFAULT_CACHE_NUM_PROC,
        help="worker processes used by automatic with-length cache construction",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        metavar="GPU",
        help="physical CUDA indices/UUIDs to expose; PT-exp2 training supports one or two",
    )
    args = parser.parse_args()
    if args.gpus:
        selectors = [selector for value in args.gpus for selector in value.split(",") if selector]
        if len(selectors) != len(set(selectors)):
            parser.error("--gpus contains duplicate CUDA selectors")
        args.gpus = selectors
    if args.cache_batch_size < 1:
        parser.error("--cache-batch-size must be positive")
    if args.cache_num_proc < 1:
        parser.error("--cache-num-proc must be positive")
    if args.action == "predict" and args.run not in PREDICT_CONFIGS:
        parser.error(f"{args.run} has no predict configuration")
    resume_requested = args.resume_from_checkpoint is not None or args.resume_approved
    if resume_requested and (args.action != "train" or args.run != "text8m"):
        parser.error("resume flags are valid only with --action train --run text8m")
    if (args.resume_from_checkpoint is None) != (not args.resume_approved):
        parser.error("--resume-from-checkpoint and --resume-approved must be supplied together")
    return args


def main() -> None:
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    if args.action in {"train", "predict"}:
        _run_train_or_predict(args)
    elif args.action == "evaluate":
        _evaluate(args)
    elif args.action == "select-final":
        _select_final(args)
    elif args.action == "materialize":
        _materialize(args)
    else:
        _approve_scale(args)


if __name__ == "__main__":
    main()
