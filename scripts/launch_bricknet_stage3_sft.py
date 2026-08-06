#!/usr/bin/env python3
"""Dry-run/gate the dormant Stage-3 exp5 train or prediction command."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
STAGE3_ROOT = BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-Reasoning/stage3"
DATASET = STAGE3_ROOT / "datasets/BrickNet-Stage3-Thinking-Semantic-10k.jsonl"
VALIDATION_REPORT = STAGE3_ROOT / "reports/BrickNet-Stage3-Thinking-Semantic-10k_validation.json"
TOKEN_REPORT = STAGE3_ROOT / "reports/token_audit/10k/BrickNet-MM-Reasoning_token_audit_report.json"
DATASET_REGISTRY = ROOT / "data/dataset_info.json"
TRAIN_CONFIG = ROOT / "examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_10k.yaml"
PREDICT_CONFIG = ROOT / "examples/train_lora/qwen35_08b_bricknet_stage3_exp5_thinking_semantic_predict.yaml"
STAGE0_ADAPTER = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora"
    / "train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64"
)
EXP5_ADAPTER = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora"
    / "train_exp5_qwen35_08b_mixedpt_stage3_thinking_semantic_10k_ep3_bs1_ga16_lora64_len16384"
)
TRAIN_OUTPUT = EXP5_ADAPTER
PREDICT_OUTPUT = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora"
    / "eval_exp5_stage3_thinking_semantic_10k_val512_in16384_out16384_p95_t1_k20"
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
    )


def _approval(path: Path | None, kind: str) -> bool:
    value = _load(path) if path else None
    return bool(
        value
        and value.get("kind") == kind
        and value.get("approved") is True
        and "FILL_ME" not in json.dumps(value, ensure_ascii=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--action", choices=("train", "predict"), required=True)
    parser.add_argument("--stage0-adapter", type=Path, default=STAGE0_ADAPTER)
    parser.add_argument("--stage0-gate-approved", action="store_true")
    parser.add_argument("--t1-10k-gate", type=Path, default=None)
    parser.add_argument("--audit-approved", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    config = TRAIN_CONFIG if args.action == "train" else PREDICT_CONFIG
    output = TRAIN_OUTPUT if args.action == "train" else PREDICT_OUTPUT
    checks["config"] = str(config)
    checks["config_ready"] = config.is_file()
    if not checks["config_ready"]:
        blockers.append("exp5 config is missing")

    checks["stage0_adapter"] = str(args.stage0_adapter)
    checks["stage0_adapter_ready"] = _adapter_ready(args.stage0_adapter)
    if not checks["stage0_adapter_ready"]:
        blockers.append("WAIT_STAGE0_ADAPTER: mixed PT-exp1 final adapter is incomplete")
    checks["stage0_gate_approved"] = args.stage0_gate_approved
    if not args.stage0_gate_approved:
        blockers.append("WAIT_STAGE0_GATE: --stage0-gate-approved is required")

    checks["pilot_audit_approved"] = _approval(args.audit_approved, "stage3_pilot_audit")
    if not checks["pilot_audit_approved"]:
        blockers.append("WAIT_STAGE3_PILOT_AUDIT: approved pilot audit artifact is missing")
    checks["t1_10k_gate_approved"] = _approval(args.t1_10k_gate, "stage2_thinking_hard_10k_gate")
    if not checks["t1_10k_gate_approved"]:
        blockers.append("WAIT_STAGE2_T1_10K: approved exp4_3 comparison gate is missing")

    validation = _load(VALIDATION_REPORT)
    checks["t2_validation_report"] = str(VALIDATION_REPORT)
    checks["t2_validation_training_eligible"] = bool(
        validation and validation.get("training_eligible") is True
    )
    if not checks["t2_validation_training_eligible"]:
        blockers.append("WAIT_T2_VALIDATION: hard replay and token audit are incomplete")

    checks["dataset"] = str(DATASET)
    checks["dataset_ready"] = DATASET.is_file()
    checks["dataset_count"] = _lines(DATASET) if DATASET.is_file() else None
    if not checks["dataset_ready"] or checks["dataset_count"] != 10_000:
        blockers.append("T2-10k dataset is missing or not exactly 10,000 rows")

    token = _load(TOKEN_REPORT)
    t2_token = token.get("datasets", {}).get("Thinking-Semantic", {}) if token else {}
    checks["token_gate"] = bool(
        token
        and token.get("training_eligible") is True
        and token.get("zero_truncation") is True
        and token.get("paired_order_and_ids") is True
        and DATASET.is_file()
        and t2_token.get("sha256") == _sha256(DATASET)
    )
    if not checks["token_gate"]:
        blockers.append("T2 real Qwen processor token gate is incomplete")

    registry = _load(DATASET_REGISTRY) or {}
    entry = registry.get("BrickNet-Stage3-Thinking-Semantic-10k")
    registered = Path(entry.get("file_name", "")) if isinstance(entry, dict) else None
    checks["dataset_registry_active"] = bool(
        registered and registered.resolve() == DATASET.resolve()
    )
    if not checks["dataset_registry_active"]:
        blockers.append("WAIT_DATASET_FREEZE: dormant exp5 dataset registry entry is not activated")

    if args.action == "predict":
        checks["exp5_adapter_ready"] = _adapter_ready(EXP5_ADAPTER)
        if not checks["exp5_adapter_ready"]:
            blockers.append("WAIT_EXP5_TRAIN: exp5 adapter is incomplete")

    checks["output"] = str(output)
    checks["output_absent"] = not output.exists()
    if output.exists():
        blockers.append(f"output directory already exists: {output}")

    if args.action == "train":
        command = [
            "conda", "run", "-n", "llamafactory", "--no-capture-output",
            "llamafactory-cli", "train", str(config.relative_to(ROOT)),
            f"adapter_name_or_path={args.stage0_adapter}",
        ]
    else:
        command = [
            "conda", "run", "-n", "llamafactory", "--no-capture-output",
            "llamafactory-cli", "train", str(config.relative_to(ROOT)),
            f"adapter_name_or_path={args.stage0_adapter},{EXP5_ADAPTER}",
        ]
    result = {
        "stage": 3,
        "experiment": "exp5",
        "action": args.action,
        "mode": "execute" if args.execute else "dry-run",
        "checks": checks,
        "blockers": blockers,
        "ready": not blockers,
        "command": shlex.join(command),
        "training_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("Stage-3 exp5 launch blocked; resolve every reported gate")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
