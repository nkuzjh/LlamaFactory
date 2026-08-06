#!/usr/bin/env python3
"""Gate and print/launch one BrickNet Stage-2 train or prediction command.

The default mode is a dry-run. Training or prediction is possible only with
both ``--execute`` and ``--stage0-gate-approved`` after all artifact checks pass.
"""

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
REASONING_ROOT = BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-Reasoning"
STAGE0_ADAPTER = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora"
    / "train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64"
)
TOKEN_REPORT = REASONING_ROOT / "reports/token_audit/BrickNet-MM-Reasoning_token_audit_report.json"
VAL_REPORT = REASONING_ROOT / "validation/reports/BrickNet-Stage2-VAL511-Train-VAL512-Eval_report.json"
VAL_TRAIN_TOKEN_REPORT = (
    REASONING_ROOT / "validation/reports/token_audit/train_val511/BrickNet-MM-Reasoning_token_audit_report.json"
)
VAL_EVAL_TOKEN_REPORT = (
    REASONING_ROOT / "validation/reports/token_audit/eval_val512/BrickNet-MM-Reasoning_token_audit_report.json"
)

TRAIN_CONFIGS = {
    "nonthinking": ROOT / "examples/train_lora/qwen35_08b_bricknet_stage2_nonthinking.yaml",
    "thinking-hard": ROOT / "examples/train_lora/qwen35_08b_bricknet_stage2_thinking_hard.yaml",
}
PREDICT_CONFIGS = {
    "nonthinking": ROOT / "examples/train_lora/qwen35_08b_bricknet_stage2_nonthinking_predict.yaml",
    "thinking-hard": ROOT / "examples/train_lora/qwen35_08b_bricknet_stage2_thinking_hard_predict.yaml",
}
DATASETS = {
    ("nonthinking", "overfit511"): "BrickNet-Stage2-NonThinking-VAL511-Train",
    ("thinking-hard", "overfit511"): "BrickNet-Stage2-ThinkingHard-VAL511-Train",
    ("nonthinking", "10k"): "BrickNet-Stage2-NonThinking-10k",
    ("thinking-hard", "10k"): "BrickNet-Stage2-ThinkingHard-10k",
    ("nonthinking", "50k"): "BrickNet-Stage2-NonThinking-50k",
    ("thinking-hard", "50k"): "BrickNet-Stage2-ThinkingHard-50k",
    ("nonthinking", "all"): "BrickNet-Stage2-NonThinking-All",
    ("thinking-hard", "all"): "BrickNet-Stage2-ThinkingHard-All",
}
DATA_FILES = {
    ("nonthinking", "overfit511"): REASONING_ROOT
    / "validation/datasets/BrickNet-Stage2-NonThinking-VAL511-Train.jsonl",
    ("thinking-hard", "overfit511"): REASONING_ROOT
    / "validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL511-Train.jsonl",
    ("nonthinking", "10k"): ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking.jsonl",
    ("thinking-hard", "10k"): ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-ThinkingHard.jsonl",
    ("nonthinking", "50k"): ROOT / "data/bricknet_stage2/50k/BrickNet-Stage2-NonThinking.jsonl",
    ("thinking-hard", "50k"): ROOT / "data/bricknet_stage2/50k/BrickNet-Stage2-ThinkingHard.jsonl",
    ("nonthinking", "all"): ROOT / "data/bricknet_stage2/all/BrickNet-Stage2-NonThinking.jsonl",
    ("thinking-hard", "all"): ROOT / "data/bricknet_stage2/all/BrickNet-Stage2-ThinkingHard.jsonl",
}
PREDICT_DATA_FILES = {
    "nonthinking": REASONING_ROOT / "validation/datasets/BrickNet-Stage2-NonThinking-VAL512-Eval.jsonl",
    "thinking-hard": REASONING_ROOT / "validation/datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl",
}
SAVE_STEPS = {"overfit511": 32, "10k": 250, "50k": 500, "all": 500}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--action", choices=("train", "predict"), required=True)
    parser.add_argument("--variant", choices=("nonthinking", "thinking-hard"), required=True)
    parser.add_argument("--scale", choices=("overfit511", "10k", "50k", "all"), required=True)
    parser.add_argument("--stage0-adapter", type=Path, default=STAGE0_ADAPTER)
    parser.add_argument("--stage0-gate-approved", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke LlamaFactory after all gates pass. Default is dry-run.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _adapter_ready(path: Path) -> bool:
    weights = path / "adapter_model.safetensors"
    legacy_weights = path / "adapter_model.bin"
    return (path / "adapter_config.json").is_file() and (weights.is_file() or legacy_weights.is_file())


def _train_output(variant: str, scale: str) -> Path:
    slug = "thinking_hard" if variant == "thinking-hard" else "nonthinking"
    return ROOT / "saves/Qwen3.5-0.8B-Thinking/lora" / f"train_stage2_{slug}_{scale}_ep3_bs1_ga16_lora64_len16384"


def _predict_output(variant: str, scale: str) -> Path:
    slug = "thinking_hard" if variant == "thinking-hard" else "nonthinking"
    return ROOT / "saves/Qwen3.5-0.8B-Thinking/lora" / f"eval_stage2_{slug}_{scale}_in16384_out16384_p95_t1_k20"


def _selection_manifest(scale: str) -> Path:
    if scale == "overfit511":
        return REASONING_ROOT / "validation/manifests/BrickNet-Stage2-Thinking-Hard-VAL511-Train_manifest.jsonl"
    return REASONING_ROOT / f"stage2/manifests/stage2_train_{scale}_seed42.jsonl"


def _materialization_hint(scale: str) -> str | None:
    if scale not in ("10k", "50k"):
        return None
    return (
        "cd /home/jiahao/task/BrickNet && "
        "PYTHONPATH=src /home/jiahao/miniconda3/envs/bricknet/bin/python "
        f"data_preprocess/prepare_bricknet_stage2_sft.py materialize --scale {scale}"
    )


def _command(args: argparse.Namespace) -> tuple[list[str], Path]:
    stage0 = args.stage0_adapter.expanduser().resolve()
    if args.action == "train":
        config = TRAIN_CONFIGS[args.variant]
        output = _train_output(args.variant, args.scale)
        dataset = DATASETS[(args.variant, args.scale)]
        slug = "ThinkingHard" if args.variant == "thinking-hard" else "NonThinking"
        cache = (
            ROOT
            / ".llamafactory_cache/tokenized_dataset"
            / f"BrickNet-Stage2-{slug}-{args.scale}-qwen35-08b-nothink-len16384"
        )
        command = [
            "conda",
            "run",
            "-n",
            "llamafactory",
            "--no-capture-output",
            "llamafactory-cli",
            "train",
            str(config.relative_to(ROOT)),
            f"dataset={dataset}",
            f"adapter_name_or_path={stage0}",
            f"output_dir={output}",
            f"tokenized_path={cache}",
            f"save_steps={SAVE_STEPS[args.scale]}",
        ]
    else:
        config = PREDICT_CONFIGS[args.variant]
        output = _predict_output(args.variant, args.scale)
        stage2_adapter = _train_output(args.variant, args.scale)
        command = [
            "conda",
            "run",
            "-n",
            "llamafactory",
            "--no-capture-output",
            "llamafactory-cli",
            "train",
            str(config.relative_to(ROOT)),
            f"adapter_name_or_path={stage0},{stage2_adapter}",
            f"output_dir={output}",
        ]
    return command, output


def main() -> None:
    args = parse_args()
    stage0 = args.stage0_adapter.expanduser().resolve()
    command, output = _command(args)
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    checks["stage0_adapter"] = str(stage0)
    checks["stage0_final_adapter_ready"] = _adapter_ready(stage0)
    if not checks["stage0_final_adapter_ready"]:
        blockers.append("WAIT_STAGE0_FINAL: mixed PT-exp1 root lacks final adapter_config/adapter_model")

    token_report = _load_json(TOKEN_REPORT)
    checks["stage1_token_report"] = str(TOKEN_REPORT)
    checks["stage1_training_eligible"] = bool(token_report and token_report.get("training_eligible") is True)
    if not checks["stage1_training_eligible"]:
        blockers.append("Stage-1 full-pool token gate is not training_eligible=true")

    val_report = _load_json(VAL_REPORT)
    val_train_token_report = _load_json(VAL_TRAIN_TOKEN_REPORT)
    val_eval_token_report = _load_json(VAL_EVAL_TOKEN_REPORT)
    checks["val511_train_annotation_gate"] = bool(
        val_report and val_report.get("train_annotation_gate_passed") is True
    )
    checks["val512_eval_preparation_gate"] = bool(val_report and val_report.get("evaluation_gate_passed") is True)
    checks["val_readiness_gate"] = bool(val_report and val_report.get("readiness_gate_passed") is True)
    checks["val511_training_eligible"] = bool(val_report and val_report.get("training_eligible") is True)
    checks["val512_prediction_eligible"] = bool(val_report and val_report.get("prediction_eligible") is True)
    checks["val511_train_token_gate"] = bool(
        val_train_token_report and val_train_token_report.get("training_eligible") is True
    )
    checks["val512_eval_token_gate"] = bool(
        val_eval_token_report and val_eval_token_report.get("training_eligible") is True
    )
    if args.action == "train" and args.scale == "overfit511":
        if not checks["val511_train_annotation_gate"]:
            blockers.append("official VAL511 overfit annotation gate is incomplete")
        if not checks["val511_train_token_gate"]:
            blockers.append("official VAL511 overfit token gate is incomplete")
        if not checks["val_readiness_gate"] or not checks["val511_training_eligible"]:
            blockers.append("official VAL511 consolidated readiness gate is incomplete")
    if args.action == "predict":
        if not checks["val512_eval_preparation_gate"]:
            blockers.append("official VAL512 inference preparation gate is incomplete")
        if not checks["val512_eval_token_gate"]:
            blockers.append("official VAL512 inference token gate is incomplete")
        if not checks["val_readiness_gate"] or not checks["val512_prediction_eligible"]:
            blockers.append("official VAL512 consolidated readiness gate is incomplete")

    data_file = DATA_FILES[(args.variant, args.scale)] if args.action == "train" else PREDICT_DATA_FILES[args.variant]
    checks["dataset"] = str(data_file)
    checks["dataset_ready"] = data_file.is_file()
    if not checks["dataset_ready"]:
        hint = _materialization_hint(args.scale)
        blockers.append("selected Stage-2 dataset is missing" + (f"; prepare with: {hint}" if hint else ""))
    elif args.action == "train" and args.scale == "all" and token_report:
        dataset_key = "Thinking-Hard" if args.variant == "thinking-hard" else "NonThinking-Control"
        expected_hash = token_report.get("datasets", {}).get(dataset_key, {}).get("sha256")
        actual_hash = _sha256(data_file)
        checks["dataset_sha256"] = actual_hash
        checks["dataset_hash_matches_token_report"] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            blockers.append("full Stage-2 dataset hash differs from the token report")

    manifest = _selection_manifest(args.scale)
    checks["selection_manifest"] = str(manifest)
    checks["selection_manifest_ready"] = manifest.is_file()
    if not checks["selection_manifest_ready"]:
        blockers.append("scale selection manifest is missing")

    if args.action == "predict":
        stage2_adapter = _train_output(args.variant, args.scale)
        checks["stage2_adapter"] = str(stage2_adapter)
        checks["stage2_adapter_ready"] = _adapter_ready(stage2_adapter)
        if not checks["stage2_adapter_ready"]:
            blockers.append("WAIT_STAGE2_TRAIN: requested Stage-2 adapter is not complete")

    checks["output_dir"] = str(output)
    checks["output_dir_absent"] = not output.exists()
    if output.exists():
        blockers.append(f"output directory already exists: {output}")

    if args.execute and not args.stage0_gate_approved:
        blockers.append("--execute also requires explicit --stage0-gate-approved")

    result = {
        "stage": 2,
        "action": args.action,
        "variant": args.variant,
        "scale": args.scale,
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
        raise SystemExit("Stage-2 launch blocked; resolve the reported gates first")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
