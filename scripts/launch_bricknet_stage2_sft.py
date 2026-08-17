#!/usr/bin/env python3
"""Gate and print/launch one approved BrickNet Stage-2 experiment.

The default mode is a dry-run. Training or prediction is possible only with
both ``--execute`` and ``--stage0-gate-approved`` after all artifact checks pass;
10k training additionally requires ``--overfit-gate-approved``. Stage-2 50k/all
experiments are deliberately not exposed while that route is paused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
TRACE_EXTRACTOR = BRICKNET_ROOT / "scripts/extract_reasoning_predictions.py"
REASONING_ROOT = BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-Reasoning"
CONFIG_ROOT = ROOT / "examples/train_lora"
STAGE0_ADAPTER = (
    ROOT
    / "saves/Qwen3.5-0.8B-Thinking/lora"
    / "train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64"
)
TOKEN_REPORT = REASONING_ROOT / "reports/token_audit/BrickNet-MM-Reasoning_token_audit_report.json"
STAGE2_V2_ROOT = REASONING_ROOT / "stage2_v2"
STAGE2_V2_REPORT = STAGE2_V2_ROOT / "reports/Stage2-V2-Lean-State_report.json"
STAGE2_V2_TRAIN_TOKEN_REPORT = (
    STAGE2_V2_ROOT
    / "reports/token_audit/train10k/BrickNet-MM-Reasoning_token_audit_report.json"
)
STAGE2_V2_EVAL_TOKEN_REPORT = (
    STAGE2_V2_ROOT
    / "reports/token_audit/eval_val512/BrickNet-MM-Reasoning_token_audit_report.json"
)
VAL_REPORT = REASONING_ROOT / "validation/reports/BrickNet-Stage2-VAL511-Train-VAL512-Eval_report.json"
VAL_TRAIN_TOKEN_REPORT = (
    REASONING_ROOT / "validation/reports/token_audit/train_val511/BrickNet-MM-Reasoning_token_audit_report.json"
)
VAL_EVAL_TOKEN_REPORT = (
    REASONING_ROOT / "validation/reports/token_audit/eval_val512/BrickNet-MM-Reasoning_token_audit_report.json"
)
DATASET_REGISTRY = ROOT / "data/dataset_info.json"


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    variant: str
    scale: str
    dataset: str
    train_file: Path
    eval_file: Path
    train_config: Path
    predict_config: Path
    train_output: Path
    predict_output: Path
    selection_manifest: Path
    expected_train_count: int


VAL_ROOT = REASONING_ROOT / "validation"
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
EXPERIMENTS = {
    ("nonthinking-control", "overfit511"): Experiment(
        experiment_id="exp4",
        variant="nonthinking-control",
        scale="overfit511",
        dataset="BrickNet-Stage2-NonThinking-Control-VAL511-Train",
        train_file=VAL_ROOT / "datasets/BrickNet-Stage2-NonThinking-Control-VAL511-Train.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_nonthinking_control_val511.yaml",
        predict_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_nonthinking_control_predict.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_qwen35_08b_mixedpt_stage2_nonthinking_control_val511_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT / "eval_exp4_stage2_nonthinking_control_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=VAL_ROOT / "manifests/BrickNet-Stage2-Thinking-Hard-VAL511-Train_manifest.jsonl",
        expected_train_count=511,
    ),
    ("thinking-hard", "overfit511"): Experiment(
        experiment_id="exp4_1",
        variant="thinking-hard",
        scale="overfit511",
        dataset="BrickNet-Stage2-ThinkingHard-VAL511-Train",
        train_file=VAL_ROOT / "datasets/BrickNet-Stage2-Thinking-Hard-VAL511-Train.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_val511.yaml",
        predict_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_1_thinking_hard_predict.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_1_qwen35_08b_mixedpt_stage2_thinking_hard_val511_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT / "eval_exp4_1_stage2_thinking_hard_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=VAL_ROOT / "manifests/BrickNet-Stage2-Thinking-Hard-VAL511-Train_manifest.jsonl",
        expected_train_count=511,
    ),
    ("nonthinking-control", "10k"): Experiment(
        experiment_id="exp4_2",
        variant="nonthinking-control",
        scale="10k",
        dataset="BrickNet-Stage2-NonThinking-Control-10k",
        train_file=ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_10k.yaml",
        predict_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_2_nonthinking_control_predict.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT / "eval_exp4_2_stage2_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=REASONING_ROOT / "stage2/manifests/stage2_train_10k_seed42.jsonl",
        expected_train_count=10_000,
    ),
    ("thinking-hard", "10k"): Experiment(
        experiment_id="exp4_3",
        variant="thinking-hard",
        scale="10k",
        dataset="BrickNet-Stage2-ThinkingHard-10k",
        train_file=ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-ThinkingHard.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_10k.yaml",
        predict_config=CONFIG_ROOT / "qwen35_08b_bricknet_stage2_exp4_3_thinking_hard_predict.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_3_qwen35_08b_mixedpt_stage2_thinking_hard_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT / "eval_exp4_3_stage2_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=REASONING_ROOT / "stage2/manifests/stage2_train_10k_seed42.jsonl",
        expected_train_count=10_000,
    ),
    ("thinking-hard-v2-lean-state", "10k"): Experiment(
        experiment_id="exp4_3_1",
        variant="thinking-hard-v2-lean-state",
        scale="10k",
        dataset="BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
        train_file=ROOT
        / "data/bricknet_stage2_v2/10k/BrickNet-Stage2-ThinkingHard-V2-LeanState.jsonl",
        eval_file=STAGE2_V2_ROOT
        / "validation/datasets/BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_3_1_thinking_hard_v2_lean_state_10k.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_3_1_thinking_hard_v2_lean_state_predict.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_3_1_qwen35_08b_mixedpt_stage2_thinking_hard_v2_lean_state_10k_ep3_bs1_ga16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_3_1_stage2_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=REASONING_ROOT / "stage2/manifests/stage2_train_10k_seed42.jsonl",
        expected_train_count=10_000,
    ),
    # PT-exp2-100k downstream variants.  These registry keys are intentionally
    # not reachable from this launcher's (variant, scale) CLI: they exist only
    # so evaluate_bricknet_stage2.py can run the complete VAL512 evaluation
    # pipeline for the externally initialized experiments.  Training and
    # prediction for these variants use direct llamafactory-cli commands.
    ("nonthinking-control-pt-exp2-100k", "10k"): Experiment(
        experiment_id="exp4_4_1",
        variant="nonthinking-control",
        scale="10k",
        dataset="BrickNet-Stage2-NonThinking-Control-10k",
        train_file=ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-NonThinking-Control.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_10k_pt_exp2_100k.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_4_1_nonthinking_control_predict_pt_exp2_100k.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_4_1_qwen35_08b_PT_exp2_100k_stage2_nonthinking_control_10k_ep3_bs1_gbs16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_4_1_PT_exp2_100k_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=REASONING_ROOT / "stage2/manifests/stage2_train_10k_seed42.jsonl",
        expected_train_count=10_000,
    ),
    ("thinking-hard-pt-exp2-100k", "10k"): Experiment(
        experiment_id="exp4_7_1",
        variant="thinking-hard",
        scale="10k",
        dataset="BrickNet-Stage2-ThinkingHard-10k",
        train_file=ROOT / "data/bricknet_stage2/10k/BrickNet-Stage2-ThinkingHard.jsonl",
        eval_file=VAL_ROOT / "datasets/BrickNet-Stage2-Thinking-Hard-VAL512-Eval.jsonl",
        train_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_10k_pt_exp2_100k.yaml",
        predict_config=CONFIG_ROOT
        / "qwen35_08b_bricknet_stage2_exp4_7_1_thinking_hard_predict_pt_exp2_100k.yaml",
        train_output=SAVE_ROOT
        / "train_exp4_7_1_qwen35_08b_PT_exp2_100k_stage2_thinking_hard_10k_ep3_bs1_gbs16_lora64_len16384",
        predict_output=SAVE_ROOT
        / "eval_exp4_7_1_PT_exp2_100k_thinking_hard_10k_val512_in16384_out16384_p95_t1_k20",
        selection_manifest=REASONING_ROOT / "stage2/manifests/stage2_train_10k_seed42.jsonl",
        expected_train_count=10_000,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--action", choices=("train", "predict"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "nonthinking-control",
            "thinking-hard",
            "thinking-hard-v2-lean-state",
        ),
        required=True,
    )
    parser.add_argument("--scale", choices=("overfit511", "10k"), required=True)
    parser.add_argument("--stage0-adapter", type=Path, default=STAGE0_ADAPTER)
    parser.add_argument("--stage0-gate-approved", action="store_true")
    parser.add_argument(
        "--overfit-gate-approved",
        action="store_true",
        help="Required to execute a 10k train after the paired VAL511 gate is approved.",
    )
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


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _adapter_ready(path: Path) -> bool:
    weights = path / "adapter_model.safetensors"
    legacy_weights = path / "adapter_model.bin"
    return (path / "adapter_config.json").is_file() and (weights.is_file() or legacy_weights.is_file())


def _command(args: argparse.Namespace, experiment: Experiment, stage0: Path) -> tuple[list[str], Path]:
    if args.action == "train":
        command = [
            "conda",
            "run",
            "-n",
            "llamafactory",
            "--no-capture-output",
            "llamafactory-cli",
            "train",
            str(experiment.train_config.relative_to(ROOT)),
            f"adapter_name_or_path={stage0}",
        ]
        output = experiment.train_output
    else:
        command = [
            "conda",
            "run",
            "-n",
            "llamafactory",
            "--no-capture-output",
            "llamafactory-cli",
            "train",
            str(experiment.predict_config.relative_to(ROOT)),
            f"adapter_name_or_path={stage0},{experiment.train_output}",
        ]
        output = experiment.predict_output
    return command, output


def _postprocess_command(experiment: Experiment) -> list[str]:
    return [
        "env",
        f"PYTHONPATH={BRICKNET_ROOT / 'src'}",
        str(BRICKNET_PYTHON),
        str(TRACE_EXTRACTOR),
        "--variant",
        _trace_variant(experiment),
        "--label-format",
        "path",
        "--input",
        str(experiment.predict_output / "generated_predictions.jsonl"),
        "--output",
        str(experiment.predict_output / "path_predictions.jsonl"),
        "--report",
        str(experiment.predict_output / "trace_extraction_report.json"),
    ]


def _trace_variant(experiment: Experiment) -> str:
    """Return the strict extractor contract for this experiment."""
    return experiment.variant


def _check_val_gates(args: argparse.Namespace, checks: dict[str, Any], blockers: list[str]) -> None:
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


def _check_10k_materialization(experiment: Experiment, checks: dict[str, Any], blockers: list[str]) -> None:
    report_path = ROOT / "data/bricknet_stage2/10k/materialization_report.json"
    report = _load_json(report_path)
    reported_manifest = report.get("selection_manifest") if report else None
    report_key = "nonthinking_control" if experiment.variant == "nonthinking-control" else "thinking_hard"
    checks["materialization_report"] = str(report_path)
    checks["materialization_training_eligible"] = bool(
        report
        and report.get("scale") == "10k"
        and report.get("paired_count") == 10_000
        and report.get("paired_order_and_ids") is True
        and report.get("training_eligible") is True
    )
    checks["materialization_manifest_matches"] = bool(
        report
        and isinstance(reported_manifest, str)
        and experiment.selection_manifest.is_file()
        and Path(reported_manifest).resolve() == experiment.selection_manifest.resolve()
        and report.get("selection_manifest_sha256") == _sha256(experiment.selection_manifest)
    )
    expected = report.get(report_key, {}) if report else {}
    expected_hash = expected.get("sha256") if isinstance(expected, dict) else None
    checks["dataset_hash_matches_materialization_report"] = bool(
        experiment.train_file.is_file() and expected_hash and _sha256(experiment.train_file) == expected_hash
    )
    if not checks["materialization_training_eligible"]:
        blockers.append("10k paired materialization report is incomplete")
    if not checks["materialization_manifest_matches"]:
        blockers.append("10k materialization used another selection manifest")
    if not checks["dataset_hash_matches_materialization_report"]:
        blockers.append("10k dataset hash differs from its materialization report")


def _check_stage2_v2_artifacts(
    args: argparse.Namespace,
    experiment: Experiment,
    checks: dict[str, Any],
    blockers: list[str],
) -> None:
    """Gate V2 data against its independent construction and processor reports."""
    report = _load_json(STAGE2_V2_REPORT)
    checks["stage2_v2_report"] = str(STAGE2_V2_REPORT)
    checks["stage2_v2_training_eligible"] = bool(
        report
        and report.get("training_eligible") is True
        and report.get("count") == experiment.expected_train_count
    )
    reported_train_file = report.get("train_file") if report else None
    checks["stage2_v2_train_path_matches"] = bool(
        isinstance(reported_train_file, str)
        and Path(reported_train_file).resolve() == experiment.train_file.resolve()
    )
    checks["stage2_v2_train_hash_matches"] = bool(
        report
        and experiment.train_file.is_file()
        and report.get("train_sha256") == _sha256(experiment.train_file)
    )
    if not checks["stage2_v2_training_eligible"]:
        blockers.append("Stage-2 V2 annotation report is not training_eligible for 10k")
    if not checks["stage2_v2_train_path_matches"]:
        blockers.append("Stage-2 V2 annotation report points to another training dataset")
    if not checks["stage2_v2_train_hash_matches"]:
        blockers.append("Stage-2 V2 training dataset hash differs from its annotation report")

    if args.action == "predict":
        val512 = report.get("val512", {}) if report else {}
        reported_eval_file = val512.get("file") if isinstance(val512, dict) else None
        checks["stage2_v2_val512_prediction_eligible"] = bool(
            isinstance(val512, dict)
            and val512.get("prediction_eligible") is True
            and val512.get("count") == 512
        )
        checks["stage2_v2_val512_path_matches"] = bool(
            isinstance(reported_eval_file, str)
            and Path(reported_eval_file).resolve() == experiment.eval_file.resolve()
        )
        checks["stage2_v2_val512_hash_matches"] = bool(
            isinstance(val512, dict)
            and experiment.eval_file.is_file()
            and val512.get("sha256") == _sha256(experiment.eval_file)
        )
        if not checks["stage2_v2_val512_prediction_eligible"]:
            blockers.append("Stage-2 V2 VAL512 report is not prediction_eligible")
        if not checks["stage2_v2_val512_path_matches"]:
            blockers.append("Stage-2 V2 report points to another VAL512 dataset")
        if not checks["stage2_v2_val512_hash_matches"]:
            blockers.append("Stage-2 V2 VAL512 dataset hash differs from its annotation report")

    token_report_path = (
        STAGE2_V2_TRAIN_TOKEN_REPORT
        if args.action == "train"
        else STAGE2_V2_EVAL_TOKEN_REPORT
    )
    token_report = _load_json(token_report_path)
    token_dataset_keys = (
        ("Thinking-Hard-V2-Lean-State",)
        if args.action == "train"
        else (
            "Thinking-Hard-V2-Lean-State-VAL512",
            "Thinking-Hard-V2-Lean-State",
        )
    )
    token_dataset: dict[str, Any] = {}
    for key in token_dataset_keys:
        candidate = token_report.get("datasets", {}).get(key, {}) if token_report else {}
        if isinstance(candidate, dict) and candidate:
            token_dataset = candidate
            break
    audited_file = experiment.train_file if args.action == "train" else experiment.eval_file
    audited_count = experiment.expected_train_count if args.action == "train" else 512
    token_source = token_dataset.get("path")
    checks["stage2_v2_token_report"] = str(token_report_path)
    checks["stage2_v2_token_gate"] = bool(
        token_report
        and token_report.get("training_eligible") is True
        and token_report.get("zero_errors") is True
        and token_report.get("zero_truncation") is True
    )
    checks["stage2_v2_token_dataset_matches"] = bool(
        isinstance(token_source, str)
        and Path(token_source).resolve() == audited_file.resolve()
        and token_dataset.get("count") == audited_count
        and audited_file.is_file()
        and token_dataset.get("sha256") == _sha256(audited_file)
    )
    if not checks["stage2_v2_token_gate"]:
        blockers.append("Stage-2 V2 processor token gate is incomplete")
    if not checks["stage2_v2_token_dataset_matches"]:
        blockers.append("Stage-2 V2 processor audit does not match the selected dataset")


def main() -> None:
    args = parse_args()
    experiment = EXPERIMENTS[(args.variant, args.scale)]
    stage0 = args.stage0_adapter.expanduser().resolve()
    command, output = _command(args, experiment, stage0)
    postprocess_command = _postprocess_command(experiment) if args.action == "predict" else None
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    checks["stage0_adapter"] = str(stage0)
    checks["stage0_final_adapter_ready"] = _adapter_ready(stage0)
    if not checks["stage0_final_adapter_ready"]:
        blockers.append("WAIT_STAGE0_FINAL: mixed PT-exp1 root lacks final adapter_config/adapter_model")

    if experiment.variant == "thinking-hard-v2-lean-state":
        _check_stage2_v2_artifacts(args, experiment, checks, blockers)
    else:
        token_report = _load_json(TOKEN_REPORT)
        checks["stage1_token_report"] = str(TOKEN_REPORT)
        checks["stage1_training_eligible"] = bool(
            token_report and token_report.get("training_eligible") is True
        )
        if not checks["stage1_training_eligible"]:
            blockers.append("Stage-1 full-pool token gate is not training_eligible=true")
        token_dataset_key = (
            "NonThinking-Control"
            if experiment.variant == "nonthinking-control"
            else "Thinking-Hard"
        )
        token_dataset = token_report.get("datasets", {}).get(token_dataset_key, {}) if token_report else {}
        token_source = Path(token_dataset.get("path", "")) if isinstance(token_dataset, dict) else None
        checks["stage1_source_dataset"] = str(token_source) if token_source else None
        checks["stage1_source_hash_matches_token_report"] = bool(
            token_source
            and token_source.is_file()
            and token_dataset.get("count") == 66_456
            and token_dataset.get("sha256") == _sha256(token_source)
        )
        if not checks["stage1_source_hash_matches_token_report"]:
            blockers.append("Stage-1 source dataset differs from the full-pool token report")

    _check_val_gates(args, checks, blockers)

    config = experiment.train_config if args.action == "train" else experiment.predict_config
    checks["config"] = str(config)
    checks["config_ready"] = config.is_file()
    if not checks["config_ready"]:
        blockers.append("selected Stage-2 configuration is missing")

    data_file = experiment.train_file if args.action == "train" else experiment.eval_file
    expected_count = experiment.expected_train_count if args.action == "train" else 512
    checks["dataset"] = str(data_file)
    checks["dataset_ready"] = data_file.is_file()
    checks["expected_row_count"] = expected_count
    checks["actual_row_count"] = _line_count(data_file) if data_file.is_file() else None
    checks["row_count_matches"] = checks["actual_row_count"] == expected_count
    if not checks["dataset_ready"]:
        blockers.append("selected Stage-2 dataset is missing")
    elif not checks["row_count_matches"]:
        blockers.append("selected Stage-2 dataset row count differs from the contract")

    registry = _load_json(DATASET_REGISTRY)
    registry_key = (
        experiment.dataset
        if args.action == "train"
        else (
            "BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval"
            if args.variant == "thinking-hard-v2-lean-state"
            else (
                "BrickNet-Stage2-NonThinking-Control-VAL512-Eval"
                if args.variant == "nonthinking-control"
                else "BrickNet-Stage2-ThinkingHard-VAL512-Eval"
            )
        )
    )
    registry_entry = registry.get(registry_key) if registry else None
    registered_file = Path(registry_entry.get("file_name", "")) if isinstance(registry_entry, dict) else None
    checks["dataset_registry_key"] = registry_key
    checks["dataset_registry_matches"] = bool(registered_file and registered_file.resolve() == data_file.resolve())
    if not checks["dataset_registry_matches"]:
        blockers.append("dataset registry entry is missing or points to another file")

    manifest = experiment.selection_manifest
    checks["selection_manifest"] = str(manifest)
    checks["selection_manifest_ready"] = manifest.is_file()
    checks["selection_manifest_row_count"] = _line_count(manifest) if manifest.is_file() else None
    checks["selection_manifest_count_matches"] = (
        checks["selection_manifest_row_count"] == experiment.expected_train_count
    )
    if not checks["selection_manifest_ready"]:
        blockers.append("scale selection manifest is missing")
    elif not checks["selection_manifest_count_matches"]:
        blockers.append("selection manifest row count differs from the contract")

    if args.scale == "10k" and experiment.variant != "thinking-hard-v2-lean-state":
        _check_10k_materialization(experiment, checks, blockers)

    if args.action == "predict":
        checks["stage2_adapter"] = str(experiment.train_output)
        checks["stage2_adapter_ready"] = _adapter_ready(experiment.train_output)
        if not checks["stage2_adapter_ready"]:
            blockers.append("WAIT_STAGE2_TRAIN: requested Stage-2 adapter is not complete")
        checks["trace_extractor"] = str(TRACE_EXTRACTOR)
        checks["trace_extractor_ready"] = TRACE_EXTRACTOR.is_file()
        checks["bricknet_python"] = str(BRICKNET_PYTHON)
        checks["bricknet_python_ready"] = BRICKNET_PYTHON.is_file()
        if not checks["trace_extractor_ready"]:
            blockers.append("BrickNet strict prediction extractor is missing")
        if not checks["bricknet_python_ready"]:
            blockers.append("BrickNet Python environment is missing")

    checks["output_dir"] = str(output)
    checks["output_dir_absent"] = not output.exists()
    if output.exists():
        blockers.append(f"output directory already exists: {output}")

    if args.execute and not args.stage0_gate_approved:
        blockers.append("--execute also requires explicit --stage0-gate-approved")
    if args.action == "train" and args.scale == "10k" and not args.overfit_gate_approved:
        blockers.append("WAIT_STAGE2_OVERFIT_GATE: 10k training requires explicit --overfit-gate-approved")

    result = {
        "stage": 2,
        "experiment": experiment.experiment_id,
        "action": args.action,
        "variant": args.variant,
        "scale": args.scale,
        "mode": "execute" if args.execute else "dry-run",
        "checks": checks,
        "blockers": blockers,
        "ready": not blockers,
        "command": shlex.join(command),
        "postprocess_command": shlex.join(postprocess_command) if postprocess_command else None,
        "training_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("Stage-2 launch blocked; resolve the reported gates first")
    subprocess.run(command, cwd=ROOT, check=True)
    if postprocess_command:
        subprocess.run(postprocess_command, cwd=BRICKNET_ROOT, check=True)


if __name__ == "__main__":
    main()
