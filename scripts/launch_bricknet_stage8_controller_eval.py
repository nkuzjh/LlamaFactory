#!/usr/bin/env python3
"""Generate or execute a named Stage-8-policy controller evaluation on frozen VAL512."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from llamafactory.extras.stage8_gate import (
    STAGE8_MAIN_EVAL_MODES,
    resolve_stage5_report_binding,
    resolve_stage8_comparator,
    resolve_stage8_eval_mode,
)


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")
CONTROLLER = BRICKNET_ROOT / "scripts/run_bricknet_agentic_inference.py"
VAL512 = (
    BRICKNET_ROOT
    / "outputs_preprocess/BrickNet-MM-Reasoning/validation/datasets"
    / "BrickNet-Stage2-NonThinking-Control-VAL512-Eval.jsonl"
)
SAVE_ROOT = ROOT / "saves/Qwen3.5-0.8B-Thinking/lora"
PT_EXP1 = SAVE_ROOT / "train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64"
EXP4_2 = SAVE_ROOT / "train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384"
STAGE8_ADAPTERS = {
    "R1-S": SAVE_ROOT / "train_stage8_r1_s_act_success_10k_ep3_bs1_ga16_lora64_len16384",
    "R1-C": SAVE_ROOT / "train_stage8_r1_c_act_correction_10k_token_matched_lora64_len16384",
    "R1-B": SAVE_ROOT / "train_stage8_r1_b_act_rollback_10k_token_matched_lora64_len16384",
}
STAGE8_COMPARATORS = {"S8-ZS-Greedy", "S8-ZS-DFS"}
STAGE8_COMPARATOR_BY_MODE = {
    "a0-act-feedback": "S8-ZS-Greedy",
    "a1-feedback-search": "S8-ZS-DFS",
}
STAGE8_BUILD_REPORT = (
    BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM-Act-SFT/10k/BrickNet-Stage8-R1-report.json"
)


def adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def output_root_for(policy_id: str, mode: str, protocol: str, limit: int | None) -> Path:
    suffix = f"_limit{limit}" if limit is not None else ""
    slug = policy_id.lower().replace("-", "_")
    return SAVE_ROOT / f"eval_stage8_{slug}_val512_stage7_{mode}_{protocol}{suffix}"


def jsonl_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def formal_comparator_ready(policy_id: str) -> tuple[bool, str, Path]:
    mode = STAGE8_MAIN_EVAL_MODES[policy_id]
    root = output_root_for(policy_id, mode, "stage8_act_main_mode", None)
    paths = [
        root / "launch_manifest.json",
        root / "controller_audit.jsonl",
        root / "predictions.jsonl",
        root / "raw_first_choice_predictions.jsonl",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return False, f"missing comparator artifacts: {missing}", root
    counts = {path.name: jsonl_count(path) for path in paths[1:]}
    if set(counts.values()) != {512}:
        return False, f"comparator JSONL counts must all equal 512: {counts}", root
    try:
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid comparator launch manifest: {exc}", root
    if (
        manifest.get("run") != policy_id
        or manifest.get("comparison_role") != "comparator"
        or manifest.get("prompt_protocol") != "stage8-act"
        or manifest.get("mode") != mode
        or manifest.get("evaluation_protocol") != "stage8_act_main_mode"
        or manifest.get("main_mode") is not True
        or manifest.get("ready") is not True
        or manifest.get("blockers") != []
    ):
        return False, "comparator launch manifest identity/protocol mismatch", root
    expected_adapters = [str(PT_EXP1), str(EXP4_2)]
    if manifest.get("adapter_chain") != expected_adapters:
        return False, "comparator launch manifest adapter chain mismatch", root
    return True, "", root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=STAGE8_MAIN_EVAL_MODES,
        required=True,
        help=(
            "Named policy identity. S8-ZS-Greedy/S8-ZS-DFS are exp4_2 zero-shot comparators "
            "under stage8-act; R1-S/C/B are interaction-trained treatments."
        ),
    )
    parser.add_argument("--mode", choices=("a0-act-feedback", "a1-feedback-search"))
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Required when --mode differs from the frozen main mode for the selected run.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    try:
        mode, mode_mismatch = resolve_stage8_eval_mode(args.run, args.mode, ablation=args.ablation)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    comparison_role = "comparator" if args.run in STAGE8_COMPARATORS else "treatment"
    if comparison_role == "comparator" and mode_mismatch:
        raise SystemExit("named Stage-8 comparators have frozen controller modes and cannot be mode ablations")
    protocol = "ablation_mode_mismatch" if mode_mismatch else "stage8_act_main_mode"
    output_root = output_root_for(args.run, mode, protocol, args.limit)
    audit_output = output_root / "controller_audit.jsonl"
    predictions_output = output_root / "predictions.jsonl"
    raw_predictions_output = output_root / "raw_first_choice_predictions.jsonl"
    comparator_id = args.run if comparison_role == "comparator" else STAGE8_COMPARATOR_BY_MODE[mode]
    if comparison_role == "treatment" and not mode_mismatch:
        assert comparator_id == resolve_stage8_comparator(args.run)
    adapters = [PT_EXP1, EXP4_2]
    if comparison_role == "treatment":
        adapters.append(STAGE8_ADAPTERS[args.run])
    command = [
        "env",
        f"PYTHONPATH={BRICKNET_ROOT / 'src'}",
        str(BRICKNET_PYTHON),
        str(CONTROLLER),
        "--input",
        str(VAL512),
        "--output",
        str(audit_output),
        "--predictions-output",
        str(predictions_output),
        "--raw-predictions-output",
        str(raw_predictions_output),
        "--media-dir",
        str(BRICKNET_ROOT),
        "--mode",
        mode,
        "--backend",
        "hf",
        "--prompt-protocol",
        "stage8-act",
        "--experiment-id",
        args.run,
        "--seed",
        "42",
    ]
    build_report = None
    stage5_binding = None
    if STAGE8_BUILD_REPORT.is_file():
        try:
            build_report = json.loads(STAGE8_BUILD_REPORT.read_text(encoding="utf-8"))
            stage5_binding = resolve_stage5_report_binding(build_report)
        except (json.JSONDecodeError, ValueError) as exc:
            stage5_error = str(exc)
        else:
            stage5_error = None
    else:
        stage5_error = f"Stage-8 build report is missing: {STAGE8_BUILD_REPORT}"
    if stage5_binding:
        command.extend(("--stage5-report", stage5_binding["path"]))
    for adapter in adapters:
        command.extend(("--adapter", str(adapter)))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))

    blockers = []
    comparator_ready = None
    comparator_error = None
    comparator_output_root = None
    if comparison_role == "treatment" and args.limit is None:
        comparator_ready, comparator_error, comparator_output_root = formal_comparator_ready(comparator_id)
        if not comparator_ready:
            blockers.append("WAIT_NAMED_STAGE8_ACT_COMPARATOR_ARTIFACT")
    if not BRICKNET_PYTHON.is_file() or not CONTROLLER.is_file() or not VAL512.is_file():
        blockers.append("WAIT_STAGE7_CONTROLLER_RUNTIME_OR_VAL512")
    if stage5_binding is None:
        blockers.append("WAIT_FIXED_STAGE5_REPORT_PATH_AND_HASH_BINDING")
    missing_adapters = [str(adapter) for adapter in adapters if not adapter_ready(adapter)]
    if missing_adapters:
        blockers.append(
            "WAIT_PT_EXP1_EXP4_2_COMPARATOR_CHAIN"
            if comparison_role == "comparator"
            else "WAIT_PT_EXP1_EXP4_2_AND_STAGE8_ADAPTER_CHAIN"
        )
    if audit_output.exists() or predictions_output.exists() or raw_predictions_output.exists():
        blockers.append("WAIT_FRESH_STAGE8_VAL512_OUTPUT_PATH")
    result = {
        "evaluation": "BrickNet VAL512 Stage8-act controller",
        "run": args.run,
        "comparison_role": comparison_role,
        "comparator_id": comparator_id,
        "comparator_ready": comparator_ready,
        "comparator_error": comparator_error,
        "comparator_output_root": str(comparator_output_root) if comparator_output_root else None,
        "mode": mode,
        "main_mode": not mode_mismatch,
        "ablation_requested": args.ablation,
        "evaluation_protocol": protocol,
        "prompt_protocol": "stage8-act",
        "stage8_build_report": str(STAGE8_BUILD_REPORT),
        "stage5_report_binding": stage5_binding,
        "stage5_report_error": stage5_error,
        "adapter_chain": [str(adapter) for adapter in adapters],
        "command": shlex.join(command),
        "blockers": blockers,
        "ready": not blockers,
        "executed": False,
    }
    print(json.dumps(result, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit(2)
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "launch_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    environment = dict(os.environ)
    subprocess.run(command, cwd=BRICKNET_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
