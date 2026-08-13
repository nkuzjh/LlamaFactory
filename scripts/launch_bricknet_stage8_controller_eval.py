#!/usr/bin/env python3
"""Generate or execute a Stage-7 controller evaluation on the frozen VAL512 split."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


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


def adapter_ready(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=STAGE8_ADAPTERS, required=True)
    parser.add_argument("--mode", choices=("a0-act-feedback", "a1-feedback-search"), default="a1-feedback-search")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    output_root = SAVE_ROOT / f"eval_stage8_{args.run.lower().replace('-', '_')}_val512_stage7_{args.mode}"
    audit_output = output_root / "controller_audit.jsonl"
    predictions_output = output_root / "predictions.jsonl"
    adapters = [PT_EXP1, EXP4_2, STAGE8_ADAPTERS[args.run]]
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
        "--media-dir",
        str(BRICKNET_ROOT),
        "--mode",
        args.mode,
        "--backend",
        "hf",
        "--prompt-protocol",
        "stage8-act",
        "--seed",
        "42",
    ]
    for adapter in adapters:
        command.extend(("--adapter", str(adapter)))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))

    blockers = []
    if not BRICKNET_PYTHON.is_file() or not CONTROLLER.is_file() or not VAL512.is_file():
        blockers.append("WAIT_STAGE7_CONTROLLER_RUNTIME_OR_VAL512")
    missing_adapters = [str(adapter) for adapter in adapters if not adapter_ready(adapter)]
    if missing_adapters:
        blockers.append("WAIT_PT_EXP1_EXP4_2_AND_STAGE8_ADAPTER_CHAIN")
    if audit_output.exists() or predictions_output.exists():
        blockers.append("WAIT_FRESH_STAGE8_VAL512_OUTPUT_PATH")
    result = {
        "evaluation": "BrickNet VAL512 Stage7 controller",
        "run": args.run,
        "mode": args.mode,
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
    environment = dict(os.environ)
    subprocess.run(command, cwd=BRICKNET_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
