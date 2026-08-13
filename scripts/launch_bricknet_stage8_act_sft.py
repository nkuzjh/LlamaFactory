#!/usr/bin/env python3
"""Fail-closed Stage-8 Act-only SFT launcher (dry-run by default)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT")
RUNS = {
    "R1-S": "qwen35_08b_bricknet_stage8_r1_s_act_success_10k.yaml",
    "R1-C": "qwen35_08b_bricknet_stage8_r1_c_act_correction_10k.yaml",
    "R1-B": "qwen35_08b_bricknet_stage8_r1_b_act_rollback_10k.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--scale", choices=("64", "10k"), default="64")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scale_dir = "smoke64" if args.scale == "64" else "10k"
    bricknet_out = BRICKNET_ROOT / scale_dir
    dataset = bricknet_out / f"BrickNet-Stage8-{args.run}.jsonl"
    report = bricknet_out / "token_audit" / f"BrickNet-Stage8-{args.run}.json"
    blockers = []
    if not dataset.is_file():
        blockers.append("WAIT_STAGE8_DATASET")
    audit = json.loads(report.read_text()) if report.is_file() else None
    if audit is None or not audit.get("training_eligible"):
        blockers.append("WAIT_STAGE8_ZERO_TRUNCATION_TOKEN_AUDIT")
    elif audit.get("dataset") != str(dataset) or audit.get("dataset_sha256") != sha256_file(dataset):
        blockers.append("WAIT_STAGE8_AUDIT_DATASET_HASH_MATCH")
    if args.run in {"R1-C", "R1-B"}:
        baseline_path = bricknet_out / "token_audit/BrickNet-Stage8-R1-S.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
        if baseline is None or audit is None:
            blockers.append("WAIT_R1_S_SUPERVISED_TOKEN_BASELINE")
        elif audit.get("matched_max_steps") is None or audit.get("matched_supervised_token_budget") != 3 * baseline.get(
            "supervised_assistant_tokens", -1
        ):
            blockers.append("WAIT_TOKEN_MATCHED_MAX_STEPS")
    config = ROOT / "examples/train_lora" / RUNS[args.run]
    if not config.is_file() or not (ROOT / "data/dataset_info.json").is_file():
        blockers.append("WAIT_STAGE8_CONFIG_OR_DATASET_REGISTRY")
    command = ["llamafactory-cli", "train", str(config)]
    command += [f"dataset=BrickNet-Stage8-{args.run}-{args.scale}"]
    if args.scale == "64":
        slug = args.run.lower().replace("-", "_")
        command += [
            f"tokenized_path=.llamafactory_cache/tokenized_dataset/stage8-{slug}-act-smoke64-len16384",
            f"output_dir=saves/Qwen3.5-0.8B-Thinking/lora/train_stage8_{slug}_act_smoke64",
        ]
    if args.run in {"R1-C", "R1-B"} and audit and audit.get("matched_max_steps"):
        command += [f"max_steps={audit['matched_max_steps']}"]
    print(json.dumps({"run": args.run, "command": command, "audit": audit, "blockers": blockers}, indent=2))
    if blockers:
        raise SystemExit(2)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
