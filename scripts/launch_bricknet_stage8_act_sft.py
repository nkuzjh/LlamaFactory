#!/usr/bin/env python3
"""Fail-closed Stage-8 Act-only SFT launcher (dry-run by default)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from llamafactory.extras.stage8_gate import (
    adapter_artifact_sha256,
    evaluate_stage8_build_report,
    validate_adapter_chain,
)


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM-Act-SFT")
INITIALIZATION_AUDIT = ROOT / "scripts/audit_bricknet_stage8_initialization.py"
TOKEN_AUDIT_SCHEMA = "bricknet-stage8-act-token-audit-v2"
INITIALIZATION_AUDIT_SCHEMA = "bricknet-stage8-initialization-audit-v1"
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


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--scale", choices=("64", "10k"), default="64")
    parser.add_argument(
        "--refresh-initialization-audit",
        action="store_true",
        help="Run the no-optimizer initialization/logit audit before evaluating gates.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scale_dir = "smoke64" if args.scale == "64" else "10k"
    bricknet_out = BRICKNET_ROOT / scale_dir
    dataset = bricknet_out / f"BrickNet-Stage8-{args.run}.jsonl"
    build_report_path = bricknet_out / "BrickNet-Stage8-R1-report.json"
    report = bricknet_out / "token_audit" / f"BrickNet-Stage8-{args.run}.json"
    initialization_report = bricknet_out / "initialization_audit" / f"BrickNet-Stage8-{args.run}.json"
    config = ROOT / "examples/train_lora" / RUNS[args.run]
    if args.refresh_initialization_audit:
        subprocess.run(
            [sys.executable, str(INITIALIZATION_AUDIT), str(config), str(initialization_report)],
            cwd=ROOT,
            check=True,
        )
    blockers = []
    if not dataset.is_file():
        blockers.append("WAIT_STAGE8_DATASET")
    audit = json.loads(report.read_text()) if report.is_file() else None
    if audit is None or audit.get("schema_version") != TOKEN_AUDIT_SCHEMA or not audit.get("training_eligible"):
        blockers.append("WAIT_STAGE8_ZERO_TRUNCATION_TOKEN_AUDIT")
    elif not dataset.is_file() or audit.get("dataset") != str(dataset.resolve()) or audit.get(
        "dataset_sha256"
    ) != sha256_file(dataset):
        blockers.append("WAIT_STAGE8_AUDIT_DATASET_HASH_MATCH")
    elif audit.get("config") != str(config.resolve()) or audit.get("config_sha256") != sha256_file(config):
        blockers.append("WAIT_STAGE8_AUDIT_CONFIG_HASH_MATCH")
    else:
        boundary_plan = Path(audit.get("boundary_plan", ""))
        if (
            not boundary_plan.is_file()
            or audit.get("boundary_plan_sha256") != sha256_file(boundary_plan)
            or audit.get("cutoff_hits") != 0
            or not audit.get("token_mix_eligible")
        ):
            blockers.append("WAIT_STAGE8_BOUNDARY_PLAN_AND_TOKEN_MIX_GATE")

    build_report = json.loads(build_report_path.read_text()) if build_report_path.is_file() else None
    build_gate = None
    if build_report is None:
        blockers.append("WAIT_STAGE8_BRICKNET_BUILD_REPORT")
    elif not dataset.is_file() or audit is None:
        blockers.append("WAIT_STAGE8_BUILD_REPORT_DATASET_AND_TOKEN_AUDIT_BINDING")
    else:
        build_gate = evaluate_stage8_build_report(
            build_report,
            run=args.run,
            expected_size=64 if args.scale == "64" else 10_000,
            dataset_path=dataset,
            dataset_sha256=sha256_file(dataset),
            dataset_count=line_count(dataset),
            dataset_window_materialized=audit.get("dataset_window_materialized"),
        )
        if not build_gate["build_report_gate_passed"]:
            blockers.extend(f"WAIT_STAGE8_{reason}" for reason in build_gate["build_report_blockers"])
        if (
            audit.get("build_report") != str(build_report_path.resolve())
            or audit.get("build_report_sha256") != sha256_file(build_report_path)
            or audit.get("input_count") != line_count(dataset)
        ):
            blockers.append("WAIT_STAGE8_BUILD_REPORT_TOKEN_AUDIT_HASH_BINDING")
    if args.run in {"R1-C", "R1-B"}:
        baseline_path = bricknet_out / "token_audit/BrickNet-Stage8-R1-S.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None
        if baseline is None or baseline.get("schema_version") != TOKEN_AUDIT_SCHEMA or audit is None:
            blockers.append("WAIT_R1_S_SUPERVISED_TOKEN_BASELINE")
        elif audit.get("matched_max_steps") is None or audit.get("matched_supervised_token_budget") != 3 * baseline.get(
            "supervised_assistant_tokens", -1
        ):
            blockers.append("WAIT_TOKEN_MATCHED_MAX_STEPS")
        elif (
            audit.get("baseline_report") != str(baseline_path.resolve())
            or audit.get("baseline_report_sha256") != sha256_file(baseline_path)
        ):
            blockers.append("WAIT_R1_S_BASELINE_REPORT_HASH_MATCH")

    initialization = json.loads(initialization_report.read_text()) if initialization_report.is_file() else None
    if (
        initialization is None
        or initialization.get("schema_version") != INITIALIZATION_AUDIT_SCHEMA
        or not initialization.get("initialization_eligible")
    ):
        blockers.append("WAIT_STAGE8_INITIALIZATION_LOGIT_AND_FREEZE_AUDIT")
    elif (
        initialization.get("config") != str(config.resolve())
        or initialization.get("config_sha256") != sha256_file(config)
    ):
        blockers.append("WAIT_STAGE8_INITIALIZATION_CONFIG_HASH_MATCH")
    else:
        try:
            adapter_paths = validate_adapter_chain(initialization.get("adapter_chain", []))
            current_hashes = {str(path): adapter_artifact_sha256(path) for path in adapter_paths}
        except (OSError, ValueError) as exc:
            current_hashes = {}
            blockers.append(f"WAIT_STAGE8_FROZEN_ADAPTER_CHAIN: {exc}")
        if current_hashes and initialization.get("adapter_artifact_sha256") != current_hashes:
            blockers.append("WAIT_STAGE8_INITIALIZATION_ADAPTER_HASH_MATCH")

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
    print(
        json.dumps(
            {
                "run": args.run,
                "command": command,
                "token_audit": audit,
                "initialization_audit": initialization,
                "bricknet_build_report": build_report,
                "bricknet_build_gate": build_gate,
                "blockers": blockers,
            },
            indent=2,
        )
    )
    if blockers:
        raise SystemExit(2)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
