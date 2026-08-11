#!/usr/bin/env python3
"""Gate-protected launcher and selector for the BrickNet PT-exp2 branch.

The default is a read-only status/dry run.  Training, prediction, evaluation,
50k materialization, alias creation, and scale approval all require explicit
``--execute``; final selection and scale approvals additionally require
``--approve``.  No PT-exp2 VAL511 train or validation run exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    "text8m": {"per_device_batch_size": 4, "global_batch_size": 32},
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
        output = SAVE_ROOT / run.output
        checks["output"] = str(output)
        checks["already_complete"] = _adapter_ready(output)
        if output.exists() and not checks["already_complete"]:
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
    command = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "llamafactory-cli",
        "train",
        str(CONFIG_ROOT / config_name),
    ]
    selectors = _visible_gpu_selectors()
    if args.action == "train" and selectors and len(selectors) in SUPPORTED_TRAIN_WORLD_SIZES:
        profile = _train_batch_profile(args.run, len(selectors))
        command.append(f"gradient_accumulation_steps={profile['gradient_accumulation_steps']}")
    env = os.environ.copy()
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
    if launch_env:
        display_command = ["env", *[f"{key}={value}" for key, value in launch_env.items()], *command]
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
    subprocess.run(command, cwd=ROOT, env=env, check=True)


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
    if args.action == "predict" and args.run not in PREDICT_CONFIGS:
        parser.error(f"{args.run} has no predict configuration")
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
