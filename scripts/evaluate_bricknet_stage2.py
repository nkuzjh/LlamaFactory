#!/usr/bin/env python3
"""Run the complete BrickNet Stage-2 evaluation pipeline for one experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from launch_bricknet_stage2_sft import (
    BRICKNET_PYTHON,
    BRICKNET_ROOT,
    EXPERIMENTS,
    Experiment,
    _trace_variant,
)


LLAMAFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")
MS_SWIFT_ROOT = Path("/home/jiahao/task/ms-swift")
MS_SWIFT_EVALUATOR = (
    MS_SWIFT_ROOT / "examples/train/grpo/plugin/bricknet/evaluate_experiment.py"
)
BRICKNET_EVALUATOR = BRICKNET_ROOT / "scripts/evaluate_experiment.py"
TRACE_EXTRACTOR = BRICKNET_ROOT / "scripts/extract_reasoning_predictions.py"
PROMPTS = BRICKNET_ROOT / "data/bricknet_datasets/captions_val.jsonl"
ALIGNMENT_DATASET = (
    BRICKNET_ROOT / "outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.jsonl"
)
EXPECTED_SAMPLES = 512
ALIGNMENT_SCHEMA_VERSION = "bricknet-stage2-alignment-v1"
POSE_TOLERANCES = {
    "translation": 0.5,
    "rotation_degrees": 5.0,
    "success_threshold": 1.0,
}
REWARD_WEIGHTS = {
    "parse_prefix": 0.20,
    "inventory_f1": 0.20,
    "length_score": 0.10,
    "collision_prefix": 0.20,
    "pose_match": 0.30,
}
EXPERIMENTS_BY_ID = {
    experiment.experiment_id: experiment for experiment in EXPERIMENTS.values()
}


@dataclass(frozen=True)
class EvaluationPaths:
    generated: Path
    canonical: Path
    extraction_report: Path
    output: Path
    text_input: Path
    text_metrics: Path
    alignment_input: Path
    alignment: Path
    scored: Path
    metrics_json: Path
    metrics_md: Path
    alignment_manifest: Path

    @classmethod
    def for_experiment(
        cls, experiment: Experiment, output: Path
    ) -> "EvaluationPaths":
        prediction = experiment.predict_output
        return cls(
            generated=prediction / "generated_predictions.jsonl",
            canonical=prediction / "path_predictions.jsonl",
            extraction_report=prediction / "trace_extraction_report.json",
            output=output,
            text_input=output / "path_text_input.jsonl",
            text_metrics=output / "path_text_metrics.json",
            alignment_input=output / "alignment_input.jsonl",
            alignment=output / "alignment.jsonl",
            scored=output / "scored.jsonl",
            metrics_json=output / "metrics.json",
            metrics_md=output / "metrics.md",
            alignment_manifest=output / "alignment_manifest.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS_BY_ID), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-image-metrics", action="store_true")
    parser.add_argument("--render-jobs", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()
    for name in ("render_jobs", "eval_workers", "eval_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_ready(paths: EvaluationPaths, variant: str) -> bool:
    report = _load_json(paths.extraction_report)
    if not paths.generated.is_file() or not paths.canonical.is_file() or report is None:
        return False
    return bool(
        report.get("variant") == variant
        and report.get("count") == EXPECTED_SAMPLES
        and report.get("input_sha256") == _sha256(paths.generated)
        and report.get("output_sha256") == _sha256(paths.canonical)
    )


def _is_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metrics_complete(metrics: dict[str, Any]) -> bool:
    structure = metrics.get("structure", {})
    task_alignment = metrics.get("task_alignment", {})
    condition_generation = metrics.get("condition_generation", {})
    dense_reward = task_alignment.get("dense_reward_mean")
    strict_success = task_alignment.get("strict_success")
    strict_success_rate = task_alignment.get("strict_success_rate")
    return bool(
        structure.get("samples") == EXPECTED_SAMPLES
        and task_alignment.get("samples") == EXPECTED_SAMPLES
        and condition_generation.get("samples") == EXPECTED_SAMPLES
        and metrics.get("artifacts", {}).get("predictions") == EXPECTED_SAMPLES
        and metrics.get("artifacts", {}).get("scored") == EXPECTED_SAMPLES
        and _is_number(dense_reward)
        and isinstance(strict_success, int)
        and not isinstance(strict_success, bool)
        and 0 <= strict_success <= EXPECTED_SAMPLES
        and _is_number(strict_success_rate)
        and 0.0 <= float(strict_success_rate) <= 1.0
        and strict_success_rate == strict_success / EXPECTED_SAMPLES
        and task_alignment.get("weights") == REWARD_WEIGHTS
        and task_alignment.get("pose_tolerances") == POSE_TOLERANCES
        and condition_generation.get("dense_reward") == dense_reward
        and condition_generation.get("strict_success_num") == strict_success
        and condition_generation.get("strict_success_rate") == strict_success_rate
    )


def _same_resolved_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except OSError:
        return False


def _base_evaluation_complete(paths: EvaluationPaths) -> bool:
    manifest_path = paths.output / "evaluation_manifest.json"
    manifest = _load_json(manifest_path)
    metrics = _load_json(paths.metrics_json)
    if manifest is None or metrics is None or not paths.scored.is_file():
        return False
    try:
        scored_count = len(_load_jsonl(paths.scored))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("samples") == EXPECTED_SAMPLES
        and _same_resolved_path(manifest.get("predictions"), paths.canonical)
        and manifest.get("predictions_sha256") == _sha256(paths.canonical)
        and _same_resolved_path(manifest.get("metrics"), paths.metrics_json)
        and scored_count == EXPECTED_SAMPLES
        and _metrics_complete(metrics)
    )


def _metric_inputs_complete(paths: EvaluationPaths) -> bool:
    if not (paths.canonical.is_file() and paths.alignment_input.is_file()):
        return False
    try:
        canonical_rows = _load_jsonl(paths.canonical)
        alignment_input_rows = _load_jsonl(paths.alignment_input)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if len(canonical_rows) != EXPECTED_SAMPLES:
        return False
    expected_rows: list[dict[str, str]] = []
    for row in canonical_rows:
        prediction = row.get("predict")
        label = row.get("label")
        if not isinstance(prediction, str) or not isinstance(label, str):
            return False
        if "<think>" in prediction:
            return False
        expected_rows.append({"response": prediction, "label": label})
    return alignment_input_rows == expected_rows


def _normalize_path_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _assistant_reference(row: dict[str, Any], index: int) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    assistants = [
        item.get("content")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    if not assistants or not isinstance(assistants[-1], str):
        return None
    return assistants[-1]


def _close(left: Any, right: Any) -> bool:
    return bool(
        _is_number(left)
        and _is_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _alignment_artifacts_consistent(paths: EvaluationPaths) -> bool:
    metrics = _load_json(paths.metrics_json)
    if metrics is None:
        return False
    try:
        canonical = _load_jsonl(paths.canonical)
        alignment_input = _load_jsonl(paths.alignment_input)
        dataset = _load_jsonl(ALIGNMENT_DATASET)
        scored = _load_jsonl(paths.scored)
        alignment = _load_jsonl(paths.alignment)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not all(
        len(rows) == EXPECTED_SAMPLES
        for rows in (canonical, alignment_input, dataset, scored, alignment)
    ):
        return False

    score_keys = (
        "parse_prefix",
        "inventory_precision",
        "inventory_recall",
        "inventory_f1",
        "length_score",
        "collision_prefix",
        "pose_match",
        "dense_reward",
    )
    mean_fields = {
        "parse_prefix": "parse_prefix_mean",
        "inventory_precision": "inventory_precision_mean",
        "inventory_recall": "inventory_recall_mean",
        "inventory_f1": "inventory_f1_mean",
        "length_score": "length_score_mean",
        "collision_prefix": "collision_prefix_mean",
        "pose_match": "pose_match_mean",
        "dense_reward": "dense_reward_mean",
    }
    values: dict[str, list[float]] = {key: [] for key in score_keys}
    strict_success = 0
    seen_ids: set[Any] = set()
    for index, (canonical_row, input_row, dataset_row, scored_row, alignment_row) in enumerate(
        zip(canonical, alignment_input, dataset, scored, alignment)
    ):
        prediction = canonical_row.get("predict")
        label = canonical_row.get("label")
        reference = _assistant_reference(dataset_row, index)
        if not all(isinstance(item, str) for item in (prediction, label, reference)):
            return False
        assert isinstance(prediction, str) and isinstance(label, str) and isinstance(reference, str)
        if _normalize_path_text(label) != _normalize_path_text(reference):
            return False
        if input_row != {"response": prediction, "label": label}:
            return False

        scored_text = scored_row.get("text", scored_row.get("path"))
        collisions = scored_row.get("collisions")
        if (
            scored_row.get("id") != index
            or not isinstance(scored_text, str)
            or _normalize_path_text(scored_text) != _normalize_path_text(prediction)
            or not isinstance(collisions, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in collisions
            )
        ):
            return False

        expected_id = dataset_row.get("id", index)
        if expected_id in seen_ids:
            return False
        seen_ids.add(expected_id)
        if (
            alignment_row.get("index") != index
            or alignment_row.get("id") != expected_id
            or alignment_row.get("collisions") != collisions
        ):
            return False
        for key in score_keys:
            value = alignment_row.get(key)
            if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
                return False
            values[key].append(float(value))
        strict_value = alignment_row.get("strict_success")
        if not isinstance(strict_value, bool):
            return False
        strict_success += int(strict_value)
        expected_dense = sum(
            REWARD_WEIGHTS[key] * float(alignment_row[key])
            for key in REWARD_WEIGHTS
        )
        if not _close(alignment_row["dense_reward"], expected_dense):
            return False

    task_alignment = metrics.get("task_alignment", {})
    for key, metric_key in mean_fields.items():
        mean = math.fsum(values[key]) / EXPECTED_SAMPLES
        if not _close(task_alignment.get(metric_key), mean):
            return False
    return bool(
        task_alignment.get("strict_success") == strict_success
        and _close(
            task_alignment.get("strict_success_rate"),
            strict_success / EXPECTED_SAMPLES,
        )
    )


def _numeric_evaluation_complete(
    experiment: Experiment, paths: EvaluationPaths
) -> bool:
    required = (
        paths.generated,
        paths.canonical,
        paths.extraction_report,
        paths.scored,
        paths.alignment_input,
        paths.alignment,
        paths.metrics_json,
        paths.metrics_md,
        paths.output / "evaluation_manifest.json",
        ALIGNMENT_DATASET,
        MS_SWIFT_EVALUATOR,
    )
    if not all(path.is_file() for path in required):
        return False
    if not _canonical_ready(paths, _trace_variant(experiment)):
        return False
    if (
        not _base_evaluation_complete(paths)
        or not _metric_inputs_complete(paths)
        or not _alignment_artifacts_consistent(paths)
    ):
        return False
    try:
        return bool(
            len(_load_jsonl(paths.generated)) == EXPECTED_SAMPLES
            and len(_load_jsonl(paths.scored)) == EXPECTED_SAMPLES
            and len(_load_jsonl(paths.alignment)) == EXPECTED_SAMPLES
            and len(_load_jsonl(ALIGNMENT_DATASET)) == EXPECTED_SAMPLES
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False


def _jsonl_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "count": len(_load_jsonl(path)),
    }


def _file_artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _manifest_sections(
    experiment: Experiment, paths: EvaluationPaths
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = _load_json(paths.metrics_json)
    if metrics is None:
        raise ValueError(f"{paths.metrics_json}: expected a JSON object")
    identity = {
        "experiment": experiment.experiment_id,
        "variant": experiment.variant,
        "trace_variant": _trace_variant(experiment),
        "expected_samples": EXPECTED_SAMPLES,
        "generated_predictions": _jsonl_artifact(paths.generated),
        "canonical_predictions": _jsonl_artifact(paths.canonical),
        "scored": _jsonl_artifact(paths.scored),
        "alignment_dataset": _jsonl_artifact(ALIGNMENT_DATASET),
        "alignment_evaluator": _file_artifact(MS_SWIFT_EVALUATOR),
        "base_evaluation_manifest": _file_artifact(
            paths.output / "evaluation_manifest.json"
        ),
        "reward_weights": REWARD_WEIGHTS,
        "pose_tolerances": POSE_TOLERANCES,
    }
    artifacts = {
        "alignment_input": _jsonl_artifact(paths.alignment_input),
        "alignment": _jsonl_artifact(paths.alignment),
        "metrics_json": _file_artifact(paths.metrics_json),
        "metrics_md": _file_artifact(paths.metrics_md),
    }
    summary = {
        "structure_samples": metrics["structure"]["samples"],
        "task_alignment_samples": metrics["task_alignment"]["samples"],
        "condition_generation_samples": metrics["condition_generation"]["samples"],
        "dense_reward_mean": metrics["task_alignment"]["dense_reward_mean"],
        "strict_success": metrics["task_alignment"]["strict_success"],
        "strict_success_rate": metrics["task_alignment"]["strict_success_rate"],
    }
    return identity, artifacts, summary


def _valid_completed_at(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _alignment_manifest_complete(
    experiment: Experiment, paths: EvaluationPaths
) -> bool:
    if not _numeric_evaluation_complete(experiment, paths):
        return False
    manifest = _load_json(paths.alignment_manifest)
    if manifest is None:
        return False
    try:
        identity, artifacts, summary = _manifest_sections(experiment, paths)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError):
        return False
    return bool(
        set(manifest) == {
            "schema_version",
            "status",
            "completed_at",
            "freeze",
            "identity",
            "artifacts",
            "summary",
        }
        and manifest.get("schema_version") == ALIGNMENT_SCHEMA_VERSION
        and manifest.get("status") == "complete"
        and _valid_completed_at(manifest.get("completed_at"))
        and manifest.get("freeze")
        in (
            {"mode": "verified_numeric_backfill"},
            {"mode": "post_evaluation_freeze"},
        )
        and manifest.get("identity") == identity
        and manifest.get("artifacts") == artifacts
        and manifest.get("summary") == summary
    )


def _write_alignment_manifest(
    experiment: Experiment, paths: EvaluationPaths, *, freeze_mode: str
) -> None:
    if freeze_mode not in ("verified_numeric_backfill", "post_evaluation_freeze"):
        raise ValueError(f"unsupported alignment manifest freeze mode: {freeze_mode}")
    if not _numeric_evaluation_complete(experiment, paths):
        raise RuntimeError("cannot freeze an incomplete Stage-2 numeric evaluation")
    identity, artifacts, summary = _manifest_sections(experiment, paths)
    _write_json(
        paths.alignment_manifest,
        {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "freeze": {"mode": freeze_mode},
            "identity": identity,
            "artifacts": artifacts,
            "summary": summary,
        },
    )
    if not _alignment_manifest_complete(experiment, paths):
        raise RuntimeError("alignment manifest failed its post-write completion gate")


def _commands(
    experiment: Experiment, paths: EvaluationPaths, args: argparse.Namespace
) -> dict[str, list[str]]:
    extract = [
        "env",
        f"PYTHONPATH={BRICKNET_ROOT / 'src'}",
        str(BRICKNET_PYTHON),
        str(TRACE_EXTRACTOR),
        "--variant",
        _trace_variant(experiment),
        "--label-format",
        "path",
        "--input",
        str(paths.generated),
        "--output",
        str(paths.canonical),
        "--report",
        str(paths.extraction_report),
    ]
    if args.force:
        extract.append("--overwrite")

    text = [
        str(LLAMAFACTORY_PYTHON),
        str(MS_SWIFT_EVALUATOR),
        "text-worker",
        "--results",
        str(paths.text_input),
        "--output",
        str(paths.text_metrics),
    ]
    evaluate = [
        str(BRICKNET_PYTHON),
        str(BRICKNET_EVALUATOR),
        "--predictions",
        str(paths.canonical),
        "--text-metrics",
        str(paths.text_metrics),
        "--output-dir",
        str(paths.output),
        "--prompts-file",
        str(PROMPTS),
        "--render-jobs",
        str(args.render_jobs),
        "--eval-workers",
        str(args.eval_workers),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    if args.skip_image_metrics:
        evaluate.append("--skip-image-metrics")
    if args.force:
        evaluate.append("--force")

    alignment = [
        str(BRICKNET_PYTHON),
        str(MS_SWIFT_EVALUATOR),
        "alignment-worker",
        "--results",
        str(paths.alignment_input),
        "--dataset",
        str(ALIGNMENT_DATASET),
        "--scored",
        str(paths.scored),
        "--metrics-json",
        str(paths.metrics_json),
        "--metrics-md",
        str(paths.metrics_md),
        "--output",
        str(paths.alignment),
        "--bricknet-root",
        str(BRICKNET_ROOT),
    ]
    return {"extract": extract, "text": text, "evaluate": evaluate, "alignment": alignment}


def _prepare_metric_inputs(paths: EvaluationPaths) -> None:
    rows = _load_jsonl(paths.canonical)
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{paths.canonical}: expected {EXPECTED_SAMPLES} rows, found {len(rows)}"
        )
    if any("<think>" in str(row.get("predict", "")) for row in rows):
        raise ValueError(f"{paths.canonical}: reasoning trace leaked into canonical paths")
    if any(
        not isinstance(row.get("predict"), str)
        or not isinstance(row.get("label"), str)
        for row in rows
    ):
        raise ValueError(f"{paths.canonical}: predict and label must be strings")
    _write_jsonl(
        paths.text_input,
        [{"response": row["predict"], "labels": row["label"]} for row in rows],
    )
    _write_jsonl(
        paths.alignment_input,
        [{"response": row["predict"], "label": row["label"]} for row in rows],
    )


def main() -> None:
    args = parse_args()
    experiment = EXPERIMENTS_BY_ID[args.experiment]
    trace_variant = _trace_variant(experiment)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else BRICKNET_ROOT / "outputs_val/qwen35_08b" / experiment.predict_output.name
    )
    paths = EvaluationPaths.for_experiment(experiment, output)
    canonical_ready = _canonical_ready(paths, trace_variant)
    numeric_complete = _numeric_evaluation_complete(experiment, paths)
    already_complete = _alignment_manifest_complete(experiment, paths)
    commands = _commands(experiment, paths, args)
    blockers: list[str] = []
    checks = {
        "variant": experiment.variant,
        "trace_variant": trace_variant,
        "generated_predictions": str(paths.generated),
        "generated_predictions_ready": paths.generated.is_file(),
        "canonical_predictions": str(paths.canonical),
        "canonical_predictions_ready": canonical_ready,
        "evaluation_output": str(paths.output),
        "numeric_evaluation_complete": numeric_complete,
        "alignment_manifest": str(paths.alignment_manifest),
        "alignment_manifest_complete": already_complete,
        "evaluation_complete": already_complete,
    }
    required = {
        "bricknet_python": BRICKNET_PYTHON,
        "llamafactory_python": LLAMAFACTORY_PYTHON,
        "trace_extractor": TRACE_EXTRACTOR,
        "bricknet_evaluator": BRICKNET_EVALUATOR,
        "alignment_evaluator": MS_SWIFT_EVALUATOR,
        "prompts": PROMPTS,
        "alignment_dataset": ALIGNMENT_DATASET,
    }
    for name, path in required.items():
        checks[f"{name}_ready"] = path.is_file()
        if not path.is_file():
            blockers.append(f"missing {name}: {path}")
    if not paths.generated.is_file():
        blockers.append("WAIT_PREDICTION: generated_predictions.jsonl is missing")
    elif len(_load_jsonl(paths.generated)) != EXPECTED_SAMPLES:
        blockers.append(
            f"WAIT_PREDICTION: generated prediction count is not {EXPECTED_SAMPLES}"
        )
    canonical_artifacts_exist = paths.canonical.exists() or paths.extraction_report.exists()
    if canonical_artifacts_exist and not canonical_ready and not args.force:
        blockers.append("canonical prediction artifacts are stale; pass --force to rebuild")
    if paths.alignment_manifest.exists() and not already_complete and not args.force:
        blockers.append(
            "INVALID_ALIGNMENT_MANIFEST_REVIEW_REQUIRED: pass --force only after review"
        )

    payload = {
        "experiment": experiment.experiment_id,
        "mode": "execute" if args.execute else "dry-run",
        "ready": not blockers,
        "already_complete": already_complete,
        "checks": checks,
        "blockers": blockers,
        "commands": {name: shlex.join(command) for name, command in commands.items()},
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if blockers:
        raise SystemExit("Stage-2 evaluation blocked; resolve the reported checks first")
    if already_complete and not args.force:
        print(f"Evaluation already complete: {paths.metrics_json}")
        return

    # A prior evaluator version could finish every numeric stage but omit the
    # final provenance manifest.  In that exact state, freeze the verified
    # artifacts only; do not re-run extraction, text/image evaluation, render,
    # or alignment.
    if numeric_complete and not args.force:
        _write_alignment_manifest(
            experiment, paths, freeze_mode="verified_numeric_backfill"
        )
        print(f"Stage-2 alignment manifest backfilled: {paths.alignment_manifest}")
        return

    if not canonical_ready or args.force:
        subprocess.run(commands["extract"], cwd=BRICKNET_ROOT, check=True)
    _prepare_metric_inputs(paths)
    subprocess.run(commands["text"], cwd=MS_SWIFT_ROOT, check=True)
    subprocess.run(commands["evaluate"], cwd=BRICKNET_ROOT, check=True)
    subprocess.run(commands["alignment"], cwd=MS_SWIFT_ROOT, check=True)
    if not _numeric_evaluation_complete(experiment, paths):
        raise RuntimeError(
            f"Stage-2 evaluation finished without complete {EXPECTED_SAMPLES}-row metrics"
        )
    _write_alignment_manifest(
        experiment, paths, freeze_mode="post_evaluation_freeze"
    )
    print(f"Stage-2 evaluation complete: {paths.metrics_json}")


if __name__ == "__main__":
    main()
