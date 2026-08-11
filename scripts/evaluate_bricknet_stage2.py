#!/usr/bin/env python3
"""Run the complete BrickNet Stage-2 evaluation pipeline for one experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from launch_bricknet_stage2_sft import (
    BRICKNET_PYTHON,
    BRICKNET_ROOT,
    EXPERIMENTS,
    Experiment,
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


def _canonical_ready(paths: EvaluationPaths, variant: str) -> bool:
    report = _load_json(paths.extraction_report)
    if not paths.generated.is_file() or not paths.canonical.is_file() or report is None:
        return False
    return bool(
        report.get("variant") == variant
        and report.get("count") == 512
        and report.get("input_sha256") == _sha256(paths.generated)
        and report.get("output_sha256") == _sha256(paths.canonical)
    )


def _evaluation_complete(paths: EvaluationPaths) -> bool:
    metrics = _load_json(paths.metrics_json)
    if metrics is None or not paths.alignment.is_file():
        return False
    return bool(
        metrics.get("structure", {}).get("samples") == 512
        and metrics.get("task_alignment", {}).get("samples") == 512
        and len(_load_jsonl(paths.alignment)) == 512
    )


def _commands(
    experiment: Experiment, paths: EvaluationPaths, args: argparse.Namespace
) -> dict[str, list[str]]:
    extract = [
        "env",
        f"PYTHONPATH={BRICKNET_ROOT / 'src'}",
        str(BRICKNET_PYTHON),
        str(TRACE_EXTRACTOR),
        "--variant",
        experiment.variant,
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
    if len(rows) != 512:
        raise ValueError(f"{paths.canonical}: expected 512 rows, found {len(rows)}")
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
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else BRICKNET_ROOT / "outputs_val/qwen35_08b" / experiment.predict_output.name
    )
    paths = EvaluationPaths.for_experiment(experiment, output)
    canonical_ready = _canonical_ready(paths, experiment.variant)
    already_complete = _evaluation_complete(paths)
    commands = _commands(experiment, paths, args)
    blockers: list[str] = []
    checks = {
        "variant": experiment.variant,
        "generated_predictions": str(paths.generated),
        "generated_predictions_ready": paths.generated.is_file(),
        "canonical_predictions": str(paths.canonical),
        "canonical_predictions_ready": canonical_ready,
        "evaluation_output": str(paths.output),
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
    elif len(_load_jsonl(paths.generated)) != 512:
        blockers.append("WAIT_PREDICTION: generated prediction count is not 512")
    canonical_artifacts_exist = paths.canonical.exists() or paths.extraction_report.exists()
    if canonical_artifacts_exist and not canonical_ready and not args.force:
        blockers.append("canonical prediction artifacts are stale; pass --force to rebuild")

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
    if already_complete and not args.force and args.output_dir is None:
        print(f"Evaluation already complete: {paths.metrics_json}")
        return

    if not canonical_ready or args.force:
        subprocess.run(commands["extract"], cwd=BRICKNET_ROOT, check=True)
    _prepare_metric_inputs(paths)
    subprocess.run(commands["text"], cwd=MS_SWIFT_ROOT, check=True)
    subprocess.run(commands["evaluate"], cwd=BRICKNET_ROOT, check=True)
    subprocess.run(commands["alignment"], cwd=MS_SWIFT_ROOT, check=True)
    if not _evaluation_complete(paths):
        raise RuntimeError("Stage-2 evaluation finished without complete 512-row metrics")
    print(f"Stage-2 evaluation complete: {paths.metrics_json}")


if __name__ == "__main__":
    main()
