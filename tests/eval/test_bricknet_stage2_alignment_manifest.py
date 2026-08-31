import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_script("test_evaluate_bricknet_stage2", "evaluate_bricknet_stage2.py")
downstream = _load_script(
    "test_launch_bricknet_pt_exp2_v2_downstream",
    "launch_bricknet_pt_exp2_v2_downstream.py",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_id: str = "exp4_4",
    variant: str = "nonthinking-control",
):
    bricknet_root = tmp_path / "BrickNet"
    prediction = tmp_path / "predictions" / f"eval_{experiment_id}"
    output = bricknet_root / "outputs_val/qwen35_08b" / prediction.name
    dataset = tmp_path / "BrickNet-MM_VAL.jsonl"
    alignment_evaluator = tmp_path / "ms-swift/evaluate_experiment.py"
    alignment_evaluator.parent.mkdir(parents=True, exist_ok=True)
    alignment_evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
    monkeypatch.setattr(evaluator, "ALIGNMENT_DATASET", dataset)
    monkeypatch.setattr(evaluator, "MS_SWIFT_EVALUATOR", alignment_evaluator)

    experiment = SimpleNamespace(
        experiment_id=experiment_id,
        variant=variant,
        predict_output=prediction,
    )
    paths = evaluator.EvaluationPaths.for_experiment(experiment, output)
    generated = [{"raw": index} for index in range(evaluator.EXPECTED_SAMPLES)]
    canonical = [
        {"predict": f"a brick-{index} | red\n", "label": f"a brick-{index} | red\n"}
        for index in range(evaluator.EXPECTED_SAMPLES)
    ]
    dataset_rows = [
        {
            "id": f"sample-{index:03d}",
            "messages": [
                {"role": "user", "content": "build"},
                {"role": "assistant", "content": canonical[index]["label"]},
            ],
        }
        for index in range(evaluator.EXPECTED_SAMPLES)
    ]
    scored = [
        {"id": index, "text": canonical[index]["predict"], "collisions": []}
        for index in range(evaluator.EXPECTED_SAMPLES)
    ]
    alignment_input = [
        {"response": row["predict"], "label": row["label"]} for row in canonical
    ]
    alignment = [
        {
            "index": index,
            "id": dataset_rows[index]["id"],
            "collisions": [],
            "parse_prefix": 1.0,
            "inventory_precision": 1.0,
            "inventory_recall": 1.0,
            "inventory_f1": 1.0,
            "length_score": 1.0,
            "collision_prefix": 1.0,
            "pose_match": 1.0,
            "dense_reward": 1.0,
            "strict_success": True,
        }
        for index in range(evaluator.EXPECTED_SAMPLES)
    ]
    metrics = {
        "structure": {"samples": evaluator.EXPECTED_SAMPLES},
        "artifacts": {
            "predictions": evaluator.EXPECTED_SAMPLES,
            "scored": evaluator.EXPECTED_SAMPLES,
        },
        "task_alignment": {
            "samples": evaluator.EXPECTED_SAMPLES,
            "parse_prefix_mean": 1.0,
            "inventory_precision_mean": 1.0,
            "inventory_recall_mean": 1.0,
            "inventory_f1_mean": 1.0,
            "length_score_mean": 1.0,
            "collision_prefix_mean": 1.0,
            "pose_match_mean": 1.0,
            "dense_reward_mean": 1.0,
            "strict_success": evaluator.EXPECTED_SAMPLES,
            "strict_success_rate": 1.0,
            "weights": evaluator.REWARD_WEIGHTS,
            "pose_tolerances": evaluator.POSE_TOLERANCES,
        },
        "condition_generation": {
            "samples": evaluator.EXPECTED_SAMPLES,
            "dense_reward": 1.0,
            "strict_success_num": evaluator.EXPECTED_SAMPLES,
            "strict_success_rate": 1.0,
        },
    }
    _write_jsonl(paths.generated, generated)
    _write_jsonl(paths.canonical, canonical)
    _write_json(
        paths.extraction_report,
        {
            "variant": variant,
            "count": evaluator.EXPECTED_SAMPLES,
            "input_sha256": _sha256(paths.generated),
            "output_sha256": _sha256(paths.canonical),
        },
    )
    _write_jsonl(dataset, dataset_rows)
    _write_jsonl(paths.scored, scored)
    _write_jsonl(paths.alignment_input, alignment_input)
    _write_jsonl(paths.alignment, alignment)
    _write_json(paths.metrics_json, metrics)
    paths.metrics_md.write_text("complete\n", encoding="utf-8")
    _write_json(
        output / "evaluation_manifest.json",
        {
            "status": "complete",
            "samples": evaluator.EXPECTED_SAMPLES,
            "predictions": str(paths.canonical.resolve()),
            "predictions_sha256": _sha256(paths.canonical),
            "metrics": str(paths.metrics_json.resolve()),
        },
    )
    return experiment, paths, dataset, alignment_evaluator


def _main_args(*, execute: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        experiment="exp4_4",
        output_dir=None,
        execute=execute,
        force=False,
        skip_image_metrics=False,
        render_jobs=1,
        eval_workers=1,
        eval_batch_size=1,
    )


def test_execute_backfills_only_manifest_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluator, "BRICKNET_ROOT", tmp_path / "BrickNet")
    output = evaluator.BRICKNET_ROOT / "outputs_val/qwen35_08b" / paths.output.name
    assert output == paths.output
    monkeypatch.setattr(evaluator, "EXPERIMENTS_BY_ID", {"exp4_4": experiment})
    monkeypatch.setattr(evaluator, "parse_args", lambda: _main_args())
    for name in (
        "BRICKNET_PYTHON",
        "LLAMAFACTORY_PYTHON",
        "TRACE_EXTRACTOR",
        "BRICKNET_EVALUATOR",
        "PROMPTS",
    ):
        path = tmp_path / f"required/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready\n", encoding="utf-8")
        monkeypatch.setattr(evaluator, name, path)
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("numeric backfill must not run a worker"),
    )
    before = {
        path: _sha256(path)
        for path in paths.output.iterdir()
        if path.is_file()
    }

    evaluator.main()

    assert evaluator._alignment_manifest_complete(experiment, paths)
    manifest = json.loads(paths.alignment_manifest.read_text(encoding="utf-8"))
    assert manifest["freeze"] == {"mode": "verified_numeric_backfill"}
    assert {
        path: _sha256(path)
        for path in before
    } == before


def test_invalid_existing_manifest_is_not_silently_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    _write_json(paths.alignment_manifest, {"status": "complete", "wrong": True})
    before = paths.alignment_manifest.read_bytes()
    monkeypatch.setattr(evaluator, "BRICKNET_ROOT", tmp_path / "BrickNet")
    monkeypatch.setattr(evaluator, "EXPERIMENTS_BY_ID", {"exp4_4": experiment})
    monkeypatch.setattr(evaluator, "parse_args", lambda: _main_args())
    for name in (
        "BRICKNET_PYTHON",
        "LLAMAFACTORY_PYTHON",
        "TRACE_EXTRACTOR",
        "BRICKNET_EVALUATOR",
        "PROMPTS",
    ):
        path = tmp_path / f"required/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready\n", encoding="utf-8")
        monkeypatch.setattr(evaluator, name, path)

    with pytest.raises(SystemExit, match="evaluation blocked"):
        evaluator.main()

    assert paths.alignment_manifest.read_bytes() == before


def test_backfill_rejects_cross_artifact_row_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    rows = evaluator._load_jsonl(paths.alignment)
    rows[0]["id"] = "wrong-sample"
    _write_jsonl(paths.alignment, rows)

    assert not evaluator._numeric_evaluation_complete(experiment, paths)


def test_fresh_evaluation_path_writes_post_evaluation_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    alignment_rows = evaluator._load_jsonl(paths.alignment)
    paths.alignment.unlink()
    monkeypatch.setattr(evaluator, "BRICKNET_ROOT", tmp_path / "BrickNet")
    monkeypatch.setattr(evaluator, "EXPERIMENTS_BY_ID", {"exp4_4": experiment})
    monkeypatch.setattr(evaluator, "parse_args", lambda: _main_args())
    for name in (
        "BRICKNET_PYTHON",
        "LLAMAFACTORY_PYTHON",
        "TRACE_EXTRACTOR",
        "BRICKNET_EVALUATOR",
        "PROMPTS",
    ):
        path = tmp_path / f"required/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready\n", encoding="utf-8")
        monkeypatch.setattr(evaluator, name, path)

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> None:
        calls.append(command)
        if "alignment-worker" in command:
            _write_jsonl(paths.alignment, alignment_rows)

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    evaluator.main()

    assert len(calls) == 3
    assert evaluator._alignment_manifest_complete(experiment, paths)
    manifest = json.loads(paths.alignment_manifest.read_text(encoding="utf-8"))
    assert manifest["freeze"] == {"mode": "post_evaluation_freeze"}


def test_manifest_hash_drift_fails_completion_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    evaluator._write_alignment_manifest(
        experiment, paths, freeze_mode="verified_numeric_backfill"
    )
    assert evaluator._alignment_manifest_complete(experiment, paths)

    paths.metrics_md.write_text("changed after freeze\n", encoding="utf-8")

    assert not evaluator._alignment_manifest_complete(experiment, paths)


@pytest.mark.parametrize(
    ("experiment_id", "variant"),
    (
        ("exp4_4", "nonthinking-control"),
        ("exp4_7", "thinking-hard-v2-lean-state"),
    ),
)
def test_downstream_gate_validates_full_manifest_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_id: str,
    variant: str,
) -> None:
    experiment, paths, dataset, alignment_evaluator = _numeric_tree(
        tmp_path,
        monkeypatch,
        experiment_id=experiment_id,
        variant=variant,
    )
    evaluator._write_alignment_manifest(
        experiment, paths, freeze_mode="verified_numeric_backfill"
    )
    run = downstream.Run(
        experiment=experiment_id,
        variant=variant,
        train_dataset="train",
        eval_dataset="eval",
        train_file=tmp_path / "train.jsonl",
        eval_file=tmp_path / "eval.jsonl",
        train_config=tmp_path / "train.yaml",
        predict_config=tmp_path / "predict.yaml",
        train_output=tmp_path / "adapter",
        predict_output=experiment.predict_output,
        cache_path=tmp_path / "cache",
        train_audit_key="train",
        eval_audit_key="eval",
    )
    monkeypatch.setattr(downstream, "EVALUATION_ROOT", paths.output.parent)
    monkeypatch.setattr(downstream, "ALIGNMENT_DATASET", dataset)
    monkeypatch.setattr(downstream, "MS_SWIFT_EVALUATOR", alignment_evaluator)

    assert downstream._evaluation_complete(run)
    manifest = json.loads(paths.alignment_manifest.read_text(encoding="utf-8"))
    manifest["summary"]["strict_success"] = 0
    _write_json(paths.alignment_manifest, manifest)
    assert not downstream._evaluation_complete(run)


@pytest.mark.parametrize("execute", (False, True))
def test_downstream_evaluate_command_uses_one_valid_conda_environment(
    execute: bool,
) -> None:
    command = downstream._command(downstream.RUNS["exp4_4"], "evaluate", execute)
    expected = [
        "conda",
        "run",
        "-n",
        "llamafactory",
        "--no-capture-output",
        "python",
        "scripts/evaluate_bricknet_stage2.py",
        "--experiment",
        "exp4_4",
    ]
    if execute:
        expected.append("--execute")

    assert command == expected
    assert command.count("-n") == 1
    assert ("--execute" in command) is execute


def test_bool_or_nan_alignment_scores_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, paths, _, _ = _numeric_tree(tmp_path, monkeypatch)
    rows = evaluator._load_jsonl(paths.alignment)
    rows[0]["dense_reward"] = True
    _write_jsonl(paths.alignment, rows)
    assert not evaluator._numeric_evaluation_complete(experiment, paths)

    experiment, paths, _, _ = _numeric_tree(tmp_path / "nan", monkeypatch)
    rows = evaluator._load_jsonl(paths.alignment)
    rows[0]["dense_reward"] = float("nan")
    _write_jsonl(paths.alignment, rows)
    assert not evaluator._numeric_evaluation_complete(experiment, paths)
