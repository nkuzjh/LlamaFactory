import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_bricknet_pt_exp2_mm_v2.py"
)
SPEC = importlib.util.spec_from_file_location("launch_bricknet_pt_exp2_mm_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _dataset_row(sample_id: str, reference: str) -> dict:
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": "build it"},
            {"role": "assistant", "content": reference},
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_alignment_rows_binds_predictions_to_standard_references() -> None:
    predictions = [
        {"predict": "0 FILE a.dat\n", "label": "0 FILE ref-a.dat"},
        {"predict": "0 FILE b.dat", "label": "0 FILE ref-b.dat\n"},
    ]
    dataset = [
        _dataset_row("sample-a", "0 FILE ref-a.dat\n"),
        _dataset_row("sample-b", "0 FILE ref-b.dat"),
    ]
    scored = [
        {"text": "0 FILE a.dat", "collisions": []},
        {"path": "0 FILE b.dat\n", "collisions": [1]},
    ]

    assert launcher._build_alignment_rows(
        predictions,
        dataset,
        scored,
        expected_samples=2,
    ) == [
        {"response": predictions[0]["predict"], "label": predictions[0]["label"]},
        {"response": predictions[1]["predict"], "label": predictions[1]["label"]},
    ]


def test_build_alignment_rows_rejects_missing_samples() -> None:
    with pytest.raises(ValueError, match="expected 2 prediction rows"):
        launcher._build_alignment_rows([], [], expected_samples=2)


def test_build_alignment_rows_rejects_reference_mismatch() -> None:
    with pytest.raises(ValueError, match="label/reference mismatch"):
        launcher._build_alignment_rows(
            [{"predict": "prediction", "label": "wrong label"}],
            [_dataset_row("sample-a", "standard reference")],
            expected_samples=1,
        )


def test_build_alignment_rows_rejects_scored_prediction_mismatch() -> None:
    with pytest.raises(ValueError, match="prediction/scored text mismatch"):
        launcher._build_alignment_rows(
            [{"predict": "prediction", "label": "reference"}],
            [_dataset_row("sample-a", "reference")],
            [{"text": "different prediction", "collisions": []}],
            expected_samples=1,
        )


def test_alignment_metrics_gate_requires_dense_reward_and_strict_success() -> None:
    assert not launcher._alignment_metrics_complete(
        {
            "task_alignment": {"samples": launcher.EXPECTED_EVAL_SAMPLES},
            "condition_generation": {"samples": launcher.EXPECTED_EVAL_SAMPLES},
        }
    )


def test_alignment_complete_rejects_stale_artifact_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_root = tmp_path / "saves"
    bricknet_root = tmp_path / "BrickNet"
    dataset = tmp_path / "BrickNet-MM_VAL.jsonl"
    evaluator = tmp_path / "evaluate_experiment.py"
    monkeypatch.setattr(launcher, "SAVE_ROOT", save_root)
    monkeypatch.setattr(launcher, "BRICKNET_ROOT", bricknet_root)
    monkeypatch.setattr(launcher, "ALIGNMENT_DATASET", dataset)
    monkeypatch.setattr(launcher, "MS_SWIFT_EVALUATOR", evaluator)

    run_name = "mm-e1"
    prediction_path = launcher._prediction_dir(run_name) / "generated_predictions.jsonl"
    output = launcher._evaluation_dir(run_name)
    metrics_path = output / "metrics.json"
    metrics_md = output / "metrics.md"
    scored_path = output / "scored.jsonl"
    base_manifest = output / "evaluation_manifest.json"
    alignment_input = launcher._alignment_input_path(run_name)
    alignment = launcher._alignment_path(run_name)
    alignment_manifest = launcher._alignment_manifest_path(run_name)
    rows = [{"value": index} for index in range(launcher.EXPECTED_EVAL_SAMPLES)]

    _write_jsonl(prediction_path, rows)
    _write_jsonl(dataset, rows)
    _write_jsonl(scored_path, rows)
    _write_json(base_manifest, {"status": "complete"})
    _write_json(
        metrics_path,
        {
            "task_alignment": {
                "samples": launcher.EXPECTED_EVAL_SAMPLES,
                "dense_reward_mean": 0.5,
                "strict_success": 1,
                "strict_success_rate": 1 / launcher.EXPECTED_EVAL_SAMPLES,
                "weights": launcher.REWARD_WEIGHTS,
                "pose_tolerances": {
                    "translation": launcher.POSE_TRANSLATION_TOLERANCE,
                    "rotation_degrees": launcher.POSE_ROTATION_TOLERANCE,
                    "success_threshold": launcher.POSE_SUCCESS_THRESHOLD,
                },
            },
            "condition_generation": {
                "samples": launcher.EXPECTED_EVAL_SAMPLES,
                "dense_reward": 0.5,
                "strict_success_num": 1,
                "strict_success_rate": 1 / launcher.EXPECTED_EVAL_SAMPLES,
            },
        },
    )
    metrics_md.write_text("complete\n", encoding="utf-8")
    _write_jsonl(alignment_input, rows)
    _write_jsonl(alignment, rows)
    evaluator.write_text("# evaluator\n", encoding="utf-8")

    identity = launcher._alignment_identity(run_name)
    assert identity is not None
    _write_json(
        alignment_manifest,
        {
            "schema_version": launcher.ALIGNMENT_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "artifacts": {
                "alignment_input_sha256": launcher._sha256(alignment_input),
                "alignment_sha256": launcher._sha256(alignment),
                "metrics_json_sha256": launcher._sha256(metrics_path),
                "metrics_md_sha256": launcher._sha256(metrics_md),
            },
        },
    )

    assert launcher._alignment_complete(run_name)
    with alignment.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"value": "stale"}) + "\n")
    assert not launcher._alignment_complete(run_name)
