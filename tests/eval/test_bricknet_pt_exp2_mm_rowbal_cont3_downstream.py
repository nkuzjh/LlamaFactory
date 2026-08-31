import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py"
SPEC = importlib.util.spec_from_file_location("launch_rowbal_downstream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_two_runs_bind_ep3_parent_and_distinct_new_namespaces() -> None:
    assert tuple(launcher.RUNS) == ("exp4_4_3", "exp4_7_3")
    assert launcher.ROWBAL_PARENT_RELATIVE.endswith("/checkpoint-50646")
    for name, run in launcher.RUNS.items():
        train = yaml.safe_load(run.train_config.read_text(encoding="utf-8"))
        predict = yaml.safe_load(run.predict_config.read_text(encoding="utf-8"))
        assert train["adapter_name_or_path"] == launcher.ROWBAL_PARENT_RELATIVE
        assert predict["adapter_name_or_path"] == (
            f"{launcher.ROWBAL_PARENT_RELATIVE},{run.train_output.relative_to(ROOT)}"
        )
        assert run.train_output.name.startswith(f"train_{name}_")
        assert run.predict_output.name.startswith(f"eval_{name}_")
        assert "rowbal_cont3" in run.train_output.name
        assert "rowbal_cont3" in run.predict_output.name
        assert "PT-exp2-v2" not in predict["adapter_name_or_path"]
        assert not any(token in predict["adapter_name_or_path"].lower() for token in ("mm_e1", "mm_e2", "mm_e3"))


@pytest.mark.parametrize("run_name", ("exp4_4_3", "exp4_7_3"))
@pytest.mark.parametrize("action", ("train", "predict"))
def test_yaml_contract_is_exact(run_name: str, action: str) -> None:
    run = launcher.RUNS[run_name]
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_config(run, action, checks, blockers)
    assert blockers == []
    assert checks["config_sha256"]["matches"] is True
    assert checks["config_fields"]["mismatches"] == {}
    config = yaml.safe_load(Path(checks["config"]).read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["cutoff_len"] == 16384
    if action == "predict":
        assert config["max_new_tokens"] == 16384


def test_config_file_hash_drift_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    run = launcher.RUNS["exp4_4_3"]
    monkeypatch.setattr(launcher, "_sha256", lambda path: "config-drift")
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_config(run, "train", checks, blockers)
    assert "CONFIG_FILE_HASH_DRIFT" in blockers
    assert checks["config_sha256"]["matches"] is False


def test_frozen_ep3_parent_checkpoint_and_run_manifest_are_valid() -> None:
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_rowbal_parent(checks, blockers)
    assert blockers == []
    assert checks["rowbal_parent_trainer_state_ok"] is True
    assert checks["rowbal_parent_output_is_symlink"] is False
    assert checks["rowbal_parent_is_symlink"] is False
    assert checks["rowbal_run_manifest"]["sha256_matches"] is True
    assert all(item["sha256_matches"] for item in checks["rowbal_parent_files"].values())


def test_rowbal_actual_data_anchors_are_checked_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher._ROWBAL_SHARED,
        "_validate_data",
        lambda: (
            [],
            {
                "manifest_sha256": "manifest-drift",
                "content_sha256_actual": launcher.ROWBAL_DATASET_SHA256,
                "ordered_id_sha256_actual": launcher.ROWBAL_ORDERED_ID_SHA256,
            },
        ),
    )
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_rowbal_data(checks, blockers)
    assert "ROWBAL_DATA_ACTUAL_MANIFEST_SHA256_DRIFT" in blockers
    assert checks["rowbal_data_actual_anchors"]["manifest_sha256"]["matches"] is False


def test_sft_adapter_requires_final_three_epoch_state(tmp_path: Path) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors", "train_results.json", "all_results.json"):
        (output / name).write_text("{}\n", encoding="utf-8")
    (output / "trainer_state.json").write_text(
        json.dumps(
            {"global_step": 1875, "max_steps": 1875, "num_train_epochs": 3, "epoch": 1.0, "train_batch_size": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    assert launcher._adapter_complete(output) is False
    state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
    state["epoch"] = 3.0
    (output / "trainer_state.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    assert launcher._adapter_complete(output) is True


def test_train_requires_explicit_ep3_approval_but_other_actions_do_not(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(launcher, "_validate_rowbal_parent", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_validate_rowbal_data", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_validate_stage2_data", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_disk_gate", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_gpu0_compute_processes", lambda: [])
    monkeypatch.setattr(launcher, "_output_state", lambda run, action: (run.train_output, False, False))
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            run="exp4_4_3",
            action="train",
            gpus="0",
            rowbal_ep3_approved=False,
            execute=False,
        ),
    )
    launcher.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["executed"] is False
    assert "ROWBAL_EP3_APPROVAL_REQUIRED_FOR_TRAIN" in payload["blockers"]


@pytest.mark.parametrize("action", ("predict", "evaluate"))
def test_predict_and_evaluate_do_not_require_ep3_approval(
    action: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(launcher, "_validate_rowbal_parent", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_validate_rowbal_data", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_validate_stage2_data", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_disk_gate", lambda checks, blockers: None)
    monkeypatch.setattr(launcher, "_gpu0_compute_processes", lambda: [])
    monkeypatch.setattr(launcher, "_adapter_complete", lambda path: False)
    monkeypatch.setattr(launcher, "_prediction_complete", lambda path: False)
    monkeypatch.setattr(launcher, "_output_state", lambda run, action: (run.predict_output, False, False))
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            run="exp4_4_3",
            action=action,
            gpus="0",
            rowbal_ep3_approved=False,
            execute=False,
        ),
    )
    launcher.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert not any("APPROVAL" in item for item in payload["blockers"])


def test_commands_are_direct_and_evaluation_execute_is_explicit() -> None:
    run = launcher.RUNS["exp4_7_3"]
    train = launcher._command(run, "train")
    predict = launcher._command(run, "predict")
    evaluate = launcher._command(run, "evaluate")
    assert "qwen35_08b_bricknet_stage2_exp4_7_3" in " ".join(train + predict)
    assert str(run.train_config.relative_to(ROOT)) in train
    assert str(run.predict_config.relative_to(ROOT)) in predict
    assert "PT-exp2-v2" not in " ".join(train + predict + evaluate)
    assert "--execute" not in evaluate


def test_gpu0_query_is_local_and_targets_physical_gpu0(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **kwargs: object) -> Result:
        calls.append(command)
        if any("query-gpu=uuid" in item for item in command):
            return Result("GPU-0-UUID\n")
        return Result("123, 42, python\n")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher._gpu0_compute_processes() == ["123, 42, python"]
    assert calls[0][1:3] == ["--id=0", "--query-gpu=uuid"]
    assert calls[1][1] == "--id=GPU-0-UUID"
    assert not hasattr(launcher, "_gpu1_compute_processes")
