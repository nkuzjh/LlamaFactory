import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_pt_exp2_text250k_downstream.py"
SPEC = importlib.util.spec_from_file_location("launch_bricknet_pt_exp2_text250k_downstream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def test_runs_and_yaml_namespaces_are_text250k_only() -> None:
    expected = {
        "exp4_4_2": "nonthinking-control",
        "exp4_7_2": "thinking-hard-v2-lean-state",
    }
    for run_name, variant in expected.items():
        run = launcher.RUNS[run_name]
        assert run.variant == variant
        assert run.train_output.name.startswith(f"train_{run_name}_")
        assert run.predict_output.name.startswith(f"eval_{run_name}_")
        assert "text250k" in run.train_output.name
        assert "text250k" in run.predict_output.name
        assert "text250k" in run.cache_path.name
        for path in (run.train_config, run.predict_config):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            binding = config["adapter_name_or_path"]
            assert binding.startswith(str(launcher.TEXT250K_ADAPTER.relative_to(ROOT)))
            assert "PT-exp2-v2" not in binding
            assert "PT_exp2_mm" not in binding
            assert "mm_e1" not in binding.lower()
            assert "mm_e2" not in binding.lower()
            assert "mm_e3" not in binding.lower()


def test_frozen_text250k_hash_contract_is_present() -> None:
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_text250k_adapter(checks, blockers)
    assert blockers == []
    assert checks["text250k_steps_match"] is True
    assert all(item["sha256_matches"] for item in checks["text250k_adapter_files"].values())


def test_any_frozen_text250k_hash_drift_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / launcher.TEXT250K_ADAPTER.name
    adapter.mkdir()
    for name in launcher.EXPECTED_TEXT250K_ADAPTER_HASHES:
        (adapter / name).write_text("drift\n", encoding="utf-8")
    (adapter / "trainer_state.json").write_text(
        json.dumps({"global_step": 250000, "max_steps": 250000}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "TEXT250K_ADAPTER", adapter)
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_text250k_adapter(checks, blockers)
    assert any(item.endswith("HASH_DRIFT") for item in blockers)


def test_text250k_adapter_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symlink = tmp_path / launcher.TEXT250K_ADAPTER.name
    symlink.symlink_to(launcher.TEXT250K_ADAPTER, target_is_directory=True)
    monkeypatch.setattr(launcher, "TEXT250K_ADAPTER", symlink)
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_text250k_adapter(checks, blockers)
    assert checks["text250k_adapter_is_symlink"] is True
    assert "TEXT250K_ADAPTER_ALIAS_SYMLINK_FORBIDDEN" in blockers


@pytest.mark.parametrize(
    "value",
    (
        "saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2",
        "saves/Qwen3.5-0.8B-Thinking/lora/PT_exp2_v2",
        "saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2,saves/downstream",
        "saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e1_x",
        "saves/Qwen3.5-0.8B-Thinking/lora/train-pt-exp2-mm-e2-v2",
        "saves/Qwen3.5-0.8B-Thinking/lora/mm-e3",
    ),
)
def test_forbidden_alias_and_mm_spellings_fail_closed(value: str) -> None:
    assert launcher._forbidden_binding_tokens(value)


def test_direct_text250k_binding_is_not_forbidden() -> None:
    assert not launcher._forbidden_binding_tokens(
        "saves/Qwen3.5-0.8B-Thinking/lora/"
        "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_"
        "lora64_len6401_nopack"
    )


def test_config_alias_or_mm_binding_has_dedicated_blocker(tmp_path: Path) -> None:
    run = launcher.RUNS["exp4_4_2"]
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "adapter_name_or_path: saves/Qwen3.5-0.8B-Thinking/lora/PT-exp2-v2,"
        "saves/Qwen3.5-0.8B-Thinking/lora/train_PT_exp2_mm_e1\n",
        encoding="utf-8",
    )
    bad_run = replace(run, train_config=bad_config)
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_config(bad_run, "train", checks, blockers)
    assert "FORBIDDEN_PT_EXP2_ALIAS_OR_MM_ADAPTER" in blockers


@pytest.mark.parametrize("run_name", ("exp4_4_2", "exp4_7_2"))
@pytest.mark.parametrize("action", ("train", "predict"))
def test_yaml_exact_fields_are_accepted(run_name: str, action: str) -> None:
    run = launcher.RUNS[run_name]
    checks: dict = {}
    blockers: list[str] = []
    launcher._validate_config(run, action, checks, blockers)
    assert blockers == []
    assert checks["config_fields"]["mismatches"] == {}
    config = yaml.safe_load(Path(checks["config"]).read_text(encoding="utf-8"))
    assert config["seed"] == 42


def test_command_contract_is_single_environment_and_direct_config() -> None:
    run = launcher.RUNS["exp4_4_2"]
    train = launcher._command(run, "train")
    predict = launcher._command(run, "predict")
    evaluate = launcher._command(run, "evaluate", execute=True)
    assert train[:6] == ["conda", "run", "-n", "llamafactory", "--no-capture-output", "llamafactory-cli"]
    assert "qwen35_08b_bricknet_stage2_exp4_4_2" in " ".join(train)
    assert "PT-exp2-v2" not in " ".join(train + predict + evaluate)
    assert "mm_e1" not in " ".join(train + predict + evaluate).lower()
    assert evaluate[-1] == "--execute"


def test_complete_output_is_a_safe_noop_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = launcher.RUNS["exp4_4_2"]
    monkeypatch.setattr(launcher, "_gpu1_compute_processes", lambda: [])
    monkeypatch.setattr(launcher, "_adapter_complete", lambda *_: True)
    monkeypatch.setattr(launcher, "_prediction_complete", lambda *_: True)
    monkeypatch.setattr(launcher, "_output_state", lambda *_: (run.predict_output, True, True))
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            run="exp4_4_2",
            action="evaluate",
            gpus="1",
            text250k_approved=False,
            execute=True,
        ),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("complete evaluate must be a safe no-op"),
    )
    launcher.main()
    payload = json.loads(capsys.readouterr().out.split("\nAlready complete", 1)[0])
    assert payload["already_complete"] is True
    assert payload["checks"]["output_action_safe_noop"] is True


def test_existing_incomplete_output_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = launcher.RUNS["exp4_4_2"]
    partial = tmp_path / "partial-output"
    partial.mkdir()
    bad_run = replace(run, train_output=partial)
    monkeypatch.setattr(launcher, "RUNS", {"exp4_4_2": bad_run})
    monkeypatch.setattr(launcher, "_validate_config", lambda *args: None)
    monkeypatch.setattr(launcher, "_gpu1_compute_processes", lambda: [])
    monkeypatch.setattr(
        launcher,
        "parse_args",
        lambda: argparse.Namespace(
            run="exp4_4_2",
            action="train",
            gpus="1",
            text250k_approved=True,
            execute=False,
        ),
    )
    launcher.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["already_complete"] is False
    assert "OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED" in payload["blockers"]


def test_evaluation_requires_alignment_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = launcher.RUNS["exp4_4_2"]
    prediction = tmp_path / "prediction"
    output_root = tmp_path / "outputs"
    prediction.mkdir()
    output_root.mkdir()
    evaluation_run = replace(run, predict_output=prediction)
    monkeypatch.setattr(launcher, "EVALUATION_ROOT", output_root)
    output = launcher._evaluation_path(evaluation_run)
    output.mkdir()
    (output / "evaluation_manifest.json").write_text(
        json.dumps({"status": "complete", "samples": 512}) + "\n", encoding="utf-8"
    )
    assert not (output / "alignment_manifest.json").exists()
    assert launcher._evaluation_complete(evaluation_run) is False
