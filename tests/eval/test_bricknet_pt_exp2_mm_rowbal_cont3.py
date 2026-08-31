import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py"
SPEC = importlib.util.spec_from_file_location("launch_bricknet_pt_exp2_mm_rowbal_cont3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)

VALIDATOR_SCRIPT = ROOT / "scripts/validate_bricknet_pt_exp2_mm_rowbal_cont3.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_bricknet_pt_exp2_mm_rowbal_cont3", VALIDATOR_SCRIPT
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(sample_id: str, *, multimodal: bool) -> dict:
    return {
        "id": sample_id,
        "images": ["images/example.png"] if multimodal else [],
        "messages": [
            {"role": "system", "content": "You are a LEGO planner."},
            {
                "role": "user",
                "content": "<image>\nCaption:\n\nInventory of parts:" if multimodal else "Build it.",
            },
            {"role": "assistant", "content": "a Brick 1 x 1 | Red\n"},
        ],
    }


def _make_data_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """Install a two-row version of the immutable data contract in tmp_path."""

    data_root = tmp_path / "data"
    data_file = data_root / launcher.DATA_FILE_NAME
    registry_path = data_root / "dataset_info.json"
    manifest_path = data_root / "manifest.v1.json"
    loader_path = data_root / "loader_validation_report.json"
    processor_path = data_root / "processor_audit.json"
    processor_evidence_root = data_root / "processor_audit_full"
    rows = [_row("mm-0", multimodal=True), _row("text-0", multimodal=False)]
    _write_rows(data_file, rows)
    registry = launcher._registry_expected()
    _write_json(registry_path, registry)
    data_hash = _sha256(data_file)
    ordered_id_hash = hashlib.sha256(b"mm-0\ntext-0\n").hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "eligible": True,
        "dataset": launcher.DATASET_NAME,
        "dataset_file": launcher.DATA_FILE_NAME,
        "rows": 2,
        "multimodal_rows": 1,
        "text_rows": 1,
        "sha256": data_hash,
        "ordered_id_sha256": ordered_id_hash,
        "schema": {
            "top_level_fields": ["id", "images", "messages"],
            "top_level_key_signatures": {"id,images,messages": 2},
            "message_fields": ["role", "content"],
            "message_roles": ["system", "user", "assistant"],
            "message_role_sequences": {"system,user,assistant": 2},
            "row_order": "multimodal_prefix_then_text_suffix",
        },
        "arrow_materialization": {
            "eligible": True,
            "rows": 2,
            "columns": ["id", "images", "messages"],
        },
        "target_token_mass": {
            "required": True,
            "eligible": True,
            "multimodal": {"rows": 1, "total_tokens": 10},
            "selected_text": {"rows": 1, "total_tokens": 20},
            "text_to_multimodal_target_token_ratio": 2.0,
            "row_balance_ratio": 1.0,
        },
        "gates": {"target_token_audit": True},
        "registry": {"path": "dataset_info.json", "sha256": _sha256(registry_path)},
    }
    # Reports bind to the immutable manifest hash.  They are written after the
    # manifest and are intentionally not themselves listed by hash in it,
    # avoiding a circular hash dependency.
    _write_json(manifest_path, manifest)
    manifest_hash = _sha256(manifest_path)
    _write_json(
        loader_path,
        {
            "eligible": True,
            "raw_full_materialization": True,
            "model_weights_loaded": False,
            "optimizer_started": False,
            "trainer_started": False,
            "dataset": launcher.DATASET_NAME,
            "rows": 2,
            "columns": ["id", "images", "messages"],
            "preprocessed_rows": 4,
            "preprocessed_columns": ["input_ids", "attention_mask", "labels"],
            "arrow_materialization": {
                "eligible": True,
                "rows": 2,
                "columns": ["id", "images", "messages"],
            },
            "config_sha256": _sha256(launcher.TRAIN_CONFIG),
            "dataset_sha256": data_hash,
            "manifest_sha256": manifest_hash,
        },
    )
    _write_json(
        processor_path,
        {
            "eligible": True,
            "full_pool": True,
            "training_eligible": True,
            "dataset": launcher.DATASET_NAME,
            "rows": 2,
            "dataset_sha256": data_hash,
            "manifest_sha256": manifest_hash,
            "errors": 0,
            "truncated": 0,
            "zero_errors": True,
            "zero_truncation": True,
            "cutoff_len": launcher.EXPECTED_CUTOFF_LEN,
            "source_evidence_root": str(processor_evidence_root),
            "source_report": str(processor_evidence_root / "audit.json"),
            "source_report_sha256": "pending",
            "source_summary": {
                "count": 2,
                "errors": 0,
                "truncated": 0,
                "sidecar": str(processor_evidence_root / "rows.jsonl"),
                "sidecar_sha256": "pending",
            },
            "config": {"media_dir": str((launcher.ROOT / "data").resolve())},
            "runtime": {"media_dir": str((launcher.ROOT / "data").resolve())},
        },
    )
    source_sidecar = processor_evidence_root / "rows.jsonl"
    source_sidecar.parent.mkdir(parents=True, exist_ok=True)
    source_sidecar.write_text('{"ok": true}\n', encoding="utf-8")
    source_report = processor_evidence_root / "audit.json"
    _write_json(source_report, {"datasets": {launcher.DATASET_NAME: {"sidecar": str(source_sidecar)}}})
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    processor["source_report_sha256"] = _sha256(source_report)
    processor["source_summary"]["sidecar_sha256"] = _sha256(source_sidecar)
    _write_json(processor_path, processor)
    monkeypatch.setattr(launcher, "DATA_ROOT", data_root)
    monkeypatch.setattr(launcher, "DATA_MANIFEST", manifest_path)
    monkeypatch.setattr(launcher, "DATA_FILE", data_file)
    monkeypatch.setattr(launcher, "DATA_REGISTRY", registry_path)
    monkeypatch.setattr(launcher, "LOADER_REPORT", loader_path)
    monkeypatch.setattr(launcher, "PROCESSOR_AUDIT", processor_path)
    monkeypatch.setattr(launcher, "PROCESSOR_AUDIT_EVIDENCE", processor_evidence_root)
    monkeypatch.setattr(launcher, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(launcher, "EXPECTED_MM_ROWS", 1)
    monkeypatch.setattr(launcher, "EXPECTED_TEXT_ROWS", 1)
    monkeypatch.setattr(launcher, "EXPECTED_MM_TARGET_TOKENS", 10)
    monkeypatch.setattr(launcher, "EXPECTED_TEXT_TARGET_TOKENS", 20)
    monkeypatch.setattr(launcher, "EXPECTED_TEXT_TO_MM_TARGET_TOKEN_RATIO", 2.0)
    monkeypatch.setattr(launcher, "EXPECTED_ROW_BALANCE_RATIO", 1.0)
    return data_root, manifest


def test_train_yaml_uses_string_no_and_preserves_random_length_cache_contract() -> None:
    config = yaml.safe_load(launcher.TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert config["eval_strategy"] == "no"
    assert isinstance(config["eval_strategy"], str)
    assert config["do_eval"] is False
    assert config["train_sampling_strategy"] == "random"
    assert config["length_column_name"] == "length"
    assert config["dataloader_drop_last"] is False
    assert "save_safetensors" not in config
    blockers, checks = launcher._validate_train_config(launcher.TRAIN_CONFIG)
    assert blockers == []
    assert checks["expected_steps_per_epoch"] == 16_882
    assert checks["expected_max_steps"] == 50_646


def test_real_llamafactory_typed_parser_accepts_rowbal_yaml() -> None:
    """Exercise the repository parser, not only PyYAML/static launcher checks."""

    python = launcher.LLAMAFACTORY_PYTHON
    if not python.is_file():
        pytest.skip(f"typed parser environment is unavailable: {python}")
    parser_code = """
import json
import sys
import yaml
from llamafactory.hparams.parser import _parse_train_args
raw = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
# The real training YAML intentionally requests bf16 for CUDA 1.  Add the
# parser's explicit CPU switch only in this parser-only test so this check can
# never initialize CUDA or a Trainer.
raw["use_cpu"] = True
_, _, train, _, _ = _parse_train_args(raw)
print(json.dumps({
    "eval_strategy": train.eval_strategy.value,
    "do_eval": train.do_eval,
    "train_sampling_strategy": train.train_sampling_strategy,
    "length_column_name": train.length_column_name,
    "dataloader_drop_last": train.dataloader_drop_last,
    "per_device_train_batch_size": train.per_device_train_batch_size,
    "gradient_accumulation_steps": train.gradient_accumulation_steps,
    "num_train_epochs": train.num_train_epochs,
}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [str(python), "-c", parser_code, str(launcher.TRAIN_CONFIG)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    assert parsed == {
        "eval_strategy": "no",
        "do_eval": False,
        "train_sampling_strategy": "random",
        "length_column_name": "length",
        "dataloader_drop_last": False,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3.0,
    }


def test_predict_yamls_bind_distinct_epoch_checkpoints_and_outputs() -> None:
    for epoch, step in ((1, 16_882), (2, 33_764), (3, 50_646)):
        run = launcher.PREDICTION_RUNS[f"ep{epoch}"]
        config = yaml.safe_load(run.config.read_text(encoding="utf-8"))
        blockers, _checks = launcher._validate_predict_config(run)
        assert blockers == []
        assert config["adapter_name_or_path"].endswith(f"checkpoint-{step}")
        assert config["eval_dataset"] == "BrickNet-MM-PT-VAL"
        assert config["cutoff_len"] == 4096
        assert config["max_new_tokens"] == 4096
        assert config["do_sample"] is True
        assert config["temperature"] == 1.0
        assert config["top_k"] == 20
        assert config["top_p"] == 0.95
        assert config["seed"] == 42
    assert len({run.output_name for run in launcher.PREDICTION_RUNS.values()}) == 3


def test_data_gate_audits_single_file_registry_arrow_and_real_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_data_fixture(tmp_path, monkeypatch)
    blockers, checks = launcher._validate_data()
    assert blockers == []
    assert checks["actual_rows"] == 2
    assert checks["actual_multimodal_rows"] == 1
    assert checks["actual_text_rows"] == 1
    assert checks["schema"]["message_role_sequences"] == {"system,user,assistant": 2}
    assert checks["arrow_materialization"]["columns"] == ["id", "images", "messages"]
    assert checks["target_token_mass"]["eligible"] is True
    assert checks["target_token_mass"]["actual"]["multimodal"]["total_tokens"] == 10
    assert checks["target_token_mass"]["actual"]["selected_text"]["total_tokens"] == 20
    assert checks["loader_validation"]["eligible"] is True
    assert checks["processor_audit"]["eligible"] is True


def test_data_gate_rejects_message_role_sequence_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_data_fixture(tmp_path, monkeypatch)
    manifest = json.loads(launcher.DATA_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema"]["message_role_sequences"] = {"user,assistant": 2}
    launcher.DATA_MANIFEST.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    blockers, _checks = launcher._validate_data()
    assert "ROWBAL_DATA_SCHEMA_DECLARATION_DRIFT" in blockers


def test_data_gate_rejects_target_token_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_data_fixture(tmp_path, monkeypatch)
    manifest = json.loads(launcher.DATA_MANIFEST.read_text(encoding="utf-8"))
    manifest["target_token_mass"]["selected_text"]["total_tokens"] = 21
    launcher.DATA_MANIFEST.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    blockers, checks = launcher._validate_data()
    assert "ROWBAL_TARGET_TOKEN_MASS_GATE_FAILED" in blockers
    assert checks["target_token_mass"]["eligible"] is False


def test_data_gate_rejects_loader_config_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_data_fixture(tmp_path, monkeypatch)
    loader = json.loads(launcher.LOADER_REPORT.read_text(encoding="utf-8"))
    loader["config_sha256"] = "0" * 64
    launcher.LOADER_REPORT.write_text(json.dumps(loader) + "\n", encoding="utf-8")
    blockers, _checks = launcher._validate_data()
    assert "ROWBAL_LOADER_VALIDATION_GATE_FAILED" in blockers


def test_data_gate_rejects_processor_media_root_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_data_fixture(tmp_path, monkeypatch)
    processor = json.loads(launcher.PROCESSOR_AUDIT.read_text(encoding="utf-8"))
    processor["config"]["media_dir"] = str(tmp_path / "wrong-media-root")
    launcher.PROCESSOR_AUDIT.write_text(json.dumps(processor) + "\n", encoding="utf-8")
    blockers, checks = launcher._validate_data()
    assert "ROWBAL_PROCESSOR_AUDIT_GATE_FAILED" in blockers
    assert checks["processor_audit"]["processor_media_root_ok"] is False


def test_cache_gate_binds_rows_columns_manifest_and_keeps_random_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, _manifest = _make_data_fixture(tmp_path, monkeypatch)
    config_path = tmp_path / "rowbal.yaml"
    cache = tmp_path / "tokenized"
    config = {
        "tokenized_path": str(cache),
        "length_column_name": "length",
        "train_sampling_strategy": "random",
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from datasets import Dataset, DatasetDict

    train_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3]],
            "attention_mask": [[1, 1], [1]],
            "labels": [[1, 2], [3]],
            "images": [[], []],
            "length": [2, 1],
        }
    )
    DatasetDict({"train": train_dataset}).save_to_disk(str(cache))
    length_manifest = {
        "action": "build_tokenized_cache_with_length",
        "length_column": "length",
        "input_column": "input_ids",
        "output_cache": str(cache),
        "stats": {"train": {"rows": 2}},
    }
    _write_json(cache / "length_cache_manifest.json", length_manifest)
    data_hash = _sha256(data_root / launcher.DATA_FILE_NAME)
    ordered_hash = hashlib.sha256(b"mm-0\ntext-0\n").hexdigest()
    manifest_hash = _sha256(launcher.DATA_MANIFEST)
    _write_json(
        cache / launcher.CACHE_BINDING_NAME,
        {
            "schema_version": 1,
            "dataset": launcher.DATASET_NAME,
            "config_sha256": _sha256(config_path),
            "dataset_sha256": data_hash,
            "ordered_id_sha256": ordered_hash,
            "rows": 2,
            "length_column": "length",
            "sampling_strategy": "random",
            "source": {
                "manifest_sha256": manifest_hash,
                "dataset_sha256": data_hash,
                "ordered_id_sha256": ordered_hash,
                "rows": 2,
            },
        },
    )
    result = launcher._cache_check(config_path)
    assert result["eligible"] is True
    assert result["columns"] == ["attention_mask", "images", "input_ids", "labels", "length"]
    assert result["actual_train_rows"] == 2
    assert result["actual_train_columns"] == ["input_ids", "attention_mask", "labels", "images", "length"]
    assert result["length_mismatch_count"] == 0
    assert result["length_validation"]["mismatch"] == 0
    config["train_sampling_strategy"] = "group_by_length"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    drift = launcher._cache_check(config_path)
    assert drift["error"] == "SAMPLING_STRATEGY_MUST_REMAIN_RANDOM"


def test_cache_gate_rejects_actual_length_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "EXPECTED_ROWS", 2)
    config_path = tmp_path / "rowbal.yaml"
    cache = tmp_path / "tokenized"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tokenized_path": str(cache),
                "length_column_name": "length",
                "train_sampling_strategy": "random",
            }
        ),
        encoding="utf-8",
    )
    from datasets import Dataset, DatasetDict

    train_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3]],
            "attention_mask": [[1, 1], [1]],
            "labels": [[1, 2], [3]],
            "images": [[], []],
            "length": [3, 1],
        }
    )
    DatasetDict({"train": train_dataset}).save_to_disk(str(cache))
    _write_json(
        cache / "length_cache_manifest.json",
        {
            "length_column": "length",
            "input_column": "input_ids",
            "output_cache": str(cache),
            "stats": {"train": {"rows": 2}},
        },
    )
    result = launcher._cache_check(config_path, require_binding=False)
    assert result["error"] == "CACHE_ACTUAL_LENGTH_MISMATCH"
    assert result["actual_train_rows"] == 2
    assert result["length_mismatch_count"] == 1


def test_checkpoint_gate_requires_full_optimizer_scheduler_rng_and_typed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "train-output"
    checkpoint = output / "checkpoint-16882"
    checkpoint.mkdir(parents=True)
    _write_json(
        checkpoint / "adapter_config.json",
        {
            "peft_type": "LORA",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.0,
            "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        },
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        (checkpoint / name).write_bytes(b"state")
    _write_json(
        checkpoint / "trainer_state.json",
        {
            "epoch": 1.0,
            "global_step": 16_882,
            "max_steps": 50_646,
            "num_train_epochs": 3,
            "train_batch_size": 2,
        },
    )
    typed = {
        "output_dir": str(output),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "save_only_model": False,
        "train_sampling_strategy": "random",
        "length_column_name": "length",
        "eval_strategy": "no",
        "do_eval": False,
        "seed": 42,
        "dataloader_drop_last": False,
        "n_gpu": 1,
        "world_size": 1,
    }
    monkeypatch.setattr(launcher, "_read_training_args", lambda _path: (typed, None))
    blockers, checks = launcher._checkpoint_gate(1, output=output)
    assert blockers == []
    assert checks["eligible"] is True
    (checkpoint / "optimizer.pt").unlink()
    blockers, _checks = launcher._checkpoint_gate(1, output=output)
    assert any("optimizer" in blocker for blocker in blockers)


def test_resume_command_is_explicit_and_points_at_validated_epoch_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "TRAIN_OUTPUT", tmp_path / "train-output")
    command = launcher._train_command(argparse.Namespace(resume_epoch=2))
    assert command[-1] == f"resume_from_checkpoint={tmp_path / 'train-output' / 'checkpoint-33764'}"


def test_successful_resume_marks_run_manifest_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_writes: list[dict] = []
    captured_commands: list[list[str]] = []
    data_checks = {
        "manifest_sha256": "manifest",
        "content_sha256_actual": "dataset",
        "ordered_id_sha256_actual": "ordered",
    }
    monkeypatch.setattr(launcher, "_check_train", lambda _args: ([], {"data": data_checks}))
    monkeypatch.setattr(launcher, "_train_command", lambda _args: ["fake-train"])
    monkeypatch.setattr(launcher, "_run_provenance_path", lambda: tmp_path / "run_manifest.json")
    monkeypatch.setattr(launcher, "_single_gpu_env", lambda: {})
    monkeypatch.setattr(launcher, "_posttrain_gate", lambda: ([], {"eligible": True}))
    monkeypatch.setattr(
        launcher,
        "_write_json",
        lambda _path, payload: captured_writes.append(payload),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **_kwargs: captured_commands.append(command),
    )
    launcher._run_train(argparse.Namespace(execute=True, resume_epoch=1))
    assert captured_commands == [["fake-train"]]
    assert [payload["status"] for payload in captured_writes] == ["complete"]


def test_gpu_gate_requires_exactly_physical_cuda_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(launcher, "_selected_gpu_processes", lambda: [])
    blockers, checks = launcher._gpu_gate()
    assert blockers == []
    assert checks["selectors"] == ["1"]
    monkeypatch.setattr(launcher, "_selected_gpu_processes", lambda: ["123, 100, python"])
    blockers, checks = launcher._gpu_gate()
    assert "CUDA1_COMPUTE_PROCESSES_ACTIVE" in blockers
    assert checks["gpu_processes"] == ["123, 100, python"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    blockers, _checks = launcher._gpu_gate()
    assert "ROWBAL_REQUIRES_EXACTLY_CUDA1" in blockers


def test_validator_dry_run_does_not_create_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, _manifest = _make_data_fixture(tmp_path, monkeypatch)
    for path in (data_root / "loader_validation_report.json", data_root / "processor_audit.json"):
        path.unlink()
    shutil.rmtree(data_root / "processor_audit_full")
    config = tmp_path / "rowbal.yaml"
    config.write_text("model_name_or_path: local\n", encoding="utf-8")
    audit_script = tmp_path / "audit.py"
    audit_script.write_text("# mocked audit\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    loader_output = data_root / "loader_validation.json"
    processor_output = data_root / "processor_audit.json"
    evidence_root = data_root / "processor_audit_full"
    for name, value in (
        ("DEFAULT_DATA_ROOT", data_root),
        ("DEFAULT_DATA_FILE", data_root / launcher.DATA_FILE_NAME),
        ("DEFAULT_MANIFEST", launcher.DATA_MANIFEST),
        ("DEFAULT_REGISTRY", data_root / "dataset_info.json"),
        ("DEFAULT_LOADER_REPORT", loader_output),
        ("DEFAULT_PROCESSOR_REPORT", processor_output),
        ("DEFAULT_PROCESSOR_EVIDENCE_ROOT", evidence_root),
        ("DEFAULT_CONFIG", config),
        ("DEFAULT_AUDIT", audit_script),
        ("DEFAULT_LLAMAFACTORY_PYTHON", python),
    ):
        monkeypatch.setattr(validator, name, value)
    monkeypatch.setattr(validator, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(validator, "EXPECTED_MM_ROWS", 1)
    monkeypatch.setattr(validator, "EXPECTED_TEXT_ROWS", 1)
    monkeypatch.setattr(sys, "argv", [str(VALIDATOR_SCRIPT)])
    validator.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["executed"] is False
    assert not loader_output.exists()
    assert not processor_output.exists()
    assert not evidence_root.exists()


def test_validator_loader_smoke_uses_small_prefix_after_full_raw_arrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_name_or_path: local\n", encoding="utf-8")
    captured: dict = {}

    class FakeDataset:
        column_names = ["input_ids", "attention_mask", "labels"]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict:
            del index
            return {"input_ids": [1], "attention_mask": [1], "labels": [1]}

    class FakeTrainingArgs:
        remove_unused_columns = True

    class FakeFinetuningArgs:
        stage = "sft"

    def fake_parse(raw: dict) -> tuple:
        captured.update(raw)
        return object(), object(), FakeTrainingArgs(), FakeFinetuningArgs(), object()

    data_module = types.ModuleType("llamafactory.data")
    data_module.get_dataset = lambda *args, **kwargs: {"train_dataset": FakeDataset()}
    data_module.get_template_and_fix_tokenizer = lambda *args, **kwargs: object()
    parser_module = types.ModuleType("llamafactory.hparams.parser")
    parser_module._parse_train_args = fake_parse
    hparams_module = types.ModuleType("llamafactory.hparams")
    hparams_module.parser = parser_module
    model_module = types.ModuleType("llamafactory.model")
    model_module.load_tokenizer = lambda _model_args: {"tokenizer": object(), "processor": object()}
    llamafactory_module = types.ModuleType("llamafactory")
    llamafactory_module.data = data_module
    llamafactory_module.hparams = hparams_module
    llamafactory_module.model = model_module
    for name, module in (
        ("llamafactory", llamafactory_module),
        ("llamafactory.data", data_module),
        ("llamafactory.hparams", hparams_module),
        ("llamafactory.hparams.parser", parser_module),
        ("llamafactory.model", model_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(validator, "_load_yaml", lambda _path: {"model_name_or_path": "local"})
    report = validator._loader_audit(
        config_path,
        {"data": {"sha256": "data"}, "manifest_sha256": "manifest"},
        max_samples=4,
    )
    assert captured["max_samples"] == 4
    assert captured["tokenized_path"] is None
    assert report["raw_full_materialization"] is True
    assert report["rows"] == validator.EXPECTED_ROWS
    assert report["preprocessed_rows"] == 4


def test_validator_preserves_full_processor_report_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"id": "x"}\n', encoding="utf-8")
    data_hash = _sha256(data_file)
    temporary_root = tmp_path / "run"
    temporary_root.mkdir()
    evidence_root = tmp_path / "data" / "processor_audit_full"
    evidence_root.parent.mkdir(parents=True)
    config = {
        "model_name_or_path": "local",
        "template": "qwen3_5_nothink",
        "cutoff_len": 6400,
        "media_dir": "data",
    }
    data_checks = {"data": {"sha256": data_hash}, "manifest_sha256": "manifest-hash"}
    monkeypatch.setattr(validator, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(validator, "EXPECTED_MM_ROWS", 1)
    monkeypatch.setattr(validator, "EXPECTED_TEXT_ROWS", 1)
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        del kwargs
        captured_command.extend(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        sidecar = output_dir / f"{validator.DATASET_NAME}_token_audit.jsonl"
        sidecar.write_text('{"ok": true}\n', encoding="utf-8")
        source_report = {
            "is_full_pool": True,
            "training_eligible": True,
            "zero_errors": True,
            "zero_truncation": True,
            "datasets": {
                validator.DATASET_NAME: {
                    "path": str(data_file.resolve()),
                    "sha256": data_hash,
                    "count": 2,
                    "errors": 0,
                    "truncated": 0,
                    "sidecar": str(sidecar),
                    "sidecar_sha256": _sha256(sidecar),
                }
            },
            "config": {"cutoff_len": 6400, "media_dir": str((validator.ROOT / "data").resolve())},
        }
        (output_dir / "BrickNet-MM-Reasoning_token_audit_report.json").write_text(
            json.dumps(source_report) + "\n", encoding="utf-8"
        )

    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    normalized = validator._processor_audit(
        data_file,
        data_checks,
        config,
        Path("/tmp/fake-python"),
        Path("/tmp/fake-audit.py"),
        temporary_root,
        evidence_root,
        4,
        8,
    )
    source_report = Path(normalized["source_report"])
    source_sidecar = Path(normalized["source_summary"]["sidecar"])
    assert source_report.is_file()
    assert source_sidecar.is_file()
    assert normalized["source_report_sha256"] == _sha256(source_report)
    assert normalized["source_summary"]["sidecar_sha256"] == _sha256(source_sidecar)
    assert normalized["processor_workers"] == 4
    assert normalized["processor_chunksize"] == 8
    assert normalized["config"]["media_dir"] == str((validator.ROOT / "data").resolve())
    media_root = Path(captured_command[captured_command.index("--bricknet-root") + 1])
    assert media_root == (validator.ROOT / "data").resolve()
    persisted = json.loads(source_report.read_text(encoding="utf-8"))
    assert Path(persisted["datasets"][validator.DATASET_NAME]["sidecar"]).is_file()
