import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_pt_exp2_unconditional.py"
SPEC = importlib.util.spec_from_file_location("launch_bricknet_pt_exp2_unconditional", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_complete_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    adapter = tmp_path / "adapter"
    _write_json(
        adapter / "adapter_config.json",
        {
            "base_model_name_or_path": launcher.BASE_MODEL,
            "r": 64,
            "lora_alpha": 128,
            "peft_type": "LORA",
            "target_modules": sorted(launcher.EXPECTED_TARGET_MODULES),
        },
    )
    _write_json(
        adapter / "trainer_state.json",
        {"global_step": launcher.EXPECTED_TRAIN_STEPS, "max_steps": launcher.EXPECTED_TRAIN_STEPS},
    )
    _write_json(adapter / "train_results.json", {"status": "finished"})
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    monkeypatch.setattr(
        launcher,
        "EXPECTED_ADAPTER_HASHES",
        {name: _sha256(adapter / name) for name in launcher.EXPECTED_ADAPTER_HASHES},
    )

    snapshot = tmp_path / "snapshot" / launcher.BASE_REVISION
    snapshot.mkdir(parents=True)
    for name in launcher.REQUIRED_SNAPSHOT_FILES:
        (snapshot / name).write_bytes(b"snapshot")
    _write_json(
        snapshot / "tokenizer.json",
        {
            "model": {"vocab": {"a": 64, "Ċ": 198}},
            "added_tokens": [{"id": 248046, "content": "<|im_end|>"}],
        },
    )
    _write_json(
        snapshot / "tokenizer_config.json",
        {
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {
                "248046": {"content": "<|im_end|>"}
            },
        },
    )
    (snapshot / "model.safetensors-00001-of-00001.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(launcher, "BASE_SNAPSHOT", snapshot)
    monkeypatch.setattr(
        launcher,
        "EXPECTED_SNAPSHOT_HASHES",
        {name: _sha256(snapshot / name) for name in launcher.EXPECTED_SNAPSHOT_HASHES},
    )

    existing_file = Path(__file__)
    monkeypatch.setattr(launcher, "LLAMAFACTORY_PYTHON", Path(sys.executable))
    monkeypatch.setattr(launcher, "BRICKNET_PYTHON", Path(sys.executable))
    monkeypatch.setattr(launcher, "GENERATE_SCRIPT", existing_file)
    monkeypatch.setattr(launcher, "EVALUATE_SCRIPT", existing_file)
    monkeypatch.setattr(launcher, "SCORE_SCRIPT", existing_file)
    monkeypatch.setattr(launcher, "TRAIN_CONFIG", existing_file)

    data_root = tmp_path / "bricknet-data"
    inset = data_root / "inset"
    inset.mkdir(parents=True)
    (inset / "a.ply").write_bytes(b"a")
    (inset / "b.ply").write_bytes(b"b")
    monkeypatch.setattr(launcher, "BRICKNET_DATA", data_root)
    monkeypatch.setattr(launcher, "EXPECTED_COLLISION_MESHES", 2)
    catalog = tmp_path / "catalog-v1"
    catalog.mkdir()
    for name in launcher.CATALOG_FILES:
        (catalog / name).write_bytes(b"catalog")
    monkeypatch.setattr(launcher, "CATALOG_ROOT", catalog)

    monkeypatch.setattr(launcher, "OUTPUT_ROOT", tmp_path / "formal-output")
    return adapter


def test_generation_command_freezes_main_protocol_and_omits_seed_flags() -> None:
    command = launcher._generation_command(
        Path("/reviewed/adapter"),
        Path("/staging/out.jsonl"),
        samples=launcher.EXPECTED_SAMPLES,
        batch_size=launcher.EXPECTED_BATCH_SIZE,
    )
    assert command[command.index("--model") + 1] == str(launcher.BASE_SNAPSHOT)
    assert command[command.index("--prompt") + 1] == "a"
    assert command[command.index("--num_samples") + 1] == "2048"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--max_new_tokens") + 1] == "4096"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--top_k") + 1] == "20"
    assert command[command.index("--top_p") + 1] == "0.95"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--stop_after_newlines") + 1] == "199"
    assert "--seed" not in command
    assert "--prompts_file" not in command


def test_evaluation_command_uses_the_evaluator_contract() -> None:
    command = launcher._evaluation_command(
        Path("/staging/out.jsonl"),
        Path("/staging/evaluation"),
        expected_samples=launcher.EXPECTED_SAMPLES,
    )
    assert command[command.index("--input") + 1] == "/staging/out.jsonl"
    assert command[command.index("--output-dir") + 1] == "/staging/evaluation"
    assert command[command.index("--expected-samples") + 1] == "2048"
    assert command[-1] == "--execute"


def test_parser_accepts_documented_gpus_alias_and_gpu_spelling() -> None:
    assert launcher.parse_args(["--action", "preflight", "--gpus", "0"]).gpu == "0"
    assert launcher.parse_args(["--action", "preflight", "--gpu", "0"]).gpu == "0"


@pytest.mark.parametrize("selector", [None, "", " ", "0,1", "-1", "gpu0", "01", "0 "])
def test_gpu_gate_accepts_only_one_decimal_physical_selector(selector: str | None) -> None:
    assert not launcher._is_valid_gpu_selector(selector)


@pytest.mark.parametrize("selector", ["0", "1", "12"])
def test_gpu_gate_accepts_canonical_decimal_physical_selector(selector: str) -> None:
    assert launcher._is_valid_gpu_selector(selector)


def test_command_env_pins_catalog_over_parent_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRICKNET_CATALOG", "wrong-catalog")
    env = launcher._command_env("0")
    assert env["BRICKNET_CATALOG"] == launcher.EXPECTED_CATALOG
    assert launcher._environment("0")["bricknet_catalog"] == launcher.EXPECTED_CATALOG


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", True),
        ("id", 0.0),
        ("sample", False),
        ("sample", "0"),
        ("path", "/wrong/input"),
        ("source", "val-caption-source"),
        ("caption", "val-caption"),
    ],
)
def test_raw_rows_match_unconditional_evaluator_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    row: dict[str, object] = {
        "id": 0,
        "sample": 0,
        "source": "",
        "caption": "",
        "text": "a\n",
    }
    row[field] = value
    raw = tmp_path / "out.jsonl"
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(launcher.GateError):
        launcher._validate_raw_rows(raw, 1)


def test_raw_rows_accept_empty_unconditional_metadata(tmp_path: Path) -> None:
    raw = tmp_path / "out.jsonl"
    raw.write_text(
        json.dumps(
            {"id": 0, "sample": 0, "source": "", "caption": "", "text": "a\n"}
        )
        + "\n",
        encoding="utf-8",
    )
    assert launcher._validate_raw_rows(raw, 1)[0]["id"] == 0


def test_run_spec_raw_identity_rejects_row_or_hash_tamper(tmp_path: Path) -> None:
    raw = tmp_path / "out.jsonl"
    raw.write_text(
        json.dumps(
            {"id": 0, "sample": 0, "source": "", "caption": "", "text": "a\n"}
        )
        + "\n",
        encoding="utf-8",
    )
    rows = launcher._read_jsonl(raw)
    spec = {"raw_rows": 1, "raw_sha256": _sha256(raw)}
    launcher._validate_run_spec_raw_artifact(spec, raw, rows)
    with pytest.raises(launcher.GateError, match="raw_rows"):
        launcher._validate_run_spec_raw_artifact(
            {**spec, "raw_rows": 2}, raw, rows
        )
    with pytest.raises(launcher.GateError, match="raw_sha256"):
        launcher._validate_run_spec_raw_artifact(
            {**spec, "raw_sha256": "0" * 64}, raw, rows
        )


def test_run_spec_evaluation_identity_rejects_artifact_tamper(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    for name in ("scored.jsonl", "metrics.json", "evaluation_manifest.json"):
        (root / name).write_text(name + "\n", encoding="utf-8")
    scored_rows = [{"id": 0}]
    spec = {
        "scored_sha256": _sha256(root / "scored.jsonl"),
        "metrics_sha256": _sha256(root / "metrics.json"),
        "evaluation_manifest_sha256": _sha256(root / "evaluation_manifest.json"),
        "scored_rows": 1,
    }
    launcher._validate_run_spec_evaluation_artifacts(spec, root, scored_rows)
    for field in (
        "scored_sha256",
        "metrics_sha256",
        "evaluation_manifest_sha256",
        "scored_rows",
    ):
        tampered = {**spec, field: 2 if field == "scored_rows" else "0" * 64}
        with pytest.raises(launcher.GateError, match=field):
            launcher._validate_run_spec_evaluation_artifacts(
                tampered, root, scored_rows
            )


def test_complete_static_preflight_is_read_only_and_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    report = launcher.preflight("generate", adapter, "0")
    assert report.eligible
    assert report.blockers == []
    assert report.checks["adapter"]["trainer_state"]["global_step"] == launcher.EXPECTED_TRAIN_STEPS
    assert report.checks["tokenizer_semantics"]["eligible"]


def test_preflight_catches_tokenizer_prompt_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    tokenizer_path = launcher.BASE_SNAPSHOT / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["a"] = 65
    _write_json(tokenizer_path, tokenizer)
    report = launcher.preflight("generate", adapter, "0")
    assert "TOKENIZER_PROMPT_TOKEN_IDS_MISMATCH" in report.blockers


def test_preflight_catches_tokenizer_newline_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    tokenizer_path = launcher.BASE_SNAPSHOT / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["model"]["vocab"]["Ċ"] = 199
    _write_json(tokenizer_path, tokenizer)
    report = launcher.preflight("generate", adapter, "0")
    assert "TOKENIZER_NEWLINE_TOKEN_ID_MISMATCH" in report.blockers


def test_preflight_catches_tokenizer_eos_added_token_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    tokenizer_path = launcher.BASE_SNAPSHOT / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"][0]["id"] = 248045
    _write_json(tokenizer_path, tokenizer)
    report = launcher.preflight("generate", adapter, "0")
    assert "TOKENIZER_EOS_ADDED_TOKEN_MISMATCH" in report.blockers


def test_preflight_catches_tokenizer_eos_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    config_path = launcher.BASE_SNAPSHOT / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["added_tokens_decoder"]["248046"]["content"] = "<|im_end_drift|>"
    _write_json(config_path, config)
    report = launcher.preflight("generate", adapter, "0")
    assert "TOKENIZER_EOS_CONFIG_MISMATCH" in report.blockers


def test_preflight_catches_missing_yaml_target_even_with_fixture_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = sorted(launcher.EXPECTED_TARGET_MODULES - {"down_proj"})
    _write_json(adapter / "adapter_config.json", config)
    report = launcher.preflight("generate", adapter, "0")
    assert "ADAPTER_CONFIG_CONTRACT_FAILED" in report.blockers


def test_preflight_catches_wrong_training_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    _write_json(adapter / "trainer_state.json", {"global_step": 249999, "max_steps": 250000})
    report = launcher.preflight("generate", adapter, "0")
    assert "TRAINER_STATE_STEPS_CONTRACT_FAILED" in report.blockers


def test_preflight_catches_base_snapshot_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    snapshot_file = launcher.BASE_SNAPSHOT / "config.json"
    snapshot_file.write_bytes(b"drift")
    report = launcher.preflight("generate", adapter, "0")
    assert "BASE_SNAPSHOT_HASH_MISMATCH:config.json" in report.blockers


def test_verify_recomputes_primary_scored_metrics() -> None:
    scored = [
        {"n_actions": 4, "invalid": None, "collisions": []},
        {"n_actions": 5, "invalid": 2, "collisions": [3]},
    ]
    assert launcher._aggregate_scored_metrics(scored) == {
        "samples": 2,
        "parsable_count": 1,
        "parsable_rate": 0.5,
        "clean_count": 1,
        "clean_rate": 0.5,
        "mean_actions_before_first_failure": 2.0,
    }


def test_verify_rejects_run_spec_protocol_drift(tmp_path: Path) -> None:
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "experiment_id": "PT-exp2-text8m-250k-unconditional-v1",
        "scope": "formal",
        "protocol": {"num_samples": 1},
        "adapter": {"path": str(tmp_path / "adapter")},
    }
    with pytest.raises(launcher.GateError, match="frozen main protocol"):
        launcher._validate_run_spec_contract(spec, tmp_path / "adapter")


def test_verify_rejects_run_spec_code_identity_drift(tmp_path: Path) -> None:
    spec = {
        "schema_version": launcher.SCHEMA_VERSION,
        "experiment_id": "PT-exp2-text8m-250k-unconditional-v1",
        "scope": "formal",
        "protocol": launcher._protocol(
            launcher.EXPECTED_SAMPLES, launcher.EXPECTED_BATCH_SIZE
        ),
        "tokenizer": launcher._tokenizer_protocol(),
        "code": {},
        "adapter": {"path": str(tmp_path / "adapter")},
    }
    with pytest.raises(launcher.GateError, match="code identity"):
        launcher._validate_run_spec_contract(spec, tmp_path / "adapter")


def test_verify_rejects_unexpected_building_entry_before_promotion(tmp_path: Path) -> None:
    building = tmp_path / "run.building"
    building.mkdir()
    for name in launcher.FINAL_REQUIRED_OUTPUTS:
        (building / name).write_text("artifact\n", encoding="utf-8")
    (building / "out.jsonl.shard0").write_text("partial\n", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="entry set mismatch"):
        launcher._check_building_entries(building)


def test_manifest_provenance_separates_generation_and_verification_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_code_identity", lambda: {"stage": "verify"})
    monkeypatch.setattr(
        launcher, "_environment", lambda gpu: {"stage": "verify", "gpu": gpu}
    )
    provenance = launcher._manifest_provenance(
        {
            "code": {"stage": "generate"},
            "environment": {"stage": "generate", "gpu": "0"},
        },
        None,
    )
    assert provenance["generation"] == {
        "code": {"stage": "generate"},
        "environment": {"stage": "generate", "gpu": "0"},
    }
    assert provenance["verification"] == {
        "code": {"stage": "verify"},
        "environment": {"stage": "verify", "gpu": None},
    }


def test_verify_rejects_evaluation_manifest_report_hash_drift(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    raw = [{"id": 0, "sample": 0, "text": "a\n"}]
    scored = [
        {
            **raw[0],
            "n_actions": 1,
            "invalid": None,
            "collisions": [],
        }
    ]
    (root / "out.jsonl").write_text(json.dumps(raw[0]) + "\n", encoding="utf-8")
    (root / "scored.jsonl").write_text(json.dumps(scored[0]) + "\n", encoding="utf-8")
    (root / "metrics.md").write_text("report\n", encoding="utf-8")
    metrics = {
        "schema_version": "bricknet-unconditional-evaluation-v1",
        **launcher._aggregate_scored_metrics(scored),
        "collision": launcher._aggregate_scored_metrics(scored)[
            "mean_actions_before_first_failure"
        ],
        "input_sha256": _sha256(root / "out.jsonl"),
        "scored_sha256": _sha256(root / "scored.jsonl"),
    }
    _write_json(root / "metrics.json", metrics)
    report_hash = _sha256(root / "metrics.md")
    manifest = {
        "status": "complete",
        "input": {
            "path": str(root / "out.jsonl"),
            "sha256": metrics["input_sha256"],
            "rows": 1,
        },
        "scored": {
            "path": str(root / "scored.jsonl"),
            "sha256": metrics["scored_sha256"],
            "rows": 1,
        },
        "evaluation": {"expected_samples": 1, "expected_prompt": "a"},
        "metrics": metrics,
        "metrics_path": str(root / "metrics.json"),
        "metrics_sha256": _sha256(root / "metrics.json"),
        "report_path": str(root / "metrics.md"),
        "report_sha256": report_hash,
        "artifacts": {
            "input": {"path": str(root / "out.jsonl"), "sha256": metrics["input_sha256"]},
            "scored": {"path": str(root / "scored.jsonl"), "sha256": metrics["scored_sha256"]},
            "metrics": {"path": str(root / "metrics.json"), "sha256": _sha256(root / "metrics.json")},
            "report": {"path": str(root / "metrics.md"), "sha256": report_hash},
        },
    }
    _write_json(root / "evaluation_manifest.json", manifest)
    launcher._validate_evaluation_bundle(root, raw)
    _write_json(root / "metrics.json", {**metrics, "collision": 99.0})
    with pytest.raises(launcher.GateError, match="metric mismatch for collision"):
        launcher._validate_evaluation_bundle(root, raw)
    _write_json(root / "metrics.json", metrics)
    (root / "metrics.md").write_text("drift\n", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="artifact hash/path mismatch: report"):
        launcher._validate_evaluation_bundle(root, raw)


def test_dry_run_never_invokes_a_subprocess_for_missing_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run unexpectedly invoked a subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", fail_if_called)
    assert launcher.main(["--action", "generate", "--gpu", "0"]) == 0


def test_smoke_dry_run_plans_isolated_smoke_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("smoke dry-run unexpectedly invoked a subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", fail_if_called)
    assert launcher.main(["--action", "smoke", "--gpu", "0"]) == 0
    output = capsys.readouterr().out
    assert str(launcher._smoke_root() / "out.jsonl") in output
    assert str(launcher._smoke_root() / launcher.EVALUATION_SUBDIR) in output


def test_execute_generation_requires_manual_adapter_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("approval gate failed open")

    monkeypatch.setattr(launcher.subprocess, "run", fail_if_called)
    assert launcher.main(["--action", "generate", "--gpu", "0", "--execute"]) == 2


def test_verify_execute_is_blocked_without_a_complete_building_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("verify must not launch an external process")

    monkeypatch.setattr(launcher.subprocess, "run", fail_if_called)
    assert launcher.main(["--action", "verify", "--execute"]) == 2


def test_move_evaluation_bundle_rewrites_paths_before_outer_promotion(tmp_path: Path) -> None:
    building = tmp_path / "run.building"
    evaluation = building / launcher.EVALUATION_SUBDIR
    evaluation.mkdir(parents=True)
    _write_json(
        evaluation / "evaluation_manifest.json",
        {
            "status": "complete",
            "input": {"path": str(building / "out.jsonl")},
            "output_dir": str(evaluation),
            "scored": {"path": str(evaluation / "scored.jsonl")},
        },
    )
    for name in launcher.EVALUATION_OUTPUTS:
        if name != "evaluation_manifest.json":
            (evaluation / name).write_text("artifact\n", encoding="utf-8")

    launcher._move_evaluation_bundle(building, stable_root=tmp_path / "final")
    manifest = json.loads((building / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["path"] == str(tmp_path / "final" / "out.jsonl")
    assert manifest["output_dir"] == str(tmp_path / "final")
    assert manifest["scored"]["path"] == str(tmp_path / "final" / "scored.jsonl")
    assert not evaluation.exists()


def test_fake_subprocess_happy_path_promotes_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _install_complete_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "EXPECTED_SAMPLES", 2)
    monkeypatch.setattr(launcher, "EXPECTED_BATCH_SIZE", 1)
    calls: list[list[str]] = []

    def fake_run(command, *, cwd=None, env=None, check=False, **kwargs):
        del cwd, env, check
        command = [str(item) for item in command]
        if command[:2] == ["uname", "-p"]:
            return SimpleNamespace(returncode=0, stdout="unknown")
        calls.append(command)
        if "--model" in command and "--output" in command:
            output = Path(command[command.index("--output") + 1])
            rows = [
                {"id": index, "sample": 0, "source": "", "caption": "", "text": "a\n"}
                for index in range(launcher.EXPECTED_SAMPLES)
            ]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        elif "--input" in command and "--output-dir" in command:
            input_path = Path(command[command.index("--input") + 1])
            output_dir = Path(command[command.index("--output-dir") + 1])
            raw_rows = [json.loads(line) for line in input_path.read_text().splitlines()]
            scored_rows = [
                {**row, "n_actions": 1, "invalid": None, "collisions": []}
                for row in raw_rows
            ]
            output_dir.mkdir(parents=True)
            scored_path = output_dir / "scored.jsonl"
            scored_path.write_text(
                "".join(json.dumps(row) + "\n" for row in scored_rows), encoding="utf-8"
            )
            aggregate = launcher._aggregate_scored_metrics(scored_rows)
            metrics = {
                "schema_version": "bricknet-unconditional-evaluation-v1",
                "expected_prompt": launcher.EXPECTED_PROMPT,
                **aggregate,
                "collision": aggregate["mean_actions_before_first_failure"],
                "parsable": {
                    "count": aggregate["parsable_count"],
                    "rate": aggregate["parsable_rate"],
                },
                "clean": {
                    "count": aggregate["clean_count"],
                    "rate": aggregate["clean_rate"],
                },
                "input_sha256": _sha256(input_path),
                "scored_sha256": _sha256(scored_path),
            }
            _write_json(output_dir / "metrics.json", metrics)
            (output_dir / "metrics.md").write_text("fake report\n", encoding="utf-8")
            _write_json(
                output_dir / "evaluation_manifest.json",
                {
                    "status": "complete",
                    "input": {
                        "path": str(input_path),
                        "sha256": metrics["input_sha256"],
                        "rows": len(raw_rows),
                    },
                    "output_dir": str(output_dir),
                    "evaluation": {
                        "expected_samples": len(raw_rows),
                        "expected_prompt": launcher.EXPECTED_PROMPT,
                    },
                    "scored": {
                        "path": str(scored_path),
                        "sha256": metrics["scored_sha256"],
                        "rows": len(scored_rows),
                    },
                    "metrics": metrics,
                },
            )
        else:
            raise AssertionError(f"unexpected fake subprocess: {command}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert (
        launcher.main(
            [
                "--action",
                "generate",
                "--adapter-path",
                str(adapter),
                "--gpu",
                "0",
                "--adapter-approved",
                "--execute",
            ]
        )
        == 0
    )
    assert (
        launcher.main(
            ["--action", "evaluate", "--adapter-path", str(adapter), "--execute"]
        )
        == 0
    )
    assert launcher.main(["--action", "verify", "--adapter-path", str(adapter), "--execute"]) == 0

    assert len(calls) == 2
    final_root = launcher.OUTPUT_ROOT
    assert final_root.is_dir()
    assert not launcher._building_root().exists()
    assert {path.name for path in final_root.iterdir()} == {
        *launcher.FINAL_REQUIRED_OUTPUTS,
        launcher.FINAL_MANIFEST_NAME,
    }
    manifest = json.loads((final_root / launcher.FINAL_MANIFEST_NAME).read_text())
    assert manifest["provenance"]["generation"]["environment"]["gpu"] == "0"
    assert manifest["provenance"]["verification"]["environment"]["gpu"] is None
    evaluation_manifest = json.loads(
        (final_root / "evaluation_manifest.json").read_text()
    )
    assert evaluation_manifest["input"]["path"] == str(final_root / "out.jsonl")
    assert evaluation_manifest["output_dir"] == str(final_root)
