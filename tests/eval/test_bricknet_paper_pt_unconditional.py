"""Pure contract tests for the fixed BrickNet PT unconditional NS launcher.

These tests intentionally use temporary JSON/JSONL fixtures and monkeypatch
the launcher gates.  They must never load a model, touch CUDA or invoke a real
generator/evaluator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_paper_pt_unconditional.py"
SPEC = importlib.util.spec_from_file_location("paper_pt_unconditional_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_row(index: int, *, full_length: bool = True) -> dict[str, object]:
    return {
        "id": index,
        "sample": 0,
        "source": "",
        "caption": "",
        "text": "a0 FILE 1-1.dat\n",
        "generated_token_count": 4096 if full_length else 12,
        "generated_newline_tokens": 199 if full_length else 1,
        "stop_target_reached": full_length,
        "hit_max_new_tokens": full_length,
    }


def _scored_row(raw: dict[str, object], *, invalid: int | None = None, collisions: list[int] | None = None) -> dict[str, object]:
    return {
        **raw,
        "n_actions": 3,
        "invalid": invalid,
        "collisions": [] if collisions is None else collisions,
    }


def test_registry_has_exactly_four_fixed_identities_and_counts() -> None:
    expected_ids = {
        "bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
        "bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
        "bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
        "bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
    }
    assert set(launcher.EXPERIMENTS) == expected_ids
    assert set(launcher.NS_EXPERIMENT_IDENTITIES) == expected_ids
    assert len(launcher.EXPERIMENTS) == 4
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-8b-uncond-ns-n2048-v1"].expected_samples == 2048
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-14b-uncond-ns-n2048-v1"].expected_samples == 2048
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1"].expected_samples == 65536
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1"].expected_samples == 65536
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1"].adapter_revision == "3f5470a0cc45ecf5646d0d4e350704c9dc7e1807"
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1"].adapter_revision == "0f75d5e5922234f129a3bb42307c14c685f098f2"
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-8b-uncond-ns-n2048-v1"].adapter_revision == "a4c23748aee599577d65bf2581a8aa52e8a6bc40"
    assert launcher.EXPERIMENTS["bricknet-pt-qwen3-14b-uncond-ns-n2048-v1"].adapter_revision == "510afeeea5ba20f7202bc4c9c355b57594d07575"


def test_parser_accepts_recorded_gpu_spellings_and_rejects_extra_modes() -> None:
    args = launcher.parse_args(["--run", "qwen3_8b", "--action", "generate", "--gpus", "0", "1"])
    assert args.experiment == "bricknet-pt-qwen3-8b-uncond-ns-n2048-v1"
    assert args.gpus == "0,1"
    assert args.gpu == "0,1"
    assert launcher._parse_gpu_selector("0,00") == ()
    assert launcher._parse_gpu_selector("01") == ()
    with pytest.raises(SystemExit):
        launcher.parse_args(["--experiment", "AS", "--action", "generate", "--gpu", "0"])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--experiment", "qwen3_8b", "--action", "generate", "--gpu", "0", "--n", "3"])


def test_generation_and_evaluation_commands_freeze_protocol() -> None:
    exp = launcher._resolve_experiment("qwen3_1_7b")
    command = launcher._generation_command(exp, Path("/tmp/out.jsonl"), samples=exp.expected_samples)
    assert command[command.index("--model") + 1] == str(exp.base_path)
    assert command[command.index("--lora") + 1] == str(exp.adapter_path)
    assert command[command.index("--prompt") + 1] == "a"
    assert command[command.index("--num_samples") + 1] == "65536"
    assert command[command.index("--max_new_tokens") + 1] == "4096"
    assert command[command.index("--temperature") + 1] == "1.0"
    assert command[command.index("--top_k") + 1] == "20"
    assert command[command.index("--top_p") + 1] == "0.95"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--stop_after_newlines") + 1] == "199"
    assert "--seed" not in command
    assert launcher._protocol(exp)["generation_config"] == {
        "eos_token_id": [151645, 151643],
        "bos_token_id": 151643,
        "pad_token_id": 151643,
        "effective_generation_pad_token_id": 151645,
    }
    evaluation = launcher._evaluation_command("out.jsonl", "evaluation", expected_samples=65536)
    assert evaluation[evaluation.index("--require-generated-newlines") + 1] == "199"
    assert evaluation[evaluation.index("--expected-samples") + 1] == "65536"
    assert evaluation[-1] == "--execute"
    multi_gpu = launcher._generation_command(exp, Path("/tmp/out.jsonl"), samples=exp.expected_samples, gpus="0,1")
    assert multi_gpu[0] == str(launcher.LLAMAFACTORY_PYTHON)
    assert multi_gpu[1:4] == ["-m", "accelerate.commands.launch", "--multi_gpu"]
    for flag, value in (
        ("--num_machines", "1"),
        ("--mixed_precision", "no"),
        ("--dynamo_backend", "no"),
        ("--main_process_port", "0"),
        ("--num_processes", "2"),
        ("--gpu_ids", "all"),
    ):
        assert multi_gpu[multi_gpu.index(flag) + 1] == value


def test_raw_token_metadata_is_checked_and_newline_target_is_authoritative() -> None:
    rows = [_raw_row(0)]
    metadata = launcher._validate_raw_rows(rows, 1)
    summary = launcher._token_metadata_summary(metadata)
    assert summary["token_full_length_count"] == 1
    assert summary["token_full_length_rate"] == 1.0
    # Reaching both limits is accepted: newline target, not hit_max alone, is
    # the authoritative full-length condition.
    short = [_raw_row(0, full_length=False)]
    short[0]["generated_token_count"] = 4096
    short[0]["hit_max_new_tokens"] = True
    launcher._validate_raw_rows(short, 1)
    assert launcher._token_metadata_summary(short)["token_full_length_count"] == 0
    malformed = [_raw_row(0)]
    malformed[0]["hit_max_new_tokens"] = False
    with pytest.raises(launcher.GateError, match="hit_max_new_tokens"):
        launcher._validate_raw_rows(malformed, 1)


def test_metrics_recompute_connectivity_and_collision_prefix() -> None:
    rows = [
        _scored_row(_raw_row(0), invalid=None, collisions=[]),
        _scored_row(_raw_row(1), invalid=1, collisions=[2]),
    ]
    metrics = launcher._aggregate_scored_metrics(rows)
    assert metrics["parsable_rate"] == 0.5
    assert metrics["connectivity_mean"] == 2.0
    # min(connectivity_prefix=1, collision index=2) = 1.
    assert metrics["collision_mean"] == 0.5 * 3 + 0.5 * 1


def test_evaluation_manifest_and_generation_gate_are_strict(tmp_path: Path) -> None:
    root = tmp_path / "building"
    stable = tmp_path / "final"
    root.mkdir()
    raw_rows = [_raw_row(0)]
    scored_rows = [_scored_row(raw_rows[0])]
    raw_path = root / launcher.RAW_NAME
    scored_path = root / launcher.SCORED_NAME
    metrics_path = root / launcher.METRICS_NAME
    report_path = root / launcher.REPORT_NAME
    manifest_path = root / launcher.EVALUATION_MANIFEST_NAME
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(scored_path, scored_rows)
    expected = launcher._aggregate_scored_metrics(scored_rows)
    metrics = {
        **expected,
        "schema_version": launcher.EVALUATION_SCHEMA_VERSION,
        "input_sha256": _sha256(raw_path),
        "scored_sha256": _sha256(scored_path),
        "generation_gate": {
            "enabled": True,
            "unit": "generated_token",
            "required_generated_newlines": launcher.EXPECTED_STOP_AFTER_NEWLINES,
            "passed_count": 1,
            "failed_count": 0,
        },
    }
    _write_json(metrics_path, metrics)
    report_path.write_text("report\n", encoding="utf-8")
    metric_hash = _sha256(metrics_path)
    report_hash = _sha256(report_path)
    input_hash = _sha256(raw_path)
    scored_hash = _sha256(scored_path)
    manifest = {
        "schema_version": launcher.EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "evaluation": {
            "expected_samples": 1,
            "expected_prompt": launcher.EXPECTED_PROMPT,
            "require_generated_newlines": launcher.EXPECTED_STOP_AFTER_NEWLINES,
            "token_level_full_length_gate": True,
        },
        "input": {
            "path": str(stable / launcher.RAW_NAME),
            "sha256": input_hash,
            "rows": 1,
        },
        "scored": {
            "path": str(stable / launcher.SCORED_NAME),
            "sha256": scored_hash,
            "rows": 1,
        },
        "metrics": metrics,
        "metrics_path": str(stable / launcher.METRICS_NAME),
        "metrics_sha256": metric_hash,
        "report_path": str(stable / launcher.REPORT_NAME),
        "report_sha256": report_hash,
        "sha256": {
            "input": input_hash,
            "scored": scored_hash,
            "metrics": metric_hash,
            "metrics_md": report_hash,
        },
        "artifacts": {
            "input": {"path": str(stable / launcher.RAW_NAME), "sha256": input_hash},
            "scored": {"path": str(stable / launcher.SCORED_NAME), "sha256": scored_hash},
            "metrics": {"path": str(stable / launcher.METRICS_NAME), "sha256": metric_hash},
            "report": {"path": str(stable / launcher.REPORT_NAME), "sha256": report_hash},
        },
    }
    _write_json(manifest_path, manifest)
    bundle = launcher._validate_evaluation_bundle(root, raw_rows, stable_root=stable)
    assert bundle["scored_rows"] == 1

    for field, value, message in (
        ("schema_version", "wrong", "schema mismatch"),
        ("require_generated_newlines", None, "newline protocol mismatch"),
        ("token_level_full_length_gate", False, "token full-length gate"),
    ):
        altered = dict(manifest)
        if field == "schema_version":
            altered[field] = value
        else:
            altered["evaluation"] = {**manifest["evaluation"], field: value}
        with pytest.raises(launcher.GateError, match=message):
            launcher._validate_evaluation_manifest(altered, expected_samples=1, stable_root=stable)

    bad_gate = {
        **metrics,
        "generation_gate": {
            **metrics["generation_gate"],
            "passed_count": 0,
            "failed_count": 1,
        },
    }
    _write_json(metrics_path, bad_gate)
    _write_json(manifest_path, {**manifest, "metrics": bad_gate})
    with pytest.raises(launcher.GateError, match="does not cover all samples"):
        launcher._validate_evaluation_bundle(root, raw_rows, stable_root=stable)


def test_base_identity_accepts_single_shard_and_requires_real_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "Qwen3-0.6B"
    base.mkdir()
    for name in launcher.BASE_REQUIRED_FILES:
        value = (
            launcher.QWEN3_CONFIG_FINGERPRINTS["Qwen3-0.6B"]
            if name == "config.json"
            else launcher._generation_config_source_protocol()
            if name == "generation_config.json"
            else {}
        )
        _write_json(base / name, value)
    (base / "model.safetensors").write_bytes(b"single-shard-fixture")
    exp = replace(
        launcher._resolve_experiment("qwen3_0_6b"),
        base_path=base,
        base_revision="workspace-local",
    )

    checks, blockers = launcher._check_base(exp)
    assert blockers == []
    assert checks["weight_index"]["single_shard_without_index"] is True
    assert launcher._valid_sha256(checks["weight_files"][0]["sha256"])

    wrong_generation = launcher._generation_config_source_protocol()
    wrong_generation["eos_token_id"] = [launcher.EXPECTED_EOS_TOKEN_ID]
    _write_json(base / "generation_config.json", wrong_generation)
    _, blockers = launcher._check_base(exp)
    assert "BASE_GENERATION_CONFIG_SEMANTICS_MISMATCH" in blockers
    _write_json(base / "generation_config.json", launcher._generation_config_source_protocol())

    wrong_config = dict(launcher.QWEN3_CONFIG_FINGERPRINTS["Qwen3-0.6B"])
    wrong_config["hidden_size"] = 2048
    _write_json(base / "config.json", wrong_config)
    _, blockers = launcher._check_base(exp)
    assert "BASE_CONFIG_SEMANTIC_FINGERPRINT_MISMATCH" in blockers
    _write_json(base / "config.json", launcher.QWEN3_CONFIG_FINGERPRINTS["Qwen3-0.6B"])

    original_sha256 = launcher._sha256
    monkeypatch.setattr(
        launcher,
        "_sha256",
        lambda path: None if path == base / "config.json" else original_sha256(path),
    )
    _, blockers = launcher._check_base(exp)
    assert "BASE_FILE_UNHASHABLE:config.json" in blockers

    monkeypatch.setattr(launcher, "_sha256", original_sha256)
    _write_json(
        base / "tokenizer.json",
        {
            "model": {"vocab": {"a": 64, "Ċ": 198}},
            "added_tokens": [
                {"id": 151643, "content": launcher.EXPECTED_PAD_TOKEN},
                {"id": launcher.EXPECTED_EOS_TOKEN_ID, "content": launcher.EXPECTED_EOS_TOKEN},
            ],
        },
    )
    _write_json(
        base / "tokenizer_config.json",
        {
            "eos_token": launcher.EXPECTED_EOS_TOKEN,
            "pad_token": launcher.EXPECTED_PAD_TOKEN,
            "added_tokens_decoder": {
                "151643": {"content": launcher.EXPECTED_PAD_TOKEN},
                str(launcher.EXPECTED_EOS_TOKEN_ID): {"content": launcher.EXPECTED_EOS_TOKEN}
            },
        },
    )
    tokenizer_checks, tokenizer_blockers = launcher._check_tokenizer(exp)
    assert tokenizer_blockers == []
    assert tokenizer_checks["pad_id"] == launcher.EXPECTED_GENERATION_PAD_TOKEN_ID
    assert tokenizer_checks["pad_added_ids"] == [launcher.EXPECTED_GENERATION_PAD_TOKEN_ID]
    assert tokenizer_checks["pad_decoder_content"] == launcher.EXPECTED_PAD_TOKEN
    assert tokenizer_checks["effective_generation_pad_token_id"] == launcher.EXPECTED_EOS_TOKEN_ID

    (base / "model.safetensors").unlink()
    _write_json(
        base / "model.safetensors.index.json",
        {
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        },
    )
    (base / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (base / "model-00002-of-00002.safetensors").write_bytes(b"two")
    checks, blockers = launcher._check_base(exp)
    assert blockers == []
    assert checks["weight_index"]["valid"] is True
    (base / "model-00002-of-00002.safetensors").unlink()
    _, blockers = launcher._check_base(exp)
    assert "BASE_WEIGHT_FILE_MISSING:model-00002-of-00002.safetensors" in blockers


def test_collision_identity_hashes_catalog_and_tiny_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    inset = data_root / launcher.INSET_ROOT_NAME
    catalog = tmp_path / "catalog"
    inset.mkdir(parents=True)
    catalog.mkdir()
    (inset / "b.ply").write_bytes(b"two")
    (inset / "a.ply").write_bytes(b"one")
    for name in launcher.COLLISION_CATALOG_FILES:
        (catalog / name).write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(launcher, "BRICKNET_DATA", data_root)
    monkeypatch.setattr(launcher, "CATALOG_ROOT", catalog)
    monkeypatch.setattr(launcher, "EXPECTED_COLLISION_MESHES", 2)

    identity = launcher._collision_assets_identity()
    assert identity["inset"]["count"] == 2
    assert identity["inset"]["total_bytes"] == 6
    assert launcher._valid_sha256(identity["inset"]["manifest_sha256"])
    assert all(
        launcher._valid_sha256(identity["catalog"][name]["sha256"])
        for name in launcher.COLLISION_CATALOG_FILES
    )
    _, blockers = launcher._check_collision_assets()
    assert blockers == []
    before = identity["inset"]["manifest_sha256"]
    (inset / "b.ply").write_bytes(b"changed")
    after = launcher._tree_identity(inset)["manifest_sha256"]
    assert after != before


def test_row_count_and_protocol_drift_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(launcher.GateError, match="expected 2 generated rows"):
        launcher._validate_raw_rows([_raw_row(0)], 2)
    # Use tiny local fixtures: constructing a formal run_spec must not hash
    # the real 8B checkpoint during a pure contract test.
    fixture_root = tmp_path / "paper-launcher-fixture"
    base = fixture_root / "base"
    adapter = fixture_root / "adapter-r1"
    base.mkdir(parents=True, exist_ok=True)
    adapter.mkdir(parents=True, exist_ok=True)
    for name in launcher.BASE_REQUIRED_FILES:
        value = (
            launcher.QWEN3_CONFIG_FINGERPRINTS["Qwen3-8B"]
            if name == "config.json"
            else launcher._generation_config_source_protocol()
            if name == "generation_config.json"
            else {}
        )
        (base / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"tiny-base")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"tiny-adapter")
    exp = replace(
        launcher._resolve_experiment("qwen3_8b"),
        base_path=base,
        base_revision="workspace-local",
        adapter_path=adapter,
        adapter_revision="adapter-r1",
    )
    collision_identity = {
        "catalog": {
            name: {"path": str(tmp_path / name), "exists": True, "sha256": "0" * 64}
            for name in launcher.COLLISION_CATALOG_FILES
        },
        "inset": {
            "path": str(tmp_path / "inset"),
            "exists": True,
            "count": launcher.EXPECTED_COLLISION_MESHES,
            "total_bytes": 1,
            "manifest_sha256": "1" * 64,
            "complete": True,
        },
    }
    monkeypatch.setattr(launcher, "_collision_assets_identity", lambda: collision_identity)
    spec = launcher._run_spec(exp, scope="formal", status="generated", gpus="0")
    launcher._validate_run_spec(spec, exp, scope="formal")
    tampered = dict(spec)
    tampered["protocol"] = {**spec["protocol"], "top_k": 50}
    with pytest.raises(launcher.GateError, match="protocol drift"):
        launcher._validate_run_spec(tampered, exp, scope="formal")
    assert spec["commands"]["generation"]["argv"] == launcher._generation_command(
        exp,
        launcher._building_root(exp) / launcher.RAW_NAME,
        samples=exp.expected_samples,
        gpus="0",
    )
    tampered_command = dict(spec)
    tampered_command["commands"] = {
        **spec["commands"],
        "generation": {
            "argv": [*spec["commands"]["generation"]["argv"], "--unexpected"]
        },
    }
    with pytest.raises(launcher.GateError, match="command argv drift"):
        launcher._validate_run_spec(tampered_command, exp, scope="formal")
    wrong_count = dict(spec)
    wrong_count["protocol"] = {**spec["protocol"], "num_samples": launcher.SMOKE_SAMPLES}
    with pytest.raises(launcher.GateError, match="formal run_spec sample count"):
        launcher._validate_run_spec(wrong_count, exp, scope="formal")
    wrong_environment = dict(spec)
    wrong_environment["environment"] = {**spec["environment"], "world_size": 2}
    with pytest.raises(launcher.GateError, match="world-size/GPU provenance"):
        launcher._validate_run_spec(wrong_environment, exp, scope="formal")
    wrong = dict(spec)
    wrong["experiment_id"] = launcher._resolve_experiment("qwen3_14b").identity
    with pytest.raises(launcher.GateError, match="identity mismatch"):
        launcher._validate_run_spec(wrong, exp, scope="formal")
    incomplete_code = dict(spec)
    incomplete_code["code"] = {
        **spec["code"],
        "score": {**spec["code"]["score"], "sha256": None},
    }
    with pytest.raises(launcher.GateError, match="code identity is incomplete"):
        launcher._validate_run_spec(incomplete_code, exp, scope="formal")


def test_code_identity_gate_rejects_missing_or_unhashable_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = launcher._code_identity_paths()
    complete = {
        name: {"path": str(path), "sha256": "0" * 64}
        for name, path in paths.items()
    }
    assert launcher._code_identity_complete(complete)
    missing = dict(complete)
    del missing["tree"]
    assert not launcher._code_identity_complete(missing)
    unhashable = dict(complete)
    unhashable["score"] = {"path": str(paths["score"]), "sha256": None}
    assert not launcher._code_identity_complete(unhashable)

    monkeypatch.setattr(launcher, "_code_identity", lambda: unhashable)
    _, blockers = launcher._check_runtime()
    assert "CODE_IDENTITY_INCOMPLETE" in blockers


def test_dry_run_has_no_subprocess_and_reports_isolated_paths(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    exp = launcher._resolve_experiment("qwen3_8b")
    report = {
        "schema_version": launcher.SCHEMA_VERSION,
        "action": "generate",
        "experiment": exp.identity,
        "experiment_key": exp.key,
        "eligible": False,
        "blockers": ["fixture"],
        "checks": {},
        "output_root": str(launcher._exp_output_root(exp)),
        "building_root": str(launcher._building_root(exp)),
        "smoke_root": str(launcher._smoke_root(exp)),
        "pilot_root": str(launcher._pilot_root(exp)),
    }
    monkeypatch.setattr(launcher, "preflight", lambda action, experiment, gpus: report)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run subprocess"))
    assert launcher.main(["--experiment", exp.key, "--action", "generate", "--gpu", "0"]) == 0
    output = capsys.readouterr().out
    assert str(launcher._building_root(exp)) in output
    assert "--stop_after_newlines 199" in output


def test_invalid_gpu_planning_is_reported_and_execute_fails_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = launcher._resolve_experiment("qwen3_8b")
    report = {
        "schema_version": launcher.SCHEMA_VERSION,
        "action": "generate",
        "experiment": exp.identity,
        "experiment_key": exp.key,
        "eligible": True,
        "blockers": [],
        "checks": {},
        "output_root": "",
        "building_root": "",
        "smoke_root": "",
        "pilot_root": "",
    }
    monkeypatch.setattr(launcher, "preflight", lambda action, experiment, gpus: report)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("planning must not execute"))

    assert launcher.main(["--experiment", exp.key, "--action", "generate", "--gpu", "0,0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planning_blocker"] == "invalid GPU selector"
    assert payload["planned_commands"] == []
    assert payload["eligible"] is False
    assert any(item.startswith("PLANNING_COMMAND_INVALID:") for item in payload["blockers"])

    assert launcher.main(
        [
            "--experiment",
            exp.key,
            "--action",
            "generate",
            "--gpu",
            "0,0",
            "--execute",
            "--artifacts-approved",
        ]
    ) == 2
    assert "planning blocked" in capsys.readouterr().err


def test_preflight_requires_a_valid_gpu_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = launcher._resolve_experiment("qwen3_8b")
    empty_check = lambda experiment: ({"eligible": True}, [])
    monkeypatch.setattr(launcher, "_check_base", empty_check)
    monkeypatch.setattr(launcher, "_check_adapter", empty_check)
    monkeypatch.setattr(launcher, "_check_tokenizer", empty_check)
    monkeypatch.setattr(launcher, "_check_runtime", lambda: ({"eligible": True}, []))
    monkeypatch.setattr(launcher, "_check_collision_assets", lambda: ({"eligible": True}, []))

    missing_gpu = launcher.preflight("preflight", exp)
    assert "GPU_SELECTOR_REQUIRED_OR_INVALID" in missing_gpu["blockers"]
    valid_gpu = launcher.preflight("preflight", exp, "0")
    assert "GPU_SELECTOR_REQUIRED_OR_INVALID" not in valid_gpu["blockers"]
    assert valid_gpu["checks"]["gpu"]["world_size"] == 1


def test_execute_generation_requires_both_gates(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    exp = launcher._resolve_experiment("qwen3_8b")
    report = {"eligible": True, "blockers": [], "action": "generate", "experiment": exp.identity, "experiment_key": exp.key, "schema_version": launcher.SCHEMA_VERSION, "checks": {}, "output_root": "", "building_root": "", "smoke_root": "", "pilot_root": ""}
    monkeypatch.setattr(launcher, "preflight", lambda action, experiment, gpus: report)
    monkeypatch.setattr(launcher, "_planned_commands", lambda action, exp, gpus: [])
    assert launcher.main(["--experiment", exp.key, "--action", "generate", "--gpu", "0", "--execute"]) == 2
    assert "artifacts-approved" in capsys.readouterr().err


def test_building_entry_set_and_atomic_promotion_refuse_tamper_or_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    building = tmp_path / "run.building"
    building.mkdir()
    for name in launcher.STAGED_OUTPUTS:
        (building / name).write_text("artifact\n", encoding="utf-8")
    (building / "unexpected.partial").write_text("partial\n", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="entry set mismatch"):
        launcher._check_building_entries(building)

    exp = launcher._resolve_experiment("qwen3_8b")
    final = tmp_path / "final"
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", tmp_path)
    # The public promotion path fails before any rename when final exists.
    final.mkdir()
    monkeypatch.setattr(launcher, "_exp_output_root", lambda experiment: final)
    with pytest.raises(launcher.GateError, match="overwrite final"):
        launcher._verify_and_promote(exp, gpus="0")


def test_atomic_noreplace_preserves_existing_targets_and_sources(tmp_path: Path) -> None:
    source_file = tmp_path / "source-file"
    destination_file = tmp_path / "destination-file"
    source_file.write_text("source", encoding="utf-8")
    destination_file.write_text("keep", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="overwrite existing path"):
        launcher._rename_noreplace(source_file, destination_file)
    assert source_file.read_text(encoding="utf-8") == "source"
    assert destination_file.read_text(encoding="utf-8") == "keep"

    source_dir = tmp_path / "source-dir"
    destination_dir = tmp_path / "destination-dir"
    source_dir.mkdir()
    (source_dir / "payload").write_text("source", encoding="utf-8")
    destination_dir.mkdir()
    (destination_dir / "payload").write_text("keep", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="overwrite existing path"):
        launcher._rename_noreplace(source_dir, destination_dir)
    assert (source_dir / "payload").read_text(encoding="utf-8") == "source"
    assert (destination_dir / "payload").read_text(encoding="utf-8") == "keep"

    destination_json = tmp_path / "already.json"
    destination_json.write_text("keep", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="overwrite existing file"):
        launcher._write_json_new(destination_json, {"new": True})
    assert destination_json.read_text(encoding="utf-8") == "keep"

    building = tmp_path / "building"
    evaluation = building / launcher.EVALUATION_SUBDIR
    evaluation.mkdir(parents=True)
    for name in launcher.EVALUATION_OUTPUTS:
        if name == launcher.EVALUATION_MANIFEST_NAME:
            _write_json(evaluation / name, {})
        else:
            (evaluation / name).write_text("source", encoding="utf-8")
    existing_scored = building / launcher.SCORED_NAME
    existing_scored.write_text("keep", encoding="utf-8")
    with pytest.raises(launcher.GateError, match="overwrite evaluation output"):
        launcher._move_evaluation_bundle(building, stable_root=tmp_path / "final")
    assert (evaluation / launcher.SCORED_NAME).read_text(encoding="utf-8") == "source"
    assert existing_scored.read_text(encoding="utf-8") == "keep"
