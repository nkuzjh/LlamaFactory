import argparse
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/launch_bricknet_pt_exp2.py"
SPEC = importlib.util.spec_from_file_location("launch_bricknet_pt_exp2_text8m_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _valid_adapter_config() -> dict[str, object]:
    _, expected, blockers = launcher._text8m_target_modules_contract()
    assert not blockers
    return {
        **launcher.TEXT8M_ADAPTER_CONFIG_CONTRACT,
        "target_modules": sorted(expected),
    }


def _install_endpoint(path: Path, *, step: int = 250_000) -> None:
    _write_json(path / "adapter_config.json", _valid_adapter_config())
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    _write_json(path / "trainer_state.json", {"global_step": step, "max_steps": 250_000})
    _write_json(path / "train_results.json", {"train_loss": 1.0})


def _install_resume_checkpoint(output: Path, *, step: int = 5_000, world_size: int = 1) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    _write_json(
        checkpoint / "adapter_config.json",
        {
            "base_model_name_or_path": launcher._text8m_contract.BASE_MODEL,
            "peft_type": "LORA",
            "r": 64,
            "lora_alpha": 128,
            "target_modules": sorted(launcher._text8m_target_modules_contract()[1]),
        },
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "training_args.bin").write_bytes(b"training-args")
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "max_steps": launcher.TEXT8M_EXPECTED_STEPS},
    )
    if world_size == 1:
        (checkpoint / "rng_state.pth").write_bytes(b"rng")
    else:
        for rank in range(world_size):
            (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    return checkpoint


class Text8mLauncherTest(unittest.TestCase):
    def test_training_yaml_pins_reviewed_base_revision(self) -> None:
        config = yaml.safe_load((launcher.CONFIG_ROOT / launcher.RUNS["text8m"].config).read_text(encoding="utf-8"))
        assert config["model_name_or_path"] == launcher._text8m_contract.BASE_MODEL
        assert config["model_revision"] == launcher._text8m_contract.BASE_REVISION

    def test_base_contract_delegates_snapshot_and_tokenizer_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model_name_or_path": launcher._text8m_contract.BASE_MODEL,
                        "model_revision": launcher._text8m_contract.BASE_REVISION,
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    launcher._text8m_contract,
                    "_check_snapshot",
                    return_value=({"eligible": False}, ["BASE_SNAPSHOT_HASH_MISMATCH:config.json"]),
                ) as snapshot,
                mock.patch.object(
                    launcher._text8m_contract,
                    "_check_tokenizer_semantics",
                    return_value=({"eligible": True}, []),
                ) as tokenizer,
            ):
                checks, blockers = launcher._text8m_base_contract_check(config_path)

        snapshot.assert_called_once_with()
        tokenizer.assert_called_once_with()
        assert "BASE_SNAPSHOT_HASH_MISMATCH:config.json" in blockers
        assert not checks["eligible"]

    def test_base_contract_rejects_unpinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(
                yaml.safe_dump({"model_name_or_path": launcher._text8m_contract.BASE_MODEL}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    launcher._text8m_contract,
                    "_check_snapshot",
                    return_value=({"eligible": True}, []),
                ),
                mock.patch.object(
                    launcher._text8m_contract,
                    "_check_tokenizer_semantics",
                    return_value=({"eligible": True}, []),
                ),
            ):
                checks, blockers = launcher._text8m_base_contract_check(config_path)

        assert "TEXT8M_BASE_REVISION_CONFIG_DRIFT" in blockers
        assert not checks["config"]["eligible"]

    def test_completion_requires_250k_state_and_train_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / "endpoint"
            _install_endpoint(endpoint)
            assert launcher._text8m_completion_check(endpoint)["eligible"]

            _write_json(
                endpoint / "trainer_state.json",
                {"global_step": 249_999, "max_steps": 250_000},
            )
            wrong_step = launcher._text8m_completion_check(endpoint)
            assert not wrong_step["eligible"]
            assert wrong_step["trainer_state"]["global_step"] == 249999

            _write_json(
                endpoint / "trainer_state.json",
                {"global_step": 250_000, "max_steps": 250_000},
            )
            (endpoint / "train_results.json").write_text("not-json\n", encoding="utf-8")
            invalid_results = launcher._text8m_completion_check(endpoint)
            assert not invalid_results["eligible"]
            assert not invalid_results["train_results"]["valid_json_object"]

    def test_completion_requires_real_root_and_frozen_adapter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "endpoint"
            _install_endpoint(endpoint)
            for key, wrong_value in (
                ("peft_type", "IA3"),
                ("r", 32),
                ("lora_alpha", 64),
                ("base_model_name_or_path", "Qwen/Qwen3.5-1.7B"),
            ):
                config = launcher.TEXT8M_ADAPTER_CONFIG_CONTRACT.copy()
                config[key] = wrong_value
                config["target_modules"] = sorted(launcher._text8m_target_modules_contract()[1])
                _write_json(endpoint / "adapter_config.json", config)
                checks = launcher._text8m_completion_check(endpoint)
                assert not checks["eligible"], key
                assert not checks["adapter_config"]["contract_eligible"], key
                _write_json(endpoint / "adapter_config.json", _valid_adapter_config())

            symlinked_root = root / "endpoint-link"
            symlinked_root.symlink_to(endpoint, target_is_directory=True)
            checks = launcher._text8m_completion_check(symlinked_root)
            assert not checks["eligible"]
            assert checks["output_directory"]["symlink"]

    def test_completion_rejects_missing_or_extra_target_modules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / "endpoint"
            _install_endpoint(endpoint)
            _, expected, _ = launcher._text8m_target_modules_contract()
            for target_modules in (
                sorted(expected - {"down_proj"}),
                sorted(expected | {"qkv_proj"}),
            ):
                config = _valid_adapter_config()
                config["target_modules"] = target_modules
                _write_json(endpoint / "adapter_config.json", config)
                checks = launcher._text8m_completion_check(endpoint)
                assert not checks["eligible"]
                assert not checks["adapter_config"]["target_modules"]["eligible"]

    def test_resume_rejects_target_module_set_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            checkpoint = _install_resume_checkpoint(output)
            _, expected, _ = launcher._text8m_target_modules_contract()
            config_path = checkpoint / "adapter_config.json"
            checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
            checkpoint_config["target_modules"] = sorted(expected - {"down_proj"})
            _write_json(config_path, checkpoint_config)
            checks, blockers = launcher._text8m_resume_checkpoint_check(
                checkpoint,
                output=output,
                world_size=1,
            )
            assert not checks["eligible"]
            assert "TEXT8M_RESUME_ADAPTER_CONFIG_DRIFT" in blockers
            assert not checks["adapter_config"]["target_modules"]["eligible"]

    def test_completion_rejects_symlink_empty_or_ambiguous_adapter_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "endpoint"
            _install_endpoint(endpoint)
            valid_weight = endpoint / "adapter_model.safetensors"
            legacy_weight = endpoint / "adapter_model.bin"

            legacy_weight.write_bytes(b"legacy")
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert checks["adapter_weights"]["files"] == ["adapter_model.safetensors", "adapter_model.bin"]
            legacy_weight.unlink()

            valid_weight.unlink()
            valid_weight.symlink_to(root / "missing.safetensors")
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert checks["adapter_weights"]["candidates"]["adapter_model.safetensors"]["symlink"]
            valid_weight.unlink()

            valid_weight.write_bytes(b"")
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["adapter_weights"]["candidates"]["adapter_model.safetensors"]["nonempty"]

    def test_completion_rejects_nonregular_or_symlinked_json_and_bad_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "endpoint"
            _install_endpoint(endpoint)

            trainer_state = endpoint / "trainer_state.json"
            trainer_state.unlink()
            trainer_state.mkdir()
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["trainer_state"]["regular_file"]
            trainer_state.rmdir()
            _write_json(endpoint / "trainer_state.json", {"global_step": 250_000.0, "max_steps": 250_000})
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["trainer_state"]["ok"]
            _write_json(endpoint / "trainer_state.json", {"global_step": True, "max_steps": 250_000})
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["trainer_state"]["ok"]

            results = endpoint / "train_results.json"
            results.unlink()
            results.symlink_to(root / "results.json")
            (root / "results.json").write_text(json.dumps({"train_loss": 1.0}) + "\n", encoding="utf-8")
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert checks["train_results"]["symlink"]
            results.unlink()
            results.write_bytes(b"")
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["train_results"]["nonempty"]

            _write_json(endpoint / "train_results.json", {"train_loss": math.nan})
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["train_results"]["finite_train_loss"]
            _write_json(endpoint / "train_results.json", {"train_loss": "1.0"})
            checks = launcher._text8m_completion_check(endpoint)
            assert not checks["eligible"]
            assert not checks["train_results"]["finite_train_loss"]

    def test_common_gate_does_not_accept_adapter_files_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config_root = temporary / "configs"
            config_root.mkdir()
            (config_root / launcher.RUNS["text8m"].config).write_text("{}\n", encoding="utf-8")
            save_root = temporary / "saves"
            endpoint = save_root / launcher.RUNS["text8m"].output
            _write_json(endpoint / "adapter_config.json", {"peft_type": "LORA"})
            (endpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            manifest = temporary / "manifest.json"
            _write_json(
                manifest,
                {"stats": {"unique_rows": 7_698_261}, "audit": {"eligible": True}},
            )
            view = temporary / "view"
            view.mkdir()
            with (
                mock.patch.object(launcher, "CONFIG_ROOT", config_root),
                mock.patch.object(launcher, "SAVE_ROOT", save_root),
                mock.patch.object(launcher, "TEXT_MANIFEST", manifest),
                mock.patch.object(launcher, "TEXT_VIEW", view),
                mock.patch.object(
                    launcher,
                    "_text8m_base_contract_check",
                    return_value=({"eligible": True}, []),
                ),
                mock.patch.object(
                    launcher,
                    "_length_cache_check",
                    return_value={"enabled": False, "eligible": True, "build_required": False},
                ),
                mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
                mock.patch.object(launcher, "_selected_gpu_uuids", return_value={"GPU-0"}),
                mock.patch.object(launcher, "_gpu_processes", return_value=[]),
            ):
                blockers, checks = launcher._check_common("text8m", "train")

        assert not checks["already_complete"]
        assert "OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED" in blockers
        assert "RUN_ALREADY_COMPLETE" not in blockers

    def test_execute_rejects_successful_process_without_complete_endpoint(self) -> None:
        args = argparse.Namespace(action="train", run="text8m", execute=True)
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(launcher, "SAVE_ROOT", Path(directory)),
                mock.patch.object(launcher, "_check_common", return_value=([], {})),
                mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
                mock.patch.object(launcher, "_prepare_length_cache"),
                mock.patch.object(
                    launcher,
                    "_text8m_prelaunch_cache_check",
                    return_value=({"eligible": True}, []),
                ),
                mock.patch.object(
                    launcher,
                    "_text8m_prelaunch_output_check",
                    return_value=({"eligible": True}, []),
                ),
                mock.patch.object(
                    launcher,
                    "_text8m_base_contract_check",
                    return_value=({"eligible": True}, []),
                ),
                mock.patch.object(launcher.subprocess, "run") as run,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(SystemExit, "valid 250000-step endpoint"):
                    launcher._run_train_or_predict(args)
        run.assert_called_once()

    def test_execute_rechecks_base_contract_after_cache_preparation(self) -> None:
        args = argparse.Namespace(action="train", run="text8m", execute=True)
        with (
            mock.patch.object(launcher, "_check_common", return_value=([], {})),
            mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
            mock.patch.object(launcher, "_prepare_length_cache"),
            mock.patch.object(
                launcher,
                "_text8m_prelaunch_cache_check",
                return_value=({"eligible": True}, []),
            ),
            mock.patch.object(
                launcher,
                "_text8m_prelaunch_output_check",
                return_value=({"eligible": True}, []),
            ),
            mock.patch.object(
                launcher,
                "_text8m_base_contract_check",
                return_value=(
                    {"eligible": False},
                    ["BASE_SNAPSHOT_HASH_MISMATCH:config.json"],
                ),
            ),
            mock.patch.object(launcher.subprocess, "run") as run,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "base contract changed"):
                launcher._run_train_or_predict(args)
        run.assert_not_called()

    def test_execute_rechecks_cache_after_prepare_and_blocks_drift(self) -> None:
        args = argparse.Namespace(action="train", run="text8m", execute=True)
        invalid_cache = {
            "enabled": True,
            "eligible": False,
            "error": "CACHE_MANIFEST_DRIFT",
        }
        stream = io.StringIO()
        with (
            mock.patch.object(launcher, "_check_common", return_value=([], {})),
            mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
            mock.patch.object(launcher, "_prepare_length_cache") as prepare,
            mock.patch.object(launcher, "_length_cache_check", return_value=invalid_cache) as cache_check,
            mock.patch.object(launcher.subprocess, "run") as run,
            redirect_stdout(stream),
        ):
            with self.assertRaisesRegex(SystemExit, "tokenized cache changed"):
                launcher._run_train_or_predict(args)

        prepare.assert_called_once()
        cache_check.assert_called_once_with(
            launcher.CONFIG_ROOT / launcher.RUNS["text8m"].config,
            strict_text8m=True,
        )
        run.assert_not_called()
        assert "text8m_prelaunch_cache_validation" in stream.getvalue()
        assert "TEXT8M_PRELAUNCH_CACHE_INVALID" in stream.getvalue()

    def test_resume_checkpoint_gate_accepts_full_single_and_dual_gpu_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for world_size in (1, 2):
                output = root / f"output-{world_size}"
                checkpoint = _install_resume_checkpoint(output, world_size=world_size)
                checks, blockers = launcher._text8m_resume_checkpoint_check(
                    checkpoint,
                    output=output,
                    world_size=world_size,
                )
                assert blockers == []
                assert checks["eligible"]

    def test_resume_checkpoint_gate_rejects_external_path_and_step_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "expected-output"
            checkpoint = _install_resume_checkpoint(root / "external-output", step=5_000)
            _write_json(
                checkpoint / "trainer_state.json",
                {"global_step": 4_999, "max_steps": launcher.TEXT8M_EXPECTED_STEPS},
            )
            checks, blockers = launcher._text8m_resume_checkpoint_check(
                checkpoint,
                output=output,
                world_size=1,
            )
            assert not checks["eligible"]
            assert "TEXT8M_RESUME_CHECKPOINT_OUTSIDE_EXPECTED_OUTPUT" in blockers
            assert "TEXT8M_RESUME_TRAINER_STATE_DRIFT" in blockers

    def test_resume_checkpoint_gate_requires_optimizer_scheduler_and_matching_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            checkpoint = _install_resume_checkpoint(output, world_size=1)
            (checkpoint / "optimizer.pt").unlink()
            checks, blockers = launcher._text8m_resume_checkpoint_check(
                checkpoint,
                output=output,
                world_size=2,
            )
            assert not checks["eligible"]
            assert "TEXT8M_RESUME_FULL_STATE_MISSING:optimizer.pt" in blockers
            assert "TEXT8M_RESUME_RNG_STATE_MISMATCH" in blockers

    def test_validated_resume_removes_only_incomplete_output_blocker_and_injects_path(self) -> None:
        args = argparse.Namespace(
            action="train",
            run="text8m",
            execute=False,
            resume_from_checkpoint=None,
            resume_approved=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_root = Path(directory) / "saves"
            output = save_root / launcher.RUNS["text8m"].output
            checkpoint = _install_resume_checkpoint(output)
            args.resume_from_checkpoint = str(checkpoint)
            stream = io.StringIO()
            with (
                mock.patch.object(launcher, "SAVE_ROOT", save_root),
                mock.patch.object(
                    launcher,
                    "_check_common",
                    return_value=(["OUTPUT_EXISTS_INCOMPLETE_REVIEW_REQUIRED"], {}),
                ),
                mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
                redirect_stdout(stream),
            ):
                launcher._run_train_or_predict(args)
        payload = json.loads(stream.getvalue())
        assert payload["ready"]
        assert payload["blockers"] == []
        assert "overwrite_output_dir=false" in payload["command"]
        assert f"resume_from_checkpoint={checkpoint}" in payload["command"]

    def test_parser_requires_resume_checkpoint_and_explicit_approval_together(self) -> None:
        argv = [
            str(launcher.SCRIPT if hasattr(launcher, "SCRIPT") else launcher.__file__),
            "--action",
            "train",
            "--run",
            "text8m",
            "--resume-from-checkpoint",
            "/tmp/checkpoint-5000",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            redirect_stdout(io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                launcher.parse_args()

    def test_text8m_runtime_env_dry_run_is_noncreating_and_data_disk_bound(self) -> None:
        with mock.patch.object(Path, "mkdir") as mkdir:
            env, checks = launcher._text8m_runtime_env(create=False)

        mkdir.assert_not_called()
        assert checks["eligible"]
        assert not checks["created"]
        for key in launcher.TEXT8M_RUNTIME_PATHS:
            assert env[key] == checks["effective_env"][key]
            assert checks["paths"][key]["under_data_disk"]
            assert checks["paths"][key]["under_data_mount"]
        for key, value in launcher.TEXT8M_RUNTIME_FLAGS.items():
            assert env[key] == value
            assert checks["effective_env"][key] == value

    def test_text8m_runtime_env_rejects_path_redirected_outside_data_disk(self) -> None:
        unsafe = dict(launcher.TEXT8M_RUNTIME_PATHS)
        unsafe["TMPDIR"] = Path("/tmp/pt-exp2-text8m-test")
        with mock.patch.object(launcher, "TEXT8M_RUNTIME_PATHS", unsafe):
            _, checks = launcher._text8m_runtime_env(create=False)

        assert not checks["eligible"]
        assert "TEXT8M_RUNTIME_PATH_INVALID:TMPDIR" in checks["blockers"]

    def test_execute_creates_runtime_dirs_after_blockers_and_shares_env(self) -> None:
        args = argparse.Namespace(action="train", run="text8m", execute=True)
        effective_env = {
            **{key: str(value) for key, value in launcher.TEXT8M_RUNTIME_PATHS.items()},
            **launcher.TEXT8M_RUNTIME_FLAGS,
        }
        dry_env = {"BASE": "dry"}
        execute_env = {"BASE": "execute"}
        dry_checks = {"eligible": True, "created": False, "blockers": [], "effective_env": effective_env}
        execute_checks = {"eligible": True, "created": True, "blockers": [], "effective_env": effective_env}
        runtime = mock.Mock(side_effect=[(dry_env, dry_checks), (execute_env, execute_checks)])
        stream = io.StringIO()
        with (
            mock.patch.object(launcher, "_check_common", return_value=([], {})),
            mock.patch.object(launcher, "_text8m_runtime_env", runtime),
            mock.patch.object(launcher, "_visible_gpu_selectors", return_value=["0"]),
            mock.patch.object(launcher, "_prepare_length_cache") as prepare,
            mock.patch.object(launcher, "_text8m_prelaunch_cache_check", return_value=({"eligible": True}, [])),
            mock.patch.object(launcher, "_text8m_base_contract_check", return_value=({"eligible": True}, [])),
            mock.patch.object(launcher, "_text8m_prelaunch_output_check", return_value=({"eligible": True}, [])),
            mock.patch.object(launcher, "_text8m_completion_check", return_value={"eligible": True}),
            mock.patch.object(launcher.subprocess, "run") as run,
            redirect_stdout(stream),
        ):
            launcher._run_train_or_predict(args)

        assert runtime.call_args_list == [mock.call(create=False), mock.call(create=True)]
        prepare_env = prepare.call_args.kwargs["env"]
        train_env = run.call_args.kwargs["env"]
        assert prepare_env is train_env
        assert train_env["BASE"] == "execute"
        assert "text8m_runtime_environment_ready" in stream.getvalue()

    def test_execute_with_blockers_never_creates_runtime_dirs(self) -> None:
        args = argparse.Namespace(action="train", run="text8m", execute=True)
        runtime = mock.Mock(
            return_value=(
                {"BASE": "dry"},
                {"eligible": True, "created": False, "blockers": [], "effective_env": {}},
            )
        )
        with (
            mock.patch.object(launcher, "_check_common", return_value=(["BLOCKED"], {})),
            mock.patch.object(launcher, "_text8m_runtime_env", runtime),
            mock.patch.object(launcher.subprocess, "run") as run,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "launch blocked"):
                launcher._run_train_or_predict(args)

        runtime.assert_called_once_with(create=False)
        run.assert_not_called()

    def test_text8m_cache_gate_uses_strict_manifest_contract_and_exact_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            cache = temporary / "cache"
            cache.mkdir()
            config = temporary / "train.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "length_column_name": "length",
                        "tokenized_path": str(cache),
                    }
                ),
                encoding="utf-8",
            )
            strict_check = mock.Mock(return_value={"eligible": True, "train_rows": 7_698_261})
            builder = mock.Mock(check_cache=strict_check)
            with mock.patch.object(launcher, "_load_length_cache_builder", return_value=builder):
                result = launcher._length_cache_check(config, strict_text8m=True)

            assert result["eligible"]
            strict_check.assert_called_once_with(
                cache,
                "length",
                "input_ids",
                config_path=config,
                require_manifest=True,
                full_length_validation=True,
            )

            strict_check.return_value = {"eligible": True, "train_rows": 7_698_260}
            with mock.patch.object(launcher, "_load_length_cache_builder", return_value=builder):
                wrong_rows = launcher._length_cache_check(config, strict_text8m=True)
            assert not wrong_rows["eligible"]
            assert wrong_rows["error"] == "TOKENIZED_CACHE_ROW_COUNT_MISMATCH"


if __name__ == "__main__":
    unittest.main()
