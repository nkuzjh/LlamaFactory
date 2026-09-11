from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from datasets import Dataset, DatasetDict


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_tokenized_cache_with_length.py"
SPEC = importlib.util.spec_from_file_location("build_tokenized_cache_with_length", SCRIPT)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_config(
    path: Path,
    *,
    dataset: str = "tiny",
    eval_dataset: str | None = None,
    dataset_dir: Path | None = None,
) -> None:
    fields = [
        "model_name_or_path: example/model",
        "model_revision: pinned-revision",
        "stage: pt",
        f"dataset: {dataset}",
        "template: qwen3_5_nothink",
        "cutoff_len: 16",
        "packing: false",
        "length_column_name: length",
    ]
    if eval_dataset is not None:
        fields.append(f"eval_dataset: {eval_dataset}")
    if dataset_dir is not None:
        fields.append(f"dataset_dir: {dataset_dir}")
    path.write_text("\n".join(fields) + "\n", encoding="utf-8")


def _save_source(path: Path, *, eval_rows: int | None = None) -> None:
    train = Dataset.from_dict(
        {
            "input_ids": [[1], [2, 3, 4], [5, 6]],
            "attention_mask": [[1], [1, 1, 1], [1, 1]],
        }
    )
    if eval_rows is None:
        train.save_to_disk(str(path))
        return
    validation = Dataset.from_dict(
        {
            "input_ids": [[index + 10] for index in range(eval_rows)],
            "attention_mask": [[1] for _ in range(eval_rows)],
        }
    )
    DatasetDict({"train": train, "validation": validation}).save_to_disk(str(path))


def _ordered_sha(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_pt_fixture(
    data: Path,
    *,
    texts: list[str] | None = None,
    audit: dict[str, object] | None = None,
) -> Path:
    texts = texts or ["a", "b", "c"]
    registered = "bricknet_pt_exp2/text8m_train"
    canonical_path = data / "bricknet_pt_exp2/text8m/manifest.json"
    shard_path = canonical_path.parent / "part-00000.jsonl"
    raw = b"".join(
        (json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for text in texts
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(raw)
    shard_sha = hashlib.sha256(raw).hexdigest()
    ordered = _ordered_sha(texts)
    manifest_audit: dict[str, object] = {
        "completed": True,
        "parse_eligible": True,
        "eligible": True,
        "rows_audited": len(texts),
        "expected_rows": len(texts),
        "parse_errors": [],
    }
    if audit:
        manifest_audit.update(audit)
    _write_json(
        data / "dataset_info.json",
        {
            BUILDER.PT_EXP2_DATASET: {"file_name": registered},
            BUILDER.PT_EXP2_EVAL_DATASET: {
                "file_name": "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
            },
        },
    )
    _write_json(
        canonical_path,
        {
            "stats": {"unique_rows": len(texts)},
            "audit": manifest_audit,
            "dataset_sha256": ordered,
            "ordered_corpus_sha256": ordered,
            "shard_set_sha256": hashlib.sha256(shard_sha.encode("ascii")).hexdigest(),
            "shards": [{"file": shard_path.name, "rows": len(texts), "bytes": len(raw), "sha256": shard_sha}],
        },
    )
    train_view = data / registered
    train_view.mkdir(parents=True, exist_ok=True)
    (train_view / shard_path.name).symlink_to(shard_path)
    eval_path = data / "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
    eval_raw = b"".join(
        (json.dumps({"text": f"val-{index}"}, separators=(",", ":")) + "\n").encode("utf-8")
        for index in range(BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_bytes(eval_raw)
    _write_json(
        eval_path.with_suffix(".manifest.json"),
        {
            "schema_version": BUILDER.PT_EXP2_EVAL_SCHEMA_VERSION,
            "experiment": BUILDER.PT_EXP2_EVAL_EXPERIMENT,
            "role": BUILDER.PT_EXP2_EVAL_ROLE,
            "seed": BUILDER.PT_EXP2_EVAL_SEED,
            "rows": BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS,
            "output": str(eval_path.resolve()),
            "output_sha256": hashlib.sha256(eval_raw).hexdigest(),
            "not_val511_overfit": True,
        },
    )
    return canonical_path


@contextmanager
def _fake_pt_snapshot(root: Path) -> Iterator[Path]:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    hashes: dict[str, str] = {}
    for name in BUILDER.PT_EXP2_SNAPSHOT_HASHES:
        path = snapshot / name
        path.write_bytes((name + "\n").encode("utf-8"))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (
        mock.patch.object(BUILDER, "PT_EXP2_SNAPSHOT", snapshot),
        mock.patch.object(BUILDER, "PT_EXP2_SNAPSHOT_HASHES", hashes),
    ):
        yield snapshot


def _pt_config(path: Path, data: Path) -> None:
    _write_config(
        path,
        dataset=BUILDER.PT_EXP2_DATASET,
        eval_dataset=BUILDER.PT_EXP2_EVAL_DATASET,
        dataset_dir=data,
    )
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("example/model", BUILDER.PT_EXP2_MODEL)
        .replace("pinned-revision", BUILDER.PT_EXP2_MODEL_REVISION),
        encoding="utf-8",
    )


@contextmanager
def _prepared_legacy_pt_cache(
    root: Path,
) -> Iterator[tuple[Path, Path, Path, bytes, dict[str, object]]]:
    """Yield a tiny cache shaped like the one reviewed for manifest repair."""
    data = root / "data"
    config = root / "train.yaml"
    source = root / "source"
    output = root / "cache"
    _write_pt_fixture(data)
    _pt_config(config, data)
    _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

    with (
        mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
        mock.patch.object(
            BUILDER,
            "_build_base_cache",
            side_effect=lambda _config, path: shutil.copytree(source, path),
        ),
        _fake_pt_snapshot(root),
    ):
        BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
        manifest_path = output / BUILDER.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        live_metadata = BUILDER._cache_metadata_evidence(
            BUILDER._cache_datasets(output), output, reject_symlinks=True
        )
        manifest["cache_metadata_evidence"]["fingerprints"] = dict(
            BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS
        )
        manifest["builder"]["script_sha256"] = BUILDER.PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256
        _write_json(manifest_path, manifest)
        original_bytes = manifest_path.read_bytes()
        original_manifest_sha256 = hashlib.sha256(original_bytes).hexdigest()

        with (
            mock.patch.object(
                BUILDER, "PT_EXP2_LEGACY_MANIFEST_SHA256", original_manifest_sha256
            ),
            mock.patch.object(
                BUILDER,
                "PT_EXP2_STABLE_CACHE_FINGERPRINTS",
                dict(live_metadata["fingerprints"]),
            ),
        ):
            yield config, output, manifest_path, original_bytes, live_metadata


class TokenizedCacheManifestTest(unittest.TestCase):
    def test_cache_parser_forces_cpu_and_disables_mixed_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "train.yaml"
            _write_config(config)
            config.write_text(
                config.read_text(encoding="utf-8") + "bf16: true\nfp16: true\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=False):
                _, _, training_args, _ = BUILDER._parse_build_args(
                    BUILDER._load_config(config), root / "cache"
                )

            assert training_args.use_cpu is True
            assert training_args.bf16 is False
            assert training_args.fp16 is False
            assert str(training_args.device) == "cpu"

    def test_build_writes_bound_manifest_and_full_length_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_config(config)
            _save_source(source)

            result = BUILDER.build_with_length(config, source, output, "length", "input_ids", 2, 1)

            assert result["eligible"]
            manifest = json.loads((output / BUILDER.MANIFEST_NAME).read_text(encoding="utf-8"))
            assert manifest["schema_version"] == 2
            assert manifest["config_sha256"] == BUILDER._sha256(config)
            assert manifest["model"]["name_or_path"] == "example/model"
            assert manifest["model"]["revision"] == "pinned-revision"
            assert manifest["preprocessing"]["template"] == "qwen3_5_nothink"
            assert manifest["preprocessing"]["cutoff_len"] == 16
            assert not manifest["preprocessing"]["packing"]
            assert manifest["train_rows"] == 3
            assert manifest["stats"]["train"]["sum"] == 6
            assert manifest["length_validation"]["train"]["mismatch_count"] == 0
            assert manifest["builder"]["batch_size"] == 2
            assert manifest["builder"]["num_proc"] == 1
            assert manifest["builder"]["mode"] == "migrate_source_cache"
            assert "features" in manifest["dataset_schema"]["train"]
            assert not any(output.parent.glob(f".{output.name}.building-*"))
            assert manifest["cache_metadata_evidence"] is None

            check = BUILDER.check_cache(
                output,
                "length",
                config_path=config,
                require_manifest=True,
                full_length_validation=True,
            )
            assert check["eligible"], check
            assert check["contract_verified"]

            config.write_text(config.read_text(encoding="utf-8").replace("cutoff_len: 16", "cutoff_len: 15"))
            drift = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)
            assert not drift["eligible"]
            assert drift["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
            assert "config_sha256" in drift["contract_mismatches"]
            assert "preprocessing" in drift["contract_mismatches"]

    def test_builder_script_drift_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_config(config)
            _save_source(source)

            with mock.patch.object(BUILDER, "_builder_script_sha256", side_effect=["a" * 64, "b" * 64]):
                with self.assertRaisesRegex(RuntimeError, "builder script changed during construction"):
                    BUILDER.build_with_length(config, source, output, "length", "input_ids", 2, 1)
            assert not output.exists()
            assert not any(output.parent.glob(f".{output.name}.building-*"))

    def test_pt_exp2_binds_canonical_manifest_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            canonical_path = _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                manifest = json.loads((output / BUILDER.MANIFEST_NAME).read_text(encoding="utf-8"))
                assert manifest["canonical_dataset"]["manifest"] == str(canonical_path.resolve())
                assert manifest["canonical_dataset"]["manifest_sha256"] == BUILDER._sha256(canonical_path)
                assert manifest["canonical_dataset"]["dataset_sha256"] == _ordered_sha(["a", "b", "c"])
                assert manifest["model"]["snapshot"]["files"]["tokenizer.json"]
                live_metadata = BUILDER._cache_metadata_evidence(
                    BUILDER._cache_datasets(output), output, reject_symlinks=True
                )
                assert manifest["cache_metadata_evidence"] == live_metadata
                strict = BUILDER.check_cache(
                    output,
                    "length",
                    input_column="input_ids",
                    config_path=config,
                    require_manifest=True,
                    full_length_validation=True,
                )
                assert strict["eligible"], strict
                eval_path = data / "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
                assert manifest["eval_dataset"]["path"] == str(eval_path.resolve())
                assert manifest["eval_dataset"]["rows"] == BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS
                assert manifest["eval_dataset"]["sha256"] == BUILDER._sha256(eval_path)
                assert manifest["eval_dataset"]["manifest_contract"]["not_val511_overfit"] is True

                canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
                canonical["test_note"] = "manifest drift"
                _write_json(canonical_path, canonical)
                drift = BUILDER.check_cache(output, "length", config_path=config)
            assert not drift["eligible"]
            assert drift["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
            assert "canonical_dataset" in drift["contract_mismatches"]

    def test_refresh_manifest_repairs_only_known_persisted_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                manifest_path = output / BUILDER.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                live_metadata = BUILDER._cache_metadata_evidence(
                    BUILDER._cache_datasets(output), output, reject_symlinks=True
                )
                arrow_evidence = BUILDER._cache_data_files_evidence(output, reject_symlinks=True)
                manifest["cache_metadata_evidence"]["fingerprints"] = dict(
                    BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS
                )
                manifest["builder"]["script_sha256"] = BUILDER.PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256
                _write_json(manifest_path, manifest)
                original_manifest_sha256 = BUILDER._sha256(manifest_path)

                with (
                    mock.patch.object(
                        BUILDER, "PT_EXP2_LEGACY_MANIFEST_SHA256", original_manifest_sha256
                    ),
                    mock.patch.object(
                        BUILDER,
                        "PT_EXP2_STABLE_CACHE_FINGERPRINTS",
                        dict(live_metadata["fingerprints"]),
                    ),
                ):
                    result = BUILDER.refresh_manifest(config, output, "length", "input_ids")
                    assert result["eligible"]
                    assert result["changed"]
                    assert result["repaired_fields"] == [
                        "builder.script_sha256",
                        "cache_metadata_evidence",
                    ]
                    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
                    assert refreshed["cache_metadata_evidence"] == live_metadata
                    assert refreshed["builder"]["script_sha256"] == (
                        BUILDER.PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256
                    )
                    assert refreshed["repair"] == {
                        "kind": BUILDER.PT_EXP2_MANIFEST_REPAIR_KIND,
                        "schema_version": BUILDER.PT_EXP2_MANIFEST_REPAIR_SCHEMA_VERSION,
                        "original_manifest_sha256": original_manifest_sha256,
                        "old_fingerprints": BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS,
                        "new_fingerprints": live_metadata["fingerprints"],
                        "repaired_fields": BUILDER.PT_EXP2_MANIFEST_REPAIRED_FIELDS,
                        "repair_script_sha256": BUILDER._builder_script_sha256(),
                        "repaired_at": refreshed["repair"]["repaired_at"],
                    }
                    assert BUILDER._is_valid_pt_exp2_manifest_repair(
                        refreshed["repair"], BUILDER._builder_script_sha256()
                    )
                    assert BUILDER._cache_data_files_evidence(
                        output, reject_symlinks=True
                    ) == arrow_evidence
                    strict = BUILDER.check_cache(
                        output,
                        "length",
                        config_path=config,
                        require_manifest=True,
                        full_length_validation=True,
                    )
                    assert strict["eligible"], strict
                    no_op = BUILDER.refresh_manifest(config, output, "length", "input_ids")
                    assert no_op["eligible"]
                    assert not no_op["changed"]

    def test_known_pt_exp2_fingerprint_drift_requires_exact_live_identity(self) -> None:
        common = {
            "rows": {"train": 3, "validation": 1000},
            "schema": {"train": {"columns": ["input_ids"]}},
            "metadata_files": [],
            "metadata_set_sha256": "metadata-set",
        }
        recorded = {
            **common,
            "fingerprints": dict(BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS),
        }
        stable = {
            **common,
            "fingerprints": dict(BUILDER.PT_EXP2_STABLE_CACHE_FINGERPRINTS),
        }
        assert BUILDER._known_pt_exp2_metadata_fingerprint_drift(recorded, stable)

        unknown_live = {
            **common,
            "fingerprints": {
                "train": "unobserved-train",
                "validation": BUILDER.PT_EXP2_STABLE_CACHE_FINGERPRINTS["validation"],
            },
        }
        assert not BUILDER._known_pt_exp2_metadata_fingerprint_drift(recorded, unknown_live)

        recorded_new = {
            **common,
            "fingerprints": dict(BUILDER.PT_EXP2_STABLE_CACHE_FINGERPRINTS),
        }
        assert not BUILDER._known_pt_exp2_metadata_fingerprint_drift(recorded_new, stable)

    def test_refresh_manifest_lock_contention_prevents_second_refresh_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with _prepared_legacy_pt_cache(Path(directory)) as fixture:
                config, output, manifest_path, original_bytes, live_metadata = fixture
                entered_locked_refresh = threading.Event()
                release_locked_refresh = threading.Event()
                first_result: list[dict[str, object]] = []
                first_errors: list[BaseException] = []
                real_locked_refresh = BUILDER._refresh_manifest_locked

                def pause_with_lock(*args: object, **kwargs: object) -> dict[str, object]:
                    entered_locked_refresh.set()
                    if not release_locked_refresh.wait(timeout=10):
                        raise RuntimeError("test timed out while holding refresh lock")
                    return real_locked_refresh(*args, **kwargs)

                def run_first_refresh() -> None:
                    try:
                        first_result.append(
                            BUILDER.refresh_manifest(config, output, "length", "input_ids")
                        )
                    except BaseException as exc:
                        first_errors.append(exc)

                with mock.patch.object(
                    BUILDER,
                    "_refresh_manifest_locked",
                    side_effect=pause_with_lock,
                ):
                    first = threading.Thread(target=run_first_refresh)
                    first.start()
                    assert entered_locked_refresh.wait(timeout=10)
                    try:
                        with self.assertRaisesRegex(RuntimeError, "refresh lock is already held"):
                            BUILDER.refresh_manifest(config, output, "length", "input_ids")
                        assert manifest_path.read_bytes() == original_bytes
                    finally:
                        release_locked_refresh.set()
                        first.join(timeout=10)

                assert not first.is_alive()
                assert not first_errors
                assert first_result and first_result[0]["changed"] is True

                lock_path = output / BUILDER.MANIFEST_REFRESH_LOCK_NAME
                assert lock_path.is_file()
                assert not lock_path.is_symlink()
                # The stable lock lives at cache root but is intentionally
                # outside both Arrow and Dataset-metadata evidence sets.
                assert BUILDER._cache_metadata_evidence(
                    BUILDER._cache_datasets(output), output, reject_symlinks=True
                ) == live_metadata

    def test_repaired_manifest_rejects_unreviewed_origin_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with _prepared_legacy_pt_cache(Path(directory)) as fixture:
                config, output, manifest_path, original_bytes, live_metadata = fixture
                repaired = BUILDER.refresh_manifest(config, output, "length", "input_ids")
                assert repaired["eligible"], repaired
                pristine = json.loads(manifest_path.read_text(encoding="utf-8"))
                assert BUILDER._reconstruct_pt_exp2_legacy_manifest_bytes(
                    pristine, live_metadata
                ) == original_bytes

                mutations = {
                    "extra top-level field": lambda value: value.__setitem__("unreviewed", True),
                    "top-level action": lambda value: value.__setitem__("action", "forged"),
                    "source provenance": lambda value: value.__setitem__("source_cache", "/forged"),
                    "nested builder metadata": lambda value: value["builder"].__setitem__(
                        "mode", "forged"
                    ),
                }
                for label, mutate in mutations.items():
                    with self.subTest(label=label):
                        candidate = json.loads(json.dumps(pristine))
                        mutate(candidate)
                        _write_json(manifest_path, candidate)
                        result = BUILDER.check_cache(
                            output,
                            "length",
                            config_path=config,
                            require_manifest=True,
                            full_length_validation=True,
                        )
                        assert not result["eligible"]
                        assert result["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
                        assert "repair" in result["contract_mismatches"]

    def test_refresh_manifest_rolls_back_under_same_lock_after_failed_postflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with _prepared_legacy_pt_cache(Path(directory)) as fixture:
                config, output, manifest_path, original_bytes, _ = fixture
                real_check_cache = BUILDER.check_cache
                calls = 0

                def fail_postflight(*args: object, **kwargs: object) -> dict[str, object]:
                    nonlocal calls
                    calls += 1
                    result = real_check_cache(*args, **kwargs)
                    if calls == 2:
                        result = dict(result)
                        result["eligible"] = False
                        result["error"] = "INJECTED_POSTFLIGHT_FAILURE"
                    return result

                with mock.patch.object(BUILDER, "check_cache", side_effect=fail_postflight):
                    with self.assertRaisesRegex(RuntimeError, "failed strict refresh postflight"):
                        BUILDER.refresh_manifest(config, output, "length", "input_ids")

                assert calls == 2
                assert manifest_path.read_bytes() == original_bytes
                # A completed error path must release the persistent lock so a
                # later repair can acquire the same stable inode.
                with BUILDER._ManifestRefreshLock(output):
                    pass

    def test_refresh_manifest_rejects_unknown_manifest_drift_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                manifest_path = output / BUILDER.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["cache_metadata_evidence"]["fingerprints"] = {
                    "train": "not-the-reviewed-fingerprint",
                    "validation": "not-the-reviewed-fingerprint",
                }
                manifest["builder"]["script_sha256"] = BUILDER.PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256
                _write_json(manifest_path, manifest)
                before = manifest_path.read_bytes()

                with self.assertRaisesRegex(RuntimeError, "strict refresh preflight"):
                    BUILDER.refresh_manifest(config, output, "length", "input_ids")
            assert manifest_path.read_bytes() == before

    def test_refresh_manifest_rejects_missing_or_forged_repair_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                manifest_path = output / BUILDER.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                live_metadata = BUILDER._cache_metadata_evidence(
                    BUILDER._cache_datasets(output), output, reject_symlinks=True
                )
                manifest["cache_metadata_evidence"]["fingerprints"] = dict(
                    BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS
                )
                manifest["builder"]["script_sha256"] = BUILDER.PT_EXP2_LEGACY_BUILDER_SCRIPT_SHA256
                _write_json(manifest_path, manifest)
                original_manifest_sha256 = BUILDER._sha256(manifest_path)

                with (
                    mock.patch.object(
                        BUILDER, "PT_EXP2_LEGACY_MANIFEST_SHA256", original_manifest_sha256
                    ),
                    mock.patch.object(
                        BUILDER,
                        "PT_EXP2_STABLE_CACHE_FINGERPRINTS",
                        dict(live_metadata["fingerprints"]),
                    ),
                ):
                    missing = BUILDER.check_cache(
                        output, "length", config_path=config, require_manifest=True
                    )
                    assert not missing["eligible"]
                    assert "builder.script_sha256" in missing["contract_mismatches"]

                    manifest_without_repair = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_without_repair["repair"] = {"kind": BUILDER.PT_EXP2_MANIFEST_REPAIR_KIND}
                    _write_json(manifest_path, manifest_without_repair)
                    before = manifest_path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, "strict refresh preflight"):
                        BUILDER.refresh_manifest(config, output, "length", "input_ids")
                    assert manifest_path.read_bytes() == before

                    forged = json.loads(manifest_path.read_text(encoding="utf-8"))
                    forged["repair"] = {
                        "kind": BUILDER.PT_EXP2_MANIFEST_REPAIR_KIND,
                        "schema_version": BUILDER.PT_EXP2_MANIFEST_REPAIR_SCHEMA_VERSION,
                        "original_manifest_sha256": original_manifest_sha256,
                        "old_fingerprints": BUILDER.PT_EXP2_LEGACY_CACHE_FINGERPRINTS,
                        "new_fingerprints": {
                            **live_metadata["fingerprints"],
                            "train": "unobserved-train",
                        },
                        "repaired_fields": BUILDER.PT_EXP2_MANIFEST_REPAIRED_FIELDS,
                        "repair_script_sha256": BUILDER._builder_script_sha256(),
                        "repaired_at": "2026-01-01T00:00:00+00:00",
                    }
                    _write_json(manifest_path, forged)
                    before = manifest_path.read_bytes()
                    with self.assertRaisesRegex(RuntimeError, "strict refresh preflight"):
                        BUILDER.refresh_manifest(config, output, "length", "input_ids")
                    assert manifest_path.read_bytes() == before

    def test_pt_exp2_eval_payload_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                eval_path = data / "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
                eval_path.write_bytes(eval_path.read_bytes().replace(b'"val-0"', b'"drift"', 1))
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"].startswith("CACHE_CONTRACT_INPUT_INVALID: ValueError:")
            assert "output_sha256 does not match" in result["error"]

    def test_pt_exp2_eval_sidecar_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                sidecar_path = data / "bricknet_pt_exp2/val/PT-exp2-text-val1000.manifest.json"
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                sidecar["source_sha256"] = "sidecar-declared-source-drift"
                _write_json(sidecar_path, sidecar)
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
            assert "eval_dataset" in result["contract_mismatches"]

    def test_pt_exp2_requires_the_pinned_eval_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            _write_pt_fixture(data)
            _write_config(
                config,
                dataset=BUILDER.PT_EXP2_DATASET,
                eval_dataset="wrong-eval",
                dataset_dir=data,
            )
            config.write_text(
                config.read_text(encoding="utf-8")
                .replace("example/model", BUILDER.PT_EXP2_MODEL)
                .replace("pinned-revision", BUILDER.PT_EXP2_MODEL_REVISION),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "eval_dataset must be exactly"):
                BUILDER._input_contract(config, BUILDER._load_config(config))

    def test_pt_exp2_cache_requires_exact_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                with self.assertRaisesRegex(ValueError, "must contain exactly one validation split"):
                    BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)

            assert not output.exists()
            assert not any(output.parent.glob(f".{output.name}.building-*"))

    def test_pt_exp2_cache_requires_1000_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=999)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                with self.assertRaisesRegex(ValueError, "has 999 rows, expected 1000"):
                    BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)

            assert not output.exists()
            assert not any(output.parent.glob(f".{output.name}.building-*"))

    def test_pt_exp2_eval_split_length_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                original = BUILDER._cache_datasets(output)
                validation = original["validation"]
                tampered_validation = Dataset.from_dict(
                    {
                        "input_ids": validation["input_ids"],
                        "attention_mask": validation["attention_mask"],
                        "length": [999] * len(validation),
                    }
                )
                replacement = root / "replacement"
                DatasetDict(
                    {"train": original["train"], "validation": tampered_validation}
                ).save_to_disk(str(replacement))
                (replacement / BUILDER.MANIFEST_NAME).write_bytes(
                    (output / BUILDER.MANIFEST_NAME).read_bytes()
                )
                shutil.rmtree(output)
                replacement.rename(output)
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "LENGTH_VALUE_MISMATCH"
            assert result["length_validation"]["validation"]["mismatch_count"] == 1000

    def test_pt_exp2_eval_registry_path_rejects_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            _write_pt_fixture(data)
            _pt_config(config, data)

            registry_path = data / "dataset_info.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry[BUILDER.PT_EXP2_EVAL_DATASET]["file_name"] = "../outside.jsonl"
            _write_json(registry_path, registry)
            with self.assertRaisesRegex(ValueError, "file_name is unsafe"):
                BUILDER._input_contract(config, BUILDER._load_config(config))

            _write_json(
                registry_path,
                {
                    BUILDER.PT_EXP2_DATASET: {"file_name": "bricknet_pt_exp2/text8m_train"},
                    BUILDER.PT_EXP2_EVAL_DATASET: {
                        "file_name": "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
                    },
                },
            )
            eval_path = data / "bricknet_pt_exp2/val/PT-exp2-text-val1000.jsonl"
            backup = root / "eval-backup.jsonl"
            eval_path.rename(backup)
            eval_path.symlink_to(backup)
            with self.assertRaisesRegex(ValueError, "traverses a symlink"):
                BUILDER._input_contract(config, BUILDER._load_config(config))

    def test_pt_exp2_row_mismatch_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 4),
            ):
                with self.assertRaisesRegex(ValueError, "canonical row count"):
                    BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
            assert not output.exists()
            assert not any(output.parent.glob(f".{output.name}.building-*"))

    def test_pt_exp2_tampered_shard_is_rejected_by_live_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            canonical_path = _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                shard = canonical_path.parent / "part-00000.jsonl"
                shard.write_bytes(shard.read_bytes() + b'{"text":"tampered"}\n')
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "CACHE_CONTRACT_INPUT_INVALID: ValueError: PT-exp2 shard row count drift: " + str(shard)

    def test_pt_exp2_tampered_length_is_rejected_even_without_full_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                tampered = Dataset.from_dict(
                    {
                        "input_ids": [[1], [2, 3, 4], [5, 6]],
                        "attention_mask": [[1], [1, 1, 1], [1, 1]],
                        "length": [999, 3, 2],
                    }
                )
                original = BUILDER._cache_datasets(output)
                replacement = root / "replacement"
                DatasetDict({"train": tampered, "validation": original["validation"]}).save_to_disk(
                    str(replacement)
                )
                (replacement / BUILDER.MANIFEST_NAME).write_bytes(
                    (output / BUILDER.MANIFEST_NAME).read_bytes()
                )
                shutil.rmtree(output)
                replacement.rename(output)
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "LENGTH_VALUE_MISMATCH"
            assert result["length_validation"]["train"]["mismatch_count"] == 1

    def test_pt_exp2_tampered_arrow_payload_is_rejected_when_lengths_still_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                manifest = (output / BUILDER.MANIFEST_NAME).read_bytes()
                tampered = Dataset.from_dict(
                    {
                        "input_ids": [[999], [2, 3, 4], [5, 6]],
                        "attention_mask": [[1], [1, 1, 1], [1, 1]],
                        "length": [1, 3, 2],
                    }
                )
                original = BUILDER._cache_datasets(output)
                replacement = root / "replacement"
                DatasetDict({"train": tampered, "validation": original["validation"]}).save_to_disk(
                    str(replacement)
                )
                (replacement / BUILDER.MANIFEST_NAME).write_bytes(manifest)
                shutil.rmtree(output)
                replacement.rename(output)
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
            assert "cache_data_files" in result["contract_mismatches"]

    def test_pt_exp2_strict_cache_rejects_root_and_evidence_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root),
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)

                alias = root / "cache-alias"
                alias.symlink_to(output, target_is_directory=True)
                root_result = BUILDER.check_cache(alias, "length", config_path=config, require_manifest=True)
                alias.unlink()

                output_alias = root / "output-alias"
                output_alias.symlink_to(output, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "output must not be a symlink"):
                    BUILDER.build_with_length(config, None, output_alias, "length", "input_ids", 2, 1)
                output_alias.unlink()

                arrow = next(output.rglob("*.arrow"))
                arrow_backup = root / "payload.arrow"
                arrow.rename(arrow_backup)
                arrow.symlink_to(arrow_backup)
                arrow_result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)
                arrow.unlink()
                arrow_backup.rename(arrow)

                metadata = next(output.rglob("dataset_info.json"))
                metadata_backup = root / "dataset_info.json"
                metadata.rename(metadata_backup)
                metadata.symlink_to(metadata_backup)
                metadata_result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not root_result["eligible"]
            assert root_result["error"] == "CACHE_ROOT_SYMLINK"
            assert not arrow_result["eligible"]
            assert arrow_result["error"].startswith("CACHE_EVIDENCE_FAILED:")
            assert "Arrow file is a symlink" in arrow_result["error"]
            assert not metadata_result["eligible"]
            assert metadata_result["error"].startswith("CACHE_EVIDENCE_FAILED:")
            assert "metadata file is a symlink" in metadata_result["error"]

    def test_pt_exp2_tokenizer_snapshot_drift_is_rejected_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data)
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                mock.patch.object(
                    BUILDER, "_build_base_cache", side_effect=lambda _config, path: shutil.copytree(source, path)
                ),
                _fake_pt_snapshot(root) as snapshot,
            ):
                BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)
                (snapshot / "tokenizer.json").write_bytes(b"tokenizer drift")
                result = BUILDER.check_cache(output, "length", config_path=config, require_manifest=True)

            assert not result["eligible"]
            assert result["error"].startswith("CACHE_CONTRACT_INPUT_INVALID: ValueError:")
            assert "tokenizer.json" in result["error"]

    def test_pt_exp2_incomplete_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = root / "train.yaml"
            source = root / "source"
            output = root / "cache"
            _write_pt_fixture(data, audit={"completed": False})
            _pt_config(config, data)
            _save_source(source, eval_rows=BUILDER.PT_EXP2_EVAL_EXPECTED_ROWS)

            with (
                mock.patch.object(BUILDER, "PT_EXP2_EXPECTED_ROWS", 3),
                _fake_pt_snapshot(root),
            ):
                with self.assertRaisesRegex(ValueError, "audit is not completed"):
                    BUILDER.build_with_length(config, None, output, "length", "input_ids", 2, 1)

    def test_require_manifest_without_config_rejects_legacy_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "cache"
            # A lightweight cache fixture is enough for the no-config gate;
            # the dataset itself is not part of this manifest-schema test.
            Dataset.from_dict({"input_ids": [[1]], "length": [1]}).save_to_disk(str(output))
            manifest_path = output / BUILDER.MANIFEST_NAME
            _write_json(manifest_path, {"schema_version": 1, "kind": "legacy"})

            result = BUILDER.check_cache(output, "length", require_manifest=True)

            assert not result["eligible"]
            assert result["error"] == "CACHE_MANIFEST_CONTRACT_MISMATCH"
            assert result["contract_mismatches"] == ["schema_version", "kind"]


if __name__ == "__main__":
    unittest.main()
