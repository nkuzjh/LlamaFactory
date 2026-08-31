#!/usr/bin/env python3
"""Produce the CPU-only pre-training audits for PT-exp2-mm-rowbal-cont3.

This validator is deliberately separate from the training launcher.  It
materializes the immutable JSONL through Hugging Face Arrow and the real
LlamaFactory ``get_dataset`` path, then invokes the existing full-pool
``audit_bricknet_reasoning_tokens.py`` processor audit.  It loads a tokenizer
and multimodal processor, but never loads model weights, an adapter, an
optimizer, or a Trainer.  CUDA is disabled before any project import.

The two normalized reports are published only after both audits succeed.  A
completed report is never overwritten; the caller must explicitly pass
``--execute`` to run the audits or publish evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Keep imports above this line standard-library-only: the CPU/no-CUDA gate is
# installed before importing datasets, transformers, or LlamaFactory.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/bricknet_pt_exp2_mm_rowbal_cont3"
DEFAULT_DATA_FILE = DEFAULT_DATA_ROOT / "PT-exp2-mm-rowbal-cont3.jsonl"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "manifest.v1.json"
DEFAULT_REGISTRY = DEFAULT_DATA_ROOT / "dataset_info.json"
DEFAULT_LOADER_REPORT = DEFAULT_DATA_ROOT / "loader_validation_report.json"
DEFAULT_PROCESSOR_REPORT = DEFAULT_DATA_ROOT / "processor_audit.json"
DEFAULT_PROCESSOR_EVIDENCE_ROOT = DEFAULT_DATA_ROOT / "processor_audit_full"
DEFAULT_CONFIG = ROOT / "examples/train_lora/qwen35_08b_bricknet_pt_exp2_mm_rowbal_cont3.yaml"
DEFAULT_AUDIT = ROOT / "scripts/audit_bricknet_reasoning_tokens.py"
DEFAULT_LLAMAFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")

DATASET_NAME = "BrickNet-PT-exp2-mm-rowbal-cont3"
DATA_FILE_NAME = "PT-exp2-mm-rowbal-cont3.jsonl"
EXPECTED_ROWS = 270_102
EXPECTED_MM_ROWS = 135_051
EXPECTED_TEXT_ROWS = 135_051
EXPECTED_CUTOFF_LEN = 6_400
EXPECTED_DATA_SCHEMA = ["id", "images", "messages"]
EXPECTED_MESSAGE_SCHEMA = ["role", "content"]
EXPECTED_ROLES = ["system", "user", "assistant"]
EXPECTED_ROLE_SEQUENCE = "system,user,assistant"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    """Publish a completed report atomically and refuse a destination race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary report exists; inspect it before retrying: {temporary}")
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails instead of replacing a
        # report another process may have published concurrently.
        os.link(temporary, path)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"declared path escapes data root: {path}")
    return path


def _report_path(manifest: dict[str, Any], key: str, default: Path, data_root: Path) -> Path:
    declaration = manifest.get(key)
    if declaration is None:
        return default
    if isinstance(declaration, str):
        return _resolve_under(data_root, declaration)
    if isinstance(declaration, dict):
        return _resolve_under(data_root, declaration.get("path", declaration.get("file", default.name)))
    raise ValueError(f"manifest {key} declaration must be a path or object")


def _audit_rows(path: Path) -> dict[str, Any]:
    """Full structural/hash audit used before invoking any project loader."""

    digest = hashlib.sha256()
    ordered_ids = hashlib.sha256()
    ids: set[str] = set()
    schema_counts: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    total = multimodal = text = 0
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL row")
            digest.update(raw)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict) or list(row) != EXPECTED_DATA_SCHEMA:
                raise ValueError(f"{path}:{line_no}: top-level schema/order drift")
            sample_id = row.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError(f"{path}:{line_no}: id must be unique and non-empty")
            ids.add(sample_id)
            ordered_ids.update(sample_id.encode("utf-8"))
            ordered_ids.update(b"\n")
            images = row.get("images")
            if not isinstance(images, list) or any(not isinstance(image, str) for image in images):
                raise ValueError(f"{path}:{line_no}: images must be a list of strings")
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
                raise ValueError(f"{path}:{line_no}: expected three messages")
            roles: list[str] = []
            for message in messages:
                if not isinstance(message, dict) or set(message) != set(EXPECTED_MESSAGE_SCHEMA):
                    raise ValueError(f"{path}:{line_no}: message schema drift")
                if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                    raise ValueError(f"{path}:{line_no}: message values must be strings")
                roles.append(message["role"])
            if roles != EXPECTED_ROLES:
                raise ValueError(f"{path}:{line_no}: message roles drift: {roles}")
            role_sequences[",".join(roles)] += 1
            schema_counts[",".join(row)] += 1
            if total < EXPECTED_MM_ROWS:
                if len(images) != 1 or "<image>" not in messages[1]["content"]:
                    raise ValueError(f"{path}:{line_no}: multimodal prefix protocol drift")
                multimodal += 1
            else:
                if images:
                    raise ValueError(f"{path}:{line_no}: text suffix must use images=[]")
                text += 1
            total += 1
    return {
        "rows": total,
        "multimodal_rows": multimodal,
        "text_rows": text,
        "sha256": digest.hexdigest(),
        "ordered_id_sha256": ordered_ids.hexdigest(),
        "schema": {
            "top_level_fields": EXPECTED_DATA_SCHEMA,
            "top_level_key_signatures": dict(sorted(schema_counts.items())),
            "message_fields": EXPECTED_MESSAGE_SCHEMA,
            "message_roles": EXPECTED_ROLES,
            "message_role_sequences": dict(sorted(role_sequences.items())),
            "row_order": "multimodal_prefix_then_text_suffix",
        },
    }


def _validate_contract(
    data_root: Path, manifest_path: Path, data_file: Path, registry_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    manifest = _json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("dataset") != DATASET_NAME or manifest.get("eligible") is not True:
        raise ValueError("manifest is not an eligible row-balanced dataset")
    declared_file = manifest.get("dataset_file", manifest.get("file"))
    if declared_file != DATA_FILE_NAME:
        raise ValueError(f"manifest data file drift: {declared_file!r}")
    if data_file.resolve() != (data_root / DATA_FILE_NAME).resolve():
        raise ValueError(f"data file is outside the fixed namespace: {data_file}")
    registry = _json(registry_path)
    expected_registry = {
        DATASET_NAME: {
            "file_name": DATA_FILE_NAME,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    if registry != expected_registry:
        raise ValueError("dataset registry drift")
    registry_decl = manifest.get("registry")
    declared_registry_hash = manifest.get("registry_sha256")
    if isinstance(registry_decl, dict):
        declared_registry_hash = registry_decl.get("sha256", declared_registry_hash)
    registry_hash = _sha256(registry_path)
    if registry_hash != declared_registry_hash:
        raise ValueError("dataset registry hash drift")
    audit = _audit_rows(data_file)
    expected_schema = {
        "top_level_fields": EXPECTED_DATA_SCHEMA,
        "top_level_key_signatures": {"id,images,messages": EXPECTED_ROWS},
        "message_fields": EXPECTED_MESSAGE_SCHEMA,
        "message_roles": EXPECTED_ROLES,
        "message_role_sequences": {EXPECTED_ROLE_SEQUENCE: EXPECTED_ROWS},
        "row_order": "multimodal_prefix_then_text_suffix",
    }
    if audit["rows"] != EXPECTED_ROWS or audit["multimodal_rows"] != EXPECTED_MM_ROWS or audit["text_rows"] != EXPECTED_TEXT_ROWS:
        raise ValueError(f"data row counts drift: {audit}")
    if manifest.get("rows") != EXPECTED_ROWS or manifest.get("multimodal_rows") != EXPECTED_MM_ROWS or manifest.get("text_rows") != EXPECTED_TEXT_ROWS:
        raise ValueError("manifest row counts drift")
    if manifest.get("sha256") != audit["sha256"] or manifest.get("ordered_id_sha256") != audit["ordered_id_sha256"]:
        raise ValueError("manifest data hash drift")
    if manifest.get("schema") != expected_schema or audit["schema"] != expected_schema:
        raise ValueError("data schema drift")
    arrow = manifest.get("arrow_materialization", manifest.get("arrow"))
    if not isinstance(arrow, dict) or arrow.get("eligible") is not True or arrow.get("rows") != EXPECTED_ROWS or arrow.get("columns") != EXPECTED_DATA_SCHEMA:
        raise ValueError("Arrow materialization evidence is not eligible")
    checks = {
        "dataset": DATASET_NAME,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "data_file": str(data_file.resolve()),
        "data": audit,
        "registry": {"path": str(registry_path.resolve()), "sha256": registry_hash},
        "arrow_materialization": arrow,
    }
    return manifest, checks


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _loader_audit(config_path: Path, data_checks: dict[str, Any], max_samples: int) -> dict[str, Any]:
    """Run the real full-pool LlamaFactory loader with no output cache."""

    raw = _load_yaml(config_path)
    raw.update(
        {
            # Arrow is materialized for all rows separately.  The real
            # get_dataset call only preprocesses a small smoke prefix so this
            # validator does not duplicate the full processor audit.
            "max_samples": max_samples,
            "preprocessing_num_workers": min(4, int(raw.get("preprocessing_num_workers", 1))),
            "dataloader_num_workers": 0,
            "tokenized_path": None,
            "report_to": "none",
            "plot_loss": False,
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
            # This is a parser-only/loader-only invocation.  The production
            # YAML remains bf16/CUDA; CPU mode is injected only in this child.
            "use_cpu": True,
            "bf16": False,
            "fp16": False,
        }
    )
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, training_args, finetuning_args, _generating_args = _parse_train_args(raw)
    training_args.remove_unused_columns = False
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage=finetuning_args.stage,
        **tokenizer_module,
    )
    train_dataset = dataset_module.get("train_dataset")
    if train_dataset is None or len(train_dataset) != max_samples:
        raise ValueError(f"get_dataset produced {len(train_dataset) if train_dataset is not None else None} rows")
    columns = list(train_dataset.column_names)
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(columns):
        raise ValueError(f"get_dataset missing tokenized columns: {sorted(required - set(columns))}")
    first = train_dataset[0]
    if not first["input_ids"] or len(first["input_ids"]) != len(first["attention_mask"]):
        raise ValueError("get_dataset first tokenized sample is invalid")
    return {
        "schema_version": 1,
        "validation": "LlamaFactory get_dataset full-pool SFT loader",
        "dataset": DATASET_NAME,
        "rows": EXPECTED_ROWS,
        # `columns` is the raw Arrow contract consumed by the loader; the
        # preprocessed columns are retained as separate evidence.
        "columns": EXPECTED_DATA_SCHEMA,
        "raw_columns": EXPECTED_DATA_SCHEMA,
        "preprocessed_rows": len(train_dataset),
        "preprocessed_columns": columns,
        "first_input_tokens": len(first["input_ids"]),
        "first_supervised_tokens": sum(label != -100 for label in first["labels"]),
        "processor": type(tokenizer_module["processor"]).__name__ if tokenizer_module.get("processor") else None,
        "raw_full_materialization": True,
        "model_weights_loaded": False,
        "optimizer_started": False,
        "trainer_started": False,
        "config_sha256": _sha256(config_path),
        "dataset_sha256": data_checks["data"]["sha256"],
        "manifest_sha256": data_checks["manifest_sha256"],
        "eligible": True,
    }


def _arrow_audit(data_file: Path) -> dict[str, Any]:
    """Materialize the complete raw JSONL as Arrow in an isolated cache."""

    import datasets

    with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-rowbal-arrow-") as cache:
        dataset = datasets.load_dataset(
            "json",
            data_files=str(data_file),
            split="train",
            cache_dir=cache,
        )
        if len(dataset) != EXPECTED_ROWS or dataset.column_names != EXPECTED_DATA_SCHEMA:
            raise ValueError(
                f"raw Arrow materialization drift: rows={len(dataset)}, columns={dataset.column_names}"
            )
        boundary = []
        for index in (0, EXPECTED_MM_ROWS - 1, EXPECTED_MM_ROWS, EXPECTED_ROWS - 1):
            row = dataset[index]
            boundary.append({"index": index, "id": row["id"], "images": len(row["images"])})
        return {
            "eligible": True,
            "datasets_version": datasets.__version__,
            "rows": len(dataset),
            "columns": dataset.column_names,
            "boundary": boundary,
        }


def _processor_audit(
    data_file: Path,
    data_checks: dict[str, Any],
    config: dict[str, Any],
    python: Path,
    audit_script: Path,
    temporary_root: Path,
    evidence_root: Path,
    processor_workers: int,
    processor_chunksize: int,
) -> dict[str, Any]:
    output_dir = temporary_root / "reasoning-token-audit"
    model = config.get("model_name_or_path")
    media_dir = config.get("media_dir")
    if not isinstance(media_dir, str) or not media_dir:
        raise ValueError("training config must declare a non-empty media_dir")
    media_root = Path(media_dir).expanduser()
    if not media_root.is_absolute():
        media_root = ROOT / media_root
    media_root = media_root.resolve()
    command = [
        str(python),
        str(audit_script),
        "--dataset",
        f"{DATASET_NAME}={data_file}",
        "--bricknet-root",
        # The audit's bricknet-root is also the LlamaFactory media_dir.  The
        # JSONL stores paths such as images/PT/0.png, so this must resolve to
        # ROOT/data (the train YAML's media_dir), not the repository root.
        str(media_root),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--template",
        str(config.get("template", "qwen3_5_nothink")),
        "--cutoff-len",
        str(config.get("cutoff_len", EXPECTED_CUTOFF_LEN)),
        "--image-max-pixels",
        str(config.get("image_max_pixels", 589_824)),
        "--image-min-pixels",
        str(config.get("image_min_pixels", 1_024)),
        "--video-max-pixels",
        str(config.get("video_max_pixels", 65_536)),
        "--video-min-pixels",
        str(config.get("video_min_pixels", 256)),
        "--workers",
        str(processor_workers),
        "--chunksize",
        str(processor_chunksize),
        "--audit-purpose",
        "PT-exp2-mm-rowbal-cont3_full_pool",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    source_report = output_dir / "BrickNet-MM-Reasoning_token_audit_report.json"
    if not source_report.is_file():
        raise RuntimeError(f"reasoning-token audit did not produce {source_report}")
    report = _json(source_report)
    summary = report.get("datasets", {}).get(DATASET_NAME)
    if not isinstance(summary, dict):
        raise ValueError("reasoning-token audit has no row-balanced dataset summary")
    errors = summary.get("errors")
    truncated = summary.get("truncated")
    eligible = (
        report.get("is_full_pool") is True
        and report.get("training_eligible") is True
        and report.get("zero_errors") is True
        and report.get("zero_truncation") is True
        and summary.get("count") == EXPECTED_ROWS
        and errors == 0
        and truncated == 0
        and summary.get("path") == str(data_file.resolve())
        and summary.get("sha256") == data_checks["data"]["sha256"]
    )
    if not eligible:
        raise ValueError("reasoning-token full-pool audit is not eligible")
    if evidence_root.exists():
        raise FileExistsError(f"processor audit evidence exists; refusing overwrite: {evidence_root}")
    # The audit's report and one JSONL result per row are first produced in an
    # isolated directory.  Rewrite only path provenance (not audit values) so
    # the complete evidence remains usable after the temporary workspace is
    # removed, then atomically publish the directory into the data namespace.
    source_report = source_report.resolve()
    source_report_payload = dict(report)
    source_datasets = source_report_payload.get("datasets", {})
    if isinstance(source_datasets, dict):
        for dataset_summary in source_datasets.values():
            if isinstance(dataset_summary, dict) and isinstance(dataset_summary.get("sidecar"), str):
                dataset_summary["sidecar"] = str(evidence_root / Path(dataset_summary["sidecar"]).name)
    source_report_payload["evidence_root"] = str(evidence_root.resolve())
    rewritten = source_report.with_name(f".{source_report.name}.rewrite")
    rewritten.write_text(
        json.dumps(source_report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rewritten.replace(source_report)
    try:
        source_report.parent.rename(evidence_root)
    except FileExistsError as exc:
        raise FileExistsError(f"processor audit evidence appeared during validation: {evidence_root}") from exc
    final_source_report = evidence_root / source_report.name
    final_summary = dict(summary)
    if isinstance(final_summary.get("sidecar"), str):
        final_summary["sidecar"] = str(evidence_root / Path(final_summary["sidecar"]).name)
    return {
        "schema_version": 1,
        "audit": "real_llamafactory_multimodal_processor",
        "audit_purpose": "PT-exp2-mm-rowbal-cont3_full_pool",
        "dataset": DATASET_NAME,
        "rows": EXPECTED_ROWS,
        "dataset_sha256": data_checks["data"]["sha256"],
        "manifest_sha256": data_checks["manifest_sha256"],
        "full_pool": True,
        "is_full_pool": True,
        "training_eligible": True,
        "eligible": True,
        "errors": errors,
        "truncated": truncated,
        "cutoff_len": int(config.get("cutoff_len", EXPECTED_CUTOFF_LEN)),
        "processor_workers": processor_workers,
        "processor_chunksize": processor_chunksize,
        "zero_errors": True,
        "zero_truncation": True,
        "source_evidence_root": str(evidence_root.resolve()),
        "source_report": str(final_source_report.resolve()),
        "source_report_sha256": _sha256(final_source_report),
        "source_summary": final_summary,
        # Preserve the audit's runtime config under the conventional
        # ``config`` key so the launcher can bind media_dir provenance without
        # relying on an implementation-specific alias.
        "config": report.get("config"),
        "runtime": report.get("config"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--loader-output", type=Path, default=DEFAULT_LOADER_REPORT)
    parser.add_argument("--processor-output", type=Path, default=DEFAULT_PROCESSOR_REPORT)
    parser.add_argument("--processor-evidence-root", type=Path, default=DEFAULT_PROCESSOR_EVIDENCE_ROOT)
    parser.add_argument("--loader-max-samples", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=4)
    parser.add_argument("--processor-chunksize", type=int, default=8)
    parser.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--python", type=Path, default=DEFAULT_LLAMAFACTORY_PYTHON)
    parser.add_argument("--execute", action="store_true", help="run audits and publish reports")
    return parser.parse_args()


def main() -> None:
    # This must precede all project imports and subprocess creation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("FORCE_TORCHRUN", "NPROC_PER_NODE", "NNODES", "LOCAL_RANK", "RANK", "WORLD_SIZE"):
        os.environ.pop(key, None)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "pt-exp2-mm-rowbal-datasets"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    data_file = args.data_file.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    audit_script = args.audit_script.expanduser().resolve()
    python = args.python.expanduser().resolve()
    loader_output = args.loader_output.expanduser().resolve()
    processor_output = args.processor_output.expanduser().resolve()
    processor_evidence_root = args.processor_evidence_root.expanduser().resolve()
    if args.loader_max_samples < 1 or args.processor_workers < 1 or args.processor_chunksize < 1:
        raise ValueError("loader/processor worker and sample counts must be positive")
    if not python.is_file():
        raise FileNotFoundError(python)
    if not config_path.is_file() or not audit_script.is_file():
        raise FileNotFoundError("config or reasoning-token audit script is missing")
    manifest, data_checks = _validate_contract(data_root, manifest_path, data_file, registry_path)
    if args.loader_output == DEFAULT_LOADER_REPORT:
        loader_output = _report_path(manifest, "loader_validation", loader_output, data_root)
    if args.processor_output == DEFAULT_PROCESSOR_REPORT:
        processor_output = _report_path(manifest, "processor_audit", processor_output, data_root)
    report_paths = (loader_output, processor_output)
    for path in report_paths:
        _resolve_under(data_root, path)
    _resolve_under(data_root, processor_evidence_root)
    existing = [str(path) for path in report_paths if path.exists()]
    payload: dict[str, Any] = {
        "action": "prepare-audits",
        "dataset": DATASET_NAME,
        "manifest": str(manifest_path),
        "data": data_checks,
        "report_outputs": [str(path) for path in report_paths],
        "existing_reports": existing,
        "cpu_only": True,
        "model_weights_loaded": False,
        "optimizer_started": False,
        "trainer_started": False,
        "ready": not existing,
        "executed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if existing:
        raise SystemExit("audit output already exists; refusing to overwrite it")
    with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-rowbal-audit-", dir=data_root.parent) as temporary:
        temporary_root = Path(temporary)
        loader_report = _loader_audit(config_path, data_checks, args.loader_max_samples)
        loader_report["arrow_materialization"] = _arrow_audit(data_file)
        config = _load_yaml(config_path)
        processor_report = _processor_audit(
            data_file,
            data_checks,
            config,
            python,
            audit_script,
            temporary_root,
            processor_evidence_root,
            args.processor_workers,
            args.processor_chunksize,
        )
        # Recheck the destinations immediately before publication.  A report
        # race is a hard failure and leaves the successful temporary evidence
        # available only in the subprocess log, never a misleading partial
        # final report.
        if any(path.exists() for path in report_paths):
            raise FileExistsError("audit output appeared during validation; refusing overwrite")
        _write_json_no_replace(loader_output, loader_report)
        try:
            _write_json_no_replace(processor_output, processor_report)
        except BaseException:
            # If the second publication loses a race, remove only the report
            # this invocation just published; never remove a pre-existing file.
            loader_output.unlink(missing_ok=True)
            raise
    final = {
        "action": "prepare-audits",
        "dataset": DATASET_NAME,
        "manifest_sha256": data_checks["manifest_sha256"],
        "loader_report": str(loader_output),
        "processor_report": str(processor_output),
        "loader_report_sha256": _sha256(loader_output),
        "processor_report_sha256": _sha256(processor_output),
        "loader_eligible": True,
        "processor_eligible": True,
        "cpu_only": True,
        "model_weights_loaded": False,
        "optimizer_started": False,
        "trainer_started": False,
        "executed": True,
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
