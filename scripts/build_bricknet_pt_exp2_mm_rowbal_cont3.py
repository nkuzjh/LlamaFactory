#!/usr/bin/env python3
"""Build the immutable row-balanced ``PT-exp2-mm-rowbal-cont3`` view.

This command is data preparation only.  It does not import a model, start a
trainer, run inference, or use a GPU.  It materializes one 270,102-row
ShareGPT JSONL file containing all 135,051 multimodal PT rows followed by the
first 135,051 text rows under the frozen seed-42 SHA-256 ordering rule.  The
same file is intentionally consumed for three trainer epochs; e1/e2/e3 are
not three copies of the data.

The source rows contain provenance-only fields (including ``meta``).  Every
output row is a canonical projection with exactly ``id``, ``images`` and
``messages``.  Construction and promotion are fail-closed and atomic.  A
completed output is never overwritten, and an existing ``.building``
directory is preserved for diagnosis.

The normal build performs an exact CPU tokenizer target-token audit and a
full Hugging Face ``datasets.load_dataset`` Arrow materialization.  The
``--audit-only`` mode is an independent, read-only CPU audit for a completed
view; ``--skip-token-audit`` is explicit and leaves that audit as a separate
pre-training gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MM_SOURCE = ROOT / "data/BrickNet-MM_PT.json"
DEFAULT_TEXT_SOURCE = ROOT / "data/BrickNet-PT_text_270102_seed42.jsonl"
DEFAULT_OUTPUT = ROOT / "data/bricknet_pt_exp2_mm_rowbal_cont3"
DEFAULT_OLD_ROOT = ROOT / "data/bricknet_pt_exp2/mm"
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B"

EXPECTED_MM_ROWS = 135_051
EXPECTED_TEXT_ROWS = 270_102
EXPECTED_SELECTED_TEXT_ROWS = EXPECTED_MM_ROWS
EXPECTED_OUTPUT_ROWS = EXPECTED_MM_ROWS + EXPECTED_SELECTED_TEXT_ROWS
EXPECTED_MM_SHA256 = "9daa4703ae8e56afec862ce4fbe6cf9344422542b614c6327fcc09690eb4c055"
EXPECTED_TEXT_SHA256 = "f9b5e410abe1ec4e1aae436d51de12538433b1acb2a891a67eb97c6bd577dd48"
SEED = 42
NUM_EPOCHS = 3
EPOCHS = ("e1", "e2", "e3")
RETAINED_FIELDS = ("id", "images", "messages")
OUTPUT_FILE_NAME = "PT-exp2-mm-rowbal-cont3.jsonl"
SELECTED_TEXT_IDS_FILE = "selected_text_ids.jsonl"
DATASET_NAME = "BrickNet-PT-exp2-mm-rowbal-cont3"
MANIFEST_NAME = "manifest.v1.json"

# Historical v1 is read solely for the coverage audit.  It is never copied
# into the new data view.  These hashes are the currently frozen parent files.
OLD_MANIFEST_SHA256 = "678268372bd4fab6056490d97aa5cb0f4c121eb5ccd68ecc6eaa0db91b35d70e"
OLD_REPLAY_FILE_SHA256 = {
    "e1": "ce3a5185c0ffae9fdcf4b5fdb68cd6a61d00e3b69336ab5be0024c2e815c3928",
    "e2": "4c1df1d6f357b1f22b2ee918b45f8315f030a96d30ec7b07944b9d6c92d911a9",
    "e3": "acfea4ccb885376857ddeb0fba901b006a4d4f9aba75aac9bfb51f8179b769a4",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_bytes(row: dict[str, Any]) -> bytes:
    # The projection constructors insert both top-level fields and nested
    # message fields in the loader contract order.  Do not use sort_keys here:
    # LlamaFactory's strict audit also freezes message order as role/content.
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _json_object(raw: bytes, path: Path, line_no: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_no}: expected a JSON object")
    return value


def _validate_id(value: Any, path: Path, line_no: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_no}: id must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{path}:{line_no}: id must not contain a newline")
    return value


def _validate_messages(value: Any, path: Path, line_no: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_no}: messages must be a non-empty list")
    messages: list[dict[str, str]] = []
    for message_no, message in enumerate(value, 1):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"{path}:{line_no}: message {message_no} must contain exactly role/content"
            )
        role, content = message["role"], message["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{path}:{line_no}: message {message_no} values must be strings")
        messages.append({"role": role, "content": content})
    assistants = [message["content"] for message in messages if message["role"] == "assistant"]
    if len(assistants) != 1 or not assistants[0]:
        raise ValueError(f"{path}:{line_no}: expected exactly one non-empty assistant target")
    return messages


def _project_row(row: dict[str, Any], *, path: Path, line_no: int, modality: str) -> dict[str, Any]:
    """Validate one source row and return only the model-facing fields."""

    sample_id = _validate_id(row.get("id"), path, line_no)
    messages = _validate_messages(row.get("messages"), path, line_no)
    images_value = row.get("images", [])
    if not isinstance(images_value, list) or any(not isinstance(image, str) for image in images_value):
        raise ValueError(f"{path}:{line_no}: images must be a list of strings")
    if modality == "multimodal":
        if len(images_value) != 1:
            raise ValueError(f"{path}:{line_no}: multimodal row must contain exactly one image")
        images = list(images_value)
    elif modality == "text":
        if images_value:
            raise ValueError(f"{path}:{line_no}: text row must have images=[]")
        images = []
    else:  # pragma: no cover - internal misuse
        raise ValueError(f"unknown modality {modality!r}")
    return {"id": sample_id, "images": images, "messages": messages}


def _assistant_target(row: dict[str, Any], path: Path, line_no: int) -> str:
    assistants = [message["content"] for message in row["messages"] if message["role"] == "assistant"]
    if len(assistants) != 1 or not assistants[0]:
        raise ValueError(f"{path}:{line_no}: expected exactly one non-empty assistant target")
    text = assistants[0]
    return text if text.endswith("\n") else text + "\n"


def _stable_rank(sample_id: str, seed: int = SEED) -> str:
    """The frozen rank rule from build_bricknet_pt_exp2_mm.py."""

    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()


def select_text_ids(
    text_ids: Iterable[str], *, count: int = EXPECTED_SELECTED_TEXT_ROWS, seed: int = SEED
) -> list[str]:
    """Select exactly ``count`` unique IDs in stable rank order."""

    ids = list(text_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("text IDs must be unique before stable selection")
    if count < 1 or count > len(ids):
        raise ValueError(f"cannot select {count} IDs from {len(ids)} text rows")
    ordered = sorted(ids, key=lambda sample_id: (_stable_rank(sample_id, seed), sample_id))
    return ordered[:count]


def _top_level_signature(row: dict[str, Any]) -> str:
    return ",".join(sorted(row))


def _removed_source_fields(schema: Counter[str], retained: set[str]) -> list[str]:
    fields: set[str] = set()
    for signature in schema:
        fields.update(field for field in signature.split(",") if field)
    return sorted(fields - retained)


def _load_mm_rows(path: Path, expected_rows: int) -> tuple[list[dict[str, Any]], set[str], Counter[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON array: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array")
    if len(payload) != expected_rows:
        raise ValueError(f"{path}: row count {len(payload):,} != expected {expected_rows:,}")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    schema: Counter[str] = Counter()
    for row_no, row in enumerate(payload, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{row_no}: expected a JSON object")
        projected = _project_row(row, path=path, line_no=row_no, modality="multimodal")
        sample_id = projected["id"]
        if sample_id in ids:
            raise ValueError(f"{path}:{row_no}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        rows.append(row)
        schema[_top_level_signature(row)] += 1
    return rows, ids, schema


def _iter_text_objects(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for row_no, raw in enumerate(handle, 1):
            yield row_no, _json_object(raw, path, row_no)


def _load_text_ids(
    path: Path, *, expected_rows: int, mm_ids: set[str]
) -> tuple[list[str], Counter[str]]:
    ids: list[str] = []
    seen: set[str] = set()
    schema: Counter[str] = Counter()
    for row_no, row in _iter_text_objects(path):
        projected = _project_row(row, path=path, line_no=row_no, modality="text")
        sample_id = projected["id"]
        if sample_id in seen or sample_id in mm_ids:
            raise ValueError(f"{path}:{row_no}: duplicate or cross-modality id {sample_id!r}")
        seen.add(sample_id)
        ids.append(sample_id)
        schema[_top_level_signature(row)] += 1
    if len(ids) != expected_rows:
        raise ValueError(f"{path}: row count {len(ids):,} != expected {expected_rows:,}")
    return ids, schema


def _source_projection_iterator(
    mm_rows: list[dict[str, Any]], mm_source: Path, text_source: Path, selected_text_ids: set[str]
) -> Iterator[tuple[str, dict[str, Any], int]]:
    """Yield (modality, canonical row, source line) in frozen output order."""

    for row_no, row in enumerate(mm_rows, 1):
        yield "multimodal", _project_row(row, path=mm_source, line_no=row_no, modality="multimodal"), row_no
    seen_text: set[str] = set()
    selected_count = 0
    for row_no, row in _iter_text_objects(text_source):
        projected = _project_row(row, path=text_source, line_no=row_no, modality="text")
        sample_id = projected["id"]
        if sample_id in seen_text:
            raise ValueError(f"{text_source}:{row_no}: duplicate text id {sample_id!r}")
        seen_text.add(sample_id)
        if sample_id in selected_text_ids:
            selected_count += 1
            yield "text", projected, row_no
    if selected_count != len(selected_text_ids):
        raise ValueError(
            f"{text_source}: selected rows {selected_count:,} != expected {len(selected_text_ids):,}"
        )


def _tokenizer(name: str, revision: str | None, local_files_only: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "target-token audit requires transformers; use --skip-token-audit for a structural build"
        ) from exc
    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
        revision=revision,
        local_files_only=local_files_only,
    )


def _token_lengths(tokenizer: Any, texts: Iterable[str], batch_size: int) -> list[int]:
    values = list(texts)
    lengths: list[int] = []
    eos = tokenizer.eos_token or ""
    for start in range(0, len(values), batch_size):
        batch = [text + eos for text in values[start : start + batch_size]]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False, return_length=True)
        if "length" in encoded:
            lengths.extend(int(value) for value in encoded["length"])
        else:  # slow tokenizer fallback
            lengths.extend(len(value) for value in encoded["input_ids"])
    return lengths


def _mass_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("target-token audit received no rows")
    total = sum(lengths)
    return {
        "rows": len(lengths),
        "total_tokens": int(total),
        "min_tokens": int(min(lengths)),
        "max_tokens": int(max(lengths)),
        "mean_tokens": float(total / len(lengths)),
    }


def _audit_source_target_mass(
    mm_rows: list[dict[str, Any]],
    mm_source: Path,
    text_source: Path,
    selected_text_ids: set[str],
    tokenizer: Any,
    batch_size: int,
) -> dict[str, Any]:
    mm_texts = [
        _assistant_target(
            _project_row(row, path=mm_source, line_no=row_no, modality="multimodal"),
            mm_source,
            row_no,
        )
        for row_no, row in enumerate(mm_rows, 1)
    ]
    mm_lengths = _token_lengths(tokenizer, mm_texts, batch_size)
    selected_texts: list[str] = []
    for row_no, row in _iter_text_objects(text_source):
        projected = _project_row(row, path=text_source, line_no=row_no, modality="text")
        if projected["id"] in selected_text_ids:
            selected_texts.append(_assistant_target(projected, text_source, row_no))
    if len(selected_texts) != len(selected_text_ids):
        raise ValueError("selected text target-token audit count mismatch")
    text_lengths = _token_lengths(tokenizer, selected_texts, batch_size)
    mm_summary = _mass_summary(mm_lengths)
    text_summary = _mass_summary(text_lengths)
    mm_mass = mm_summary["total_tokens"]
    text_mass = text_summary["total_tokens"]
    return {
        "eligible": True,
        "method": "CPU tokenizer assistant target plus EOS; add_special_tokens=false; truncation=false",
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "batch_size": batch_size,
        "multimodal": mm_summary,
        "selected_text": text_summary,
        "text_to_multimodal_target_token_ratio": float(text_mass / mm_mass),
        "row_balance_ratio": float(len(selected_texts) / len(mm_rows)),
    }


def _schema_report(schema: Counter[str]) -> dict[str, Any]:
    return {
        "top_level_key_signatures": dict(sorted(schema.items())),
        "retained_fields": list(RETAINED_FIELDS),
        "output_message_fields": ["role", "content"],
    }


def _scan_output(path: Path, expected_rows: int, expected_mm_rows: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    semantic = hashlib.sha256()
    ordered_ids = hashlib.sha256()
    ids: set[str] = set()
    schema: Counter[str] = Counter()
    role_sequences: Counter[str] = Counter()
    modality = Counter()
    count = 0
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            row = _json_object(raw, path, line_no)
            if list(row) != ["id", "images", "messages"]:
                raise ValueError(f"{path}:{line_no}: output fields must be id/images/messages in that order")
            projected = _project_row(
                row,
                path=path,
                line_no=line_no,
                modality="multimodal" if count < expected_mm_rows else "text",
            )
            canonical = _canonical_bytes(projected)
            if raw != canonical:
                raise ValueError(f"{path}:{line_no}: non-canonical JSON serialization")
            sample_id = projected["id"]
            if sample_id in ids:
                raise ValueError(f"{path}:{line_no}: duplicate output id {sample_id!r}")
            ids.add(sample_id)
            digest.update(raw)
            semantic.update(canonical)
            ordered_ids.update(sample_id.encode("utf-8"))
            ordered_ids.update(b"\n")
            schema[_top_level_signature(row)] += 1
            role_sequences[",".join(message["role"] for message in projected["messages"])] += 1
            modality["multimodal" if projected["images"] else "text"] += 1
            count += 1
    if count != expected_rows:
        raise ValueError(f"{path}: row count {count:,} != expected {expected_rows:,}")
    return {
        "rows": count,
        "row_count": count,
        "unique_ids": len(ids),
        "unique_id_count": len(ids),
        "schema": {
            "top_level_fields": list(RETAINED_FIELDS),
            "top_level_key_signatures": dict(sorted(schema.items())),
            "message_fields": ["role", "content"],
            "message_roles": (
                next(iter(role_sequences)).split(",") if len(role_sequences) == 1 else []
            ),
            "message_role_sequences": dict(sorted(role_sequences.items())),
            "row_order": "multimodal_prefix_then_text_suffix",
        },
        "modality_counts": {
            "multimodal_rows": int(modality["multimodal"]),
            "text_rows": int(modality["text"]),
            "images_total": int(modality["multimodal"]),
        },
        "ordered_id_sha256": ordered_ids.hexdigest(),
        "semantic_projection_sha256": semantic.hexdigest(),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "_ids": ids,
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _registry() -> dict[str, Any]:
    return {
        DATASET_NAME: {
            "file_name": OUTPUT_FILE_NAME,
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


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _tracked_diff_sha256() -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _old_replay_coverage(
    old_root: Path,
    selected_text_ids: set[str],
    mm_ids: set[str],
    expected_mm_rows: int,
) -> dict[str, Any]:
    """Audit overlap with historical e1/e2/e3 replay IDs."""

    manifest_path = old_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"historical replay manifest missing: {manifest_path}")
    manifest_sha = _sha256_file(manifest_path)
    if old_root.resolve() == DEFAULT_OLD_ROOT.resolve() and manifest_sha != OLD_MANIFEST_SHA256:
        raise ValueError(f"historical replay manifest drift: {manifest_sha} != {OLD_MANIFEST_SHA256}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if manifest.get("eligible") is not True:
        raise ValueError("historical replay manifest is not eligible")

    epochs: dict[str, Any] = {}
    all_old_ids: set[str] = set()
    for epoch in EPOCHS:
        info = manifest.get("epochs", {}).get(epoch)
        if not isinstance(info, dict):
            raise ValueError(f"historical replay manifest missing {epoch}")
        path = old_root / str(info.get("file", f"PT-exp2-mm-{epoch}.jsonl"))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_file_hash = _sha256_file(path)
        declared_file_hash = info.get("sha256")
        if actual_file_hash != declared_file_hash:
            raise ValueError(f"{epoch}: historical replay file hash drift")
        if old_root.resolve() == DEFAULT_OLD_ROOT.resolve() and actual_file_hash != OLD_REPLAY_FILE_SHA256[epoch]:
            raise ValueError(f"{epoch}: historical replay file is not the frozen parent")
        expected_replay = int(info.get("text_replay_rows", -1))
        expected_rows = int(info.get("rows", -1))
        old_replay: list[str] = []
        old_seen: set[str] = set()
        with path.open("rb") as handle:
            for line_no, raw in enumerate(handle, 1):
                row = _json_object(raw, path, line_no)
                sample_id = _validate_id(row.get("id"), path, line_no)
                if sample_id in old_seen:
                    raise ValueError(f"{path}:{line_no}: duplicate historical id")
                old_seen.add(sample_id)
                if line_no > expected_mm_rows:
                    if row.get("images") != []:
                        raise ValueError(f"{path}:{line_no}: historical replay row must have images=[]")
                    old_replay.append(sample_id)
                elif sample_id not in mm_ids:
                    raise ValueError(f"{path}:{line_no}: historical MM prefix id is not in current MM source")
        if len(old_seen) != expected_rows or len(old_replay) != expected_replay:
            raise ValueError(f"{epoch}: historical replay row count mismatch")
        # The frozen v1 builder selected each replay slice in stable-rank
        # order, then wrote the selected rows by source order.  Its declared
        # replay-ID digest therefore describes the rank order rather than the
        # JSONL suffix order.  Keep both orders explicit in this audit so a
        # source-order rewrite cannot silently masquerade as the historical
        # selection order.
        ranked_old_replay = sorted(old_replay, key=lambda sample_id: (_stable_rank(sample_id), sample_id))
        declared_ids_hash = info.get("text_replay_ordered_id_sha256")
        actual_ids_hash = _sha256_lines(ranked_old_replay)
        if declared_ids_hash != actual_ids_hash:
            raise ValueError(f"{epoch}: historical replay ordered-ID hash drift")
        old_set = set(old_replay)
        if all_old_ids.intersection(old_set):
            raise ValueError("historical e1/e2/e3 replay sets overlap")
        all_old_ids.update(old_set)
        overlap = [sample_id for sample_id in old_replay if sample_id in selected_text_ids]
        ranked_overlap = sorted(overlap, key=lambda sample_id: (_stable_rank(sample_id), sample_id))
        epochs[epoch] = {
            "file": path.name,
            "file_sha256": actual_file_hash,
            "old_replay_rows": len(old_replay),
            "new_selected_text_rows": len(selected_text_ids),
            "overlap_rows": len(overlap),
            "old_replay_selected_coverage": float(len(overlap) / len(old_replay)),
            "new_selection_old_replay_coverage": float(len(overlap) / len(selected_text_ids)),
            "old_replay_ordered_id_sha256": actual_ids_hash,
            "replay_source_ordered_id_sha256": _sha256_lines(old_replay),
            "overlap_ordered_id_sha256": _sha256_lines(ranked_overlap),
            "overlap_source_ordered_id_sha256": _sha256_lines(overlap),
        }
    return {
        "eligible": True,
        "available": True,
        "root": str(old_root.resolve()),
        "manifest_sha256": manifest_sha,
        "selection_semantics": "historical e1/e2/e3 replay IDs compared with the new stable-rank selection",
        "epochs": epochs,
    }


def _arrow_materialize(
    path: Path,
    cache_dir: Path,
    expected_rows: int,
    expected_mm_rows: int = EXPECTED_MM_ROWS,
) -> dict[str, Any]:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Arrow gate requires the datasets package") from exc
    dataset = datasets.load_dataset(
        "json",
        data_files=str(path),
        split="train",
        cache_dir=str(cache_dir),
    )
    if len(dataset) != expected_rows:
        raise ValueError(f"{path}: Arrow rows {len(dataset)} != expected {expected_rows}")
    expected_columns = ["id", "images", "messages"]
    if dataset.column_names != expected_columns:
        raise ValueError(f"{path}: Arrow columns {dataset.column_names} != {expected_columns}")
    boundary_indices = sorted({0, expected_mm_rows - 1, expected_mm_rows, expected_rows - 1})
    boundary = []
    for index in boundary_indices:
        if 0 <= index < len(dataset):
            row = dataset[index]
            boundary.append({"index": index, "id": row["id"], "images": len(row["images"])})
    return {
        "eligible": True,
        "datasets_version": datasets.__version__,
        "rows": len(dataset),
        "columns": dataset.column_names,
        "features": repr(dataset.features),
        "boundary": boundary,
    }


def _validate_registry(output: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = output / "dataset_info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if registry != _registry():
        raise ValueError("row-balanced dataset registry does not match the frozen contract")
    actual_hash = _sha256_file(path)
    if expected is not None and actual_hash != expected.get("sha256"):
        raise ValueError("row-balanced dataset registry hash drift")
    registered = registry[DATASET_NAME]["file_name"]
    registered_path = output / registered
    if registered_path.resolve().parent != output.resolve():
        raise ValueError(f"registry path escapes output root: {registered}")
    if not registered_path.is_file():
        raise FileNotFoundError(registered_path)
    return {"path": str(path.resolve()), "sha256": actual_hash, "dataset_name": DATASET_NAME}


def _source_identity(
    mm_source: Path,
    text_source: Path,
    expected_mm_rows: int,
    expected_text_rows: int,
    expected_mm_sha256: str,
    expected_text_sha256: str,
) -> dict[str, Any]:
    for path in (mm_source, text_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_mm_hash = _sha256_file(mm_source)
    actual_text_hash = _sha256_file(text_source)
    if actual_mm_hash != expected_mm_sha256:
        raise ValueError(f"MM source hash drift: {actual_mm_hash} != {expected_mm_sha256}")
    if actual_text_hash != expected_text_sha256:
        raise ValueError(f"text source hash drift: {actual_text_hash} != {expected_text_sha256}")
    return {
        "multimodal": {
            "path": str(mm_source.resolve()),
            "rows": expected_mm_rows,
            "row_count": expected_mm_rows,
            "sha256": actual_mm_hash,
        },
        "text": {
            "path": str(text_source.resolve()),
            "rows": expected_text_rows,
            "row_count": expected_text_rows,
            "sha256": actual_text_hash,
        },
    }


def _args_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build(args: argparse.Namespace) -> dict[str, Any]:
    mm_source = Path(_args_value(args, "mm_source", DEFAULT_MM_SOURCE)).expanduser().resolve()
    text_source = Path(_args_value(args, "text_source", DEFAULT_TEXT_SOURCE)).expanduser().resolve()
    old_root = Path(_args_value(args, "old_root", DEFAULT_OLD_ROOT)).expanduser().resolve()
    output = Path(_args_value(args, "output", DEFAULT_OUTPUT)).expanduser().resolve()
    expected_mm_rows = int(_args_value(args, "expected_mm_rows", EXPECTED_MM_ROWS))
    expected_text_rows = int(_args_value(args, "expected_text_rows", EXPECTED_TEXT_ROWS))
    expected_mm_sha256 = str(_args_value(args, "expected_mm_sha256", EXPECTED_MM_SHA256))
    expected_text_sha256 = str(_args_value(args, "expected_text_sha256", EXPECTED_TEXT_SHA256))
    token_batch_size = int(_args_value(args, "token_batch_size", 512))
    if token_batch_size <= 0:
        raise ValueError("token_batch_size must be positive")
    if output.exists():
        raise FileExistsError(f"{output} exists; immutable row-balanced data are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"{temporary} exists; inspect incomplete evidence before retrying")

    source_info = _source_identity(
        mm_source,
        text_source,
        expected_mm_rows,
        expected_text_rows,
        expected_mm_sha256,
        expected_text_sha256,
    )
    mm_rows, mm_ids, mm_schema = _load_mm_rows(mm_source, expected_mm_rows)
    text_ids, text_schema = _load_text_ids(text_source, expected_rows=expected_text_rows, mm_ids=mm_ids)
    selected_rows = expected_mm_rows
    expected_rows = expected_mm_rows + selected_rows
    ranked_selected_ids = select_text_ids(text_ids, count=selected_rows, seed=SEED)
    selected_text_ids = set(ranked_selected_ids)
    if len(selected_text_ids) != selected_rows:
        raise ValueError("stable selection did not produce the expected unique row count")

    temporary.mkdir(parents=True)
    handle: Any | None = None
    started = datetime.now(UTC)
    try:
        selected_ids_path = temporary / SELECTED_TEXT_IDS_FILE
        _write_bytes(
            selected_ids_path,
            "".join(sample_id + "\n" for sample_id in ranked_selected_ids).encode("utf-8"),
        )
        output_path = temporary / OUTPUT_FILE_NAME
        handle = output_path.open("wb")
        digest = hashlib.sha256()
        selected_source_order: list[str] = []
        count = 0
        for modality, projected, _source_line in _source_projection_iterator(
            mm_rows, mm_source, text_source, selected_text_ids
        ):
            if modality == "text":
                selected_source_order.append(projected["id"])
            raw = _canonical_bytes(projected)
            handle.write(raw)
            digest.update(raw)
            count += 1
        if count != expected_rows:
            raise ValueError(f"output row count {count:,} != expected {expected_rows:,}")
        if len(selected_source_order) != selected_rows:
            raise ValueError("selected text source-order row count mismatch")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None

        # A source changing between passes must never be promoted.
        source_info_after = _source_identity(
            mm_source,
            text_source,
            expected_mm_rows,
            expected_text_rows,
            expected_mm_sha256,
            expected_text_sha256,
        )
        if source_info_after != source_info:
            raise ValueError("source identity changed during build")

        output_report = _scan_output(output_path, expected_rows, expected_mm_rows)
        if output_report["sha256"] != digest.hexdigest():
            raise ValueError("output digest changed during verification")
        if output_report["modality_counts"]["multimodal_rows"] != expected_mm_rows:
            raise ValueError("multimodal row count gate failed")
        if output_report["modality_counts"]["text_rows"] != selected_rows:
            raise ValueError("selected text row count gate failed")
        expected_schema = {
            "top_level_fields": list(RETAINED_FIELDS),
            "top_level_key_signatures": {"id,images,messages": expected_rows},
            "message_fields": ["role", "content"],
            "message_roles": ["system", "user", "assistant"],
            "message_role_sequences": {"system,user,assistant": expected_rows},
            "row_order": "multimodal_prefix_then_text_suffix",
        }
        if output_report["schema"] != expected_schema:
            raise ValueError(
                "output message/schema contract drift: "
                f"{output_report['schema']} != {expected_schema}"
            )
        output_report.pop("_ids", None)
        output_report["dataset"] = DATASET_NAME
        output_report["file"] = OUTPUT_FILE_NAME
        output_report["selected_text_source_order_sha256"] = _sha256_lines(selected_source_order)

        registry_path = temporary / "dataset_info.json"
        _write_bytes(
            registry_path,
            (json.dumps(_registry(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        registry = _validate_registry(temporary)
        # The manifest is promoted one directory later; bind the registry
        # path to that final immutable namespace, not the temporary builder.
        registry["path"] = str((output / "dataset_info.json").resolve())

        if _args_value(args, "skip_token_audit", False):
            token_audit: dict[str, Any] = {
                "eligible": False,
                "required": False,
                "status": "skipped_by_explicit_flag",
                "method": "independent CPU audit available via --audit-only",
                "audit_entrypoint": f"python {Path(__file__).resolve()} --audit-only --output {output}",
            }
        else:
            tokenizer = _tokenizer(
                str(_args_value(args, "tokenizer", DEFAULT_TOKENIZER)),
                _args_value(args, "revision", None),
                bool(_args_value(args, "local_files_only", False)),
            )
            token_audit = _audit_source_target_mass(
                mm_rows,
                mm_source,
                text_source,
                selected_text_ids,
                tokenizer,
                token_batch_size,
            )
            token_audit["required"] = True

        if _args_value(args, "skip_old_coverage", False):
            old_coverage: dict[str, Any] = {
                "eligible": False,
                "available": False,
                "status": "skipped_by_explicit_flag",
                "root": str(old_root),
            }
        else:
            old_coverage = _old_replay_coverage(old_root, selected_text_ids, mm_ids, expected_mm_rows)

        with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-rowbal-arrow-", dir=output.parent) as cache:
            output_report["arrow_materialization"] = _arrow_materialize(
                output_path, Path(cache), expected_rows, expected_mm_rows
            )

        selected_ids_hash = _sha256_file(selected_ids_path)
        manifest = {
            "schema_version": 1,
            "family": "PT-exp2-mm-rowbal-cont3",
            "dataset": DATASET_NAME,
            "dataset_file": OUTPUT_FILE_NAME,
            "file": OUTPUT_FILE_NAME,
            "rows": expected_rows,
            "multimodal_rows": expected_mm_rows,
            "text_rows": selected_rows,
            "sha256": output_report["sha256"],
            "ordered_id_sha256": output_report["ordered_id_sha256"],
            "schema": expected_schema,
            "arrow_materialization": output_report["arrow_materialization"],
            "registry_sha256": registry["sha256"],
            "experiment_id": "PT-exp2-mm-rowbal-cont3",
            "data_revision": "row-balanced-fixed-replay-cont3-no-meta",
            "status": "validated",
            "eligible": bool(old_coverage.get("eligible") is True),
            "created_at": started.isoformat(),
            "purpose": (
                "All 135051 MM PT rows plus 135051 fixed seed-42 text rows; one file reused for three trainer epochs"
            ),
            "training_reuse": {
                "same_dataset_each_epoch": True,
                "num_epochs": NUM_EPOCHS,
                "dataset_rows_per_epoch": expected_rows,
                "total_row_exposures": expected_rows * NUM_EPOCHS,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "global_batch_size": 16,
                "steps_per_epoch": 16_882,
                "expected_max_steps": 50_646,
            },
            "selection": {
                "seed": SEED,
                "rank": "sha256('42\\0'+id)",
                "tie_break": "id ascending",
                "rule": "sort all text IDs by (sha256('42\\0'+id), id), take first 135051",
                "selected_rows": selected_rows,
                "selected_text_ids_artifact": {
                    "file": SELECTED_TEXT_IDS_FILE,
                    "rows": selected_rows,
                    "sha256": selected_ids_hash,
                    "ordered_id_sha256": _sha256_lines(ranked_selected_ids),
                    "order": "stable rank order",
                },
                "selected_text_source_order_sha256": output_report["selected_text_source_order_sha256"],
                "replay_reuse": "the exact same selected IDs are reused in e1/e2/e3; no disjoint split",
            },
            "sources": {
                "multimodal": {
                    **source_info["multimodal"],
                    "unique_id_count": len(mm_ids),
                    "schema": _schema_report(mm_schema),
                    "removed_fields": _removed_source_fields(mm_schema, set(RETAINED_FIELDS)),
                },
                "text": {
                    **source_info["text"],
                    "unique_id_count": len(text_ids),
                    "schema": _schema_report(text_schema),
                    "removed_fields": _removed_source_fields(text_schema, {"id", "messages"}),
                },
            },
            "retained_fields": list(RETAINED_FIELDS),
            "removed_fields": sorted(
                set(_removed_source_fields(mm_schema, set(RETAINED_FIELDS)))
                | set(_removed_source_fields(text_schema, {"id", "messages"}))
            ),
            "training_columns": ["messages", "images"],
            "ordering": "all MM rows in source order, then selected text rows in source order",
            "output": output_report,
            "old_replay_coverage": old_coverage,
            "target_token_mass": token_audit,
            "registry": registry,
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "repository": {
                "root": str(ROOT),
                "head": _git_value("rev-parse", "HEAD"),
                "tracked_diff_sha256": _tracked_diff_sha256(),
                "status": (_git_value("status", "--short") or "").splitlines(),
            },
            "environment": {"python": sys.version, "platform": platform.platform()},
            "gates": {
                "source_hashes_match": True,
                "source_row_counts_match": True,
                "source_unique_ids": True,
                "stable_selection_count": selected_rows == expected_mm_rows,
                "fixed_selection_reused_for_three_epochs": True,
                "uniform_top_level_schema": True,
                "meta_removed": True,
                "row_balance": output_report["modality_counts"]["multimodal_rows"]
                == output_report["modality_counts"]["text_rows"],
                "target_token_audit": bool(token_audit.get("eligible")),
                "historical_replay_coverage": bool(old_coverage.get("eligible")),
                "full_arrow_materialization": bool(
                    output_report.get("arrow_materialization", {}).get("eligible")
                ),
                "registry_gate": True,
            },
        }
        manifest_path = temporary / MANIFEST_NAME
        _write_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        # Token auditing is optional because --audit-only is an independent
        # CPU gate; all structural/coverage/Arrow gates are mandatory.
        mandatory = (
            manifest["gates"]["source_hashes_match"]
            and manifest["gates"]["source_row_counts_match"]
            and manifest["gates"]["source_unique_ids"]
            and manifest["gates"]["stable_selection_count"]
            and manifest["gates"]["fixed_selection_reused_for_three_epochs"]
            and manifest["gates"]["uniform_top_level_schema"]
            and manifest["gates"]["meta_removed"]
            and manifest["gates"]["row_balance"]
            and manifest["gates"]["historical_replay_coverage"]
            and manifest["gates"]["full_arrow_materialization"]
            and manifest["gates"]["registry_gate"]
        )
        manifest["eligible"] = bool(mandatory)
        if not manifest["eligible"]:
            raise RuntimeError("row-balanced build failed one or more mandatory gates")
        # Rewrite the manifest after the final eligibility value is assigned.
        _write_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        try:
            os.rename(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"{output} appeared during build; refusing overwrite") from exc
        return manifest
    except BaseException:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        # Keep .building and all partial files as evidence.
        raise


def _load_manifest(output: Path) -> dict[str, Any]:
    path = output / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if manifest.get("family") != "PT-exp2-mm-rowbal-cont3" or manifest.get("eligible") is not True:
        raise ValueError("manifest is not an eligible PT-exp2-mm-rowbal-cont3 artifact")
    return manifest


def verify_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Verify an existing artifact without changing it."""

    output = Path(_args_value(args, "output", DEFAULT_OUTPUT)).expanduser().resolve()
    manifest = _load_manifest(output)
    mm_source = Path(_args_value(args, "mm_source", DEFAULT_MM_SOURCE)).expanduser().resolve()
    text_source = Path(_args_value(args, "text_source", DEFAULT_TEXT_SOURCE)).expanduser().resolve()
    old_root = Path(_args_value(args, "old_root", DEFAULT_OLD_ROOT)).expanduser().resolve()
    expected_mm_rows = int(_args_value(args, "expected_mm_rows", EXPECTED_MM_ROWS))
    expected_text_rows = int(_args_value(args, "expected_text_rows", EXPECTED_TEXT_ROWS))
    selected_rows = expected_mm_rows
    expected_rows = expected_mm_rows + selected_rows
    expected_mm_sha256 = str(_args_value(args, "expected_mm_sha256", EXPECTED_MM_SHA256))
    expected_text_sha256 = str(_args_value(args, "expected_text_sha256", EXPECTED_TEXT_SHA256))
    source_info = _source_identity(
        mm_source,
        text_source,
        expected_mm_rows,
        expected_text_rows,
        expected_mm_sha256,
        expected_text_sha256,
    )
    for source_name in ("multimodal", "text"):
        if manifest.get("sources", {}).get(source_name, {}).get("sha256") != source_info[source_name]["sha256"]:
            raise ValueError(f"manifest {source_name} source hash does not match current source")
    mm_rows, mm_ids, _mm_schema = _load_mm_rows(mm_source, expected_mm_rows)
    text_ids, _text_schema = _load_text_ids(text_source, expected_rows=expected_text_rows, mm_ids=mm_ids)
    ranked_selected_ids = select_text_ids(text_ids, count=selected_rows, seed=SEED)
    selected_text_ids = set(ranked_selected_ids)
    selection = manifest.get("selection", {})
    selected_artifact_info = selection.get("selected_text_ids_artifact", {})
    if selected_artifact_info.get("rows") != selected_rows:
        raise ValueError("manifest selected text row count drift")
    selected_artifact = output / str(selected_artifact_info.get("file", ""))
    if not selected_artifact.is_file():
        raise FileNotFoundError(selected_artifact)
    artifact_ids = selected_artifact.read_text(encoding="utf-8").splitlines()
    if artifact_ids != ranked_selected_ids:
        raise ValueError("selected_text_ids.jsonl drift from stable rank selection")
    if _sha256_file(selected_artifact) != selected_artifact_info.get("sha256"):
        raise ValueError("selected text IDs artifact hash drift")
    if selected_artifact_info.get("ordered_id_sha256") != _sha256_lines(ranked_selected_ids):
        raise ValueError("selected text ordered-ID hash drift")

    registry = _validate_registry(output, manifest.get("registry"))
    if registry != manifest.get("registry"):
        raise ValueError("manifest registry evidence drift")
    output_info = manifest.get("output")
    if not isinstance(output_info, dict) or output_info.get("file") != OUTPUT_FILE_NAME:
        raise ValueError("manifest output contract missing/drifted")
    output_path = output / OUTPUT_FILE_NAME
    report = _scan_output(output_path, expected_rows, expected_mm_rows)
    output_ids = report.pop("_ids")
    expected_schema = {
        "top_level_fields": list(RETAINED_FIELDS),
        "top_level_key_signatures": {"id,images,messages": expected_rows},
        "message_fields": ["role", "content"],
        "message_roles": ["system", "user", "assistant"],
        "message_role_sequences": {"system,user,assistant": expected_rows},
        "row_order": "multimodal_prefix_then_text_suffix",
    }
    if report.get("schema") != expected_schema or manifest.get("schema") != expected_schema:
        raise ValueError("manifest/output schema contract drift")
    if manifest.get("dataset") != DATASET_NAME or manifest.get("dataset_file") != OUTPUT_FILE_NAME:
        raise ValueError("manifest dataset/file contract drift")
    if (manifest.get("rows"), manifest.get("multimodal_rows"), manifest.get("text_rows")) != (
        expected_rows,
        expected_mm_rows,
        selected_rows,
    ):
        raise ValueError("manifest row-count contract drift")
    for key in (
        "sha256",
        "ordered_id_sha256",
        "semantic_projection_sha256",
        "rows",
        "unique_ids",
        "modality_counts",
    ):
        if report.get(key) != output_info.get(key):
            raise ValueError(f"manifest output {key} drift")
    if output_ids != set(mm_ids).union(selected_text_ids):
        raise ValueError("output IDs do not equal MM union selected text IDs")
    if report["modality_counts"]["multimodal_rows"] != expected_mm_rows:
        raise ValueError("output MM row count drift")
    if report["modality_counts"]["text_rows"] != selected_rows:
        raise ValueError("output selected text row count drift")

    expected_projection = hashlib.sha256()
    expected_ordered = hashlib.sha256()
    expected_source_order: list[str] = []
    expected_count = 0
    for modality, projected, _line_no in _source_projection_iterator(
        mm_rows, mm_source, text_source, selected_text_ids
    ):
        raw = _canonical_bytes(projected)
        expected_projection.update(raw)
        expected_ordered.update(projected["id"].encode("utf-8"))
        expected_ordered.update(b"\n")
        expected_count += 1
        if modality == "text":
            expected_source_order.append(projected["id"])
    if expected_count != expected_rows:
        raise ValueError("source projection row count mismatch")
    if expected_projection.hexdigest() != report["sha256"]:
        raise ValueError("output/source semantic projection digest mismatch")
    if expected_ordered.hexdigest() != report["ordered_id_sha256"]:
        raise ValueError("output/source ordered-ID digest mismatch")
    if manifest.get("sha256") != report["sha256"] or manifest.get("ordered_id_sha256") != report["ordered_id_sha256"]:
        raise ValueError("manifest top-level data hash drift")
    if selection.get("selected_text_source_order_sha256") != _sha256_lines(expected_source_order):
        raise ValueError("selected source-order ID hash drift")

    old_coverage = _old_replay_coverage(old_root, selected_text_ids, mm_ids, expected_mm_rows)
    if manifest.get("old_replay_coverage") != old_coverage:
        raise ValueError("historical replay coverage drift")

    token_audit = manifest.get("target_token_mass", {})
    if token_audit.get("eligible") is True:
        tokenizer = _tokenizer(
            str(_args_value(args, "tokenizer", DEFAULT_TOKENIZER)),
            _args_value(args, "revision", None),
            bool(_args_value(args, "local_files_only", False)),
        )
        actual = _audit_source_target_mass(
            mm_rows,
            mm_source,
            text_source,
            selected_text_ids,
            tokenizer,
            int(token_audit.get("batch_size", _args_value(args, "token_batch_size", 512))),
        )
        for key in (
            "multimodal",
            "selected_text",
            "text_to_multimodal_target_token_ratio",
            "row_balance_ratio",
        ):
            if actual.get(key) != token_audit.get(key):
                raise ValueError(f"target-token audit drift in {key}")

    with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-rowbal-verify-arrow-", dir=output.parent) as cache:
        arrow = _arrow_materialize(output_path, Path(cache), expected_rows, expected_mm_rows)
        if arrow != output_info.get("arrow_materialization"):
            raise ValueError("Arrow materialization evidence drift")
    return manifest


def audit_only(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only CPU target-token audit for a completed artifact."""

    output = Path(_args_value(args, "output", DEFAULT_OUTPUT)).expanduser().resolve()
    manifest = _load_manifest(output)
    tokenizer = _tokenizer(
        str(_args_value(args, "tokenizer", DEFAULT_TOKENIZER)),
        _args_value(args, "revision", None),
        bool(_args_value(args, "local_files_only", False)),
    )
    batch_size = int(_args_value(args, "token_batch_size", 512))
    if batch_size <= 0:
        raise ValueError("token_batch_size must be positive")
    path = output / OUTPUT_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    mm_texts: list[str] = []
    text_texts: list[str] = []
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            row = _json_object(raw, path, line_no)
            if list(row) != ["id", "images", "messages"]:
                raise ValueError(f"{path}:{line_no}: unexpected output schema")
            projected = _project_row(
                row,
                path=path,
                line_no=line_no,
                modality="multimodal" if len(mm_texts) < EXPECTED_MM_ROWS else "text",
            )
            target = _assistant_target(projected, path, line_no)
            (mm_texts if projected["images"] else text_texts).append(target)
    if len(mm_texts) != EXPECTED_MM_ROWS or len(text_texts) != EXPECTED_SELECTED_TEXT_ROWS:
        raise ValueError("row-balanced output modality counts are not 135051/135051")
    mm = _mass_summary(_token_lengths(tokenizer, mm_texts, batch_size))
    text = _mass_summary(_token_lengths(tokenizer, text_texts, batch_size))
    return {
        "schema_version": 1,
        "audit": "PT-exp2-mm-rowbal-cont3 independent CPU target-token mass",
        "manifest_sha256": _sha256_file(output / MANIFEST_NAME),
        "manifest_data_revision": manifest.get("data_revision"),
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "eos_token": tokenizer.eos_token,
        "batch_size": batch_size,
        "output": {
            "file": path.name,
            "sha256": _sha256_file(path),
            "multimodal": mm,
            "selected_text": text,
            "text_to_multimodal_target_token_ratio": float(text["total_tokens"] / mm["total_tokens"]),
            "row_balance_ratio": 1.0,
        },
        "eligible": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm-source", type=Path, default=DEFAULT_MM_SOURCE)
    parser.add_argument("--text-source", type=Path, default=DEFAULT_TEXT_SOURCE)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-mm-rows", type=int, default=EXPECTED_MM_ROWS)
    parser.add_argument("--expected-text-rows", type=int, default=EXPECTED_TEXT_ROWS)
    parser.add_argument("--expected-mm-sha256", default=EXPECTED_MM_SHA256)
    parser.add_argument("--expected-text-sha256", default=EXPECTED_TEXT_SHA256)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision")
    parser.add_argument("--token-batch-size", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-token-audit", action="store_true")
    parser.add_argument("--skip-old-coverage", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.token_batch_size <= 0:
        parser.error("--token-batch-size must be positive")
    if args.verify_only and args.audit_only:
        parser.error("--verify-only and --audit-only are mutually exclusive")
    if args.skip_token_audit and args.audit_only:
        parser.error("--audit-only cannot be combined with --skip-token-audit")
    return args


def main() -> None:
    args = parse_args()
    if args.audit_only:
        report = audit_only(args)
    elif args.verify_only:
        report = verify_existing(args)
    else:
        report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
