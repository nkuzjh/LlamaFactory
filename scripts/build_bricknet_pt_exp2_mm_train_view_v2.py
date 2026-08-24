#!/usr/bin/env python3
"""Build the immutable PT-exp2 MM v2 training projection.

The frozen v1 JSONL files are retained as historical evidence.  This builder
copies only the fields consumed by LlamaFactory (plus ``id`` for auditing) into
an isolated dataset directory, so heterogeneous provenance-only ``meta``
structs never reach Hugging Face Arrow inference.

The build is fail-closed and atomic: parent hashes must match the frozen v1
manifest, an existing output is never overwritten, every projected row is
compared with its parent, and each completed JSONL must materialize through
``datasets.load_dataset`` before the directory is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT / "data/bricknet_pt_exp2/mm"
DEFAULT_OUTPUT = ROOT / "data/bricknet_pt_exp2_mm_v2"
V1_MANIFEST = V1_ROOT / "manifest.json"
EXPECTED_V1_MANIFEST_SHA256 = "678268372bd4fab6056490d97aa5cb0f4c121eb5ccd68ecc6eaa0db91b35d70e"
EXPECTED_PARENT_SHA256 = {
    "e1": "ce3a5185c0ffae9fdcf4b5fdb68cd6a61d00e3b69336ab5be0024c2e815c3928",
    "e2": "4c1df1d6f357b1f22b2ee918b45f8315f030a96d30ec7b07944b9d6c92d911a9",
    "e3": "acfea4ccb885376857ddeb0fba901b006a4d4f9aba75aac9bfb51f8179b769a4",
}
RETAINED_FIELDS = ("id", "messages", "images")
DATASET_NAMES = {epoch: f"BrickNet-PT-exp2-mm-{epoch}-v2" for epoch in ("e1", "e2", "e3")}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_projection(row: dict[str, Any], *, path: Path, line_no: int) -> dict[str, Any]:
    missing = [field for field in RETAINED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{path}:{line_no}: missing retained fields {missing}")
    sample_id = row["id"]
    messages = row["messages"]
    images = row["images"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{path}:{line_no}: id must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_no}: messages must be a non-empty list")
    for message_no, message in enumerate(messages, 1):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{path}:{line_no}: message {message_no} has an unexpected schema")
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise ValueError(f"{path}:{line_no}: message {message_no} values must be strings")
    if not isinstance(images, list) or any(not isinstance(image, str) for image in images):
        raise ValueError(f"{path}:{line_no}: images must be a list of strings")
    return {"id": sample_id, "messages": messages, "images": images}


def _canonical_bytes(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


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


def _existing_file_hashes(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = path.resolve()
        entry: dict[str, Any] = {"path": str(resolved), "exists": path.is_file()}
        if path.is_file():
            entry.update({"size": path.stat().st_size, "sha256": _sha256_file(path)})
        result[str(path.relative_to(ROOT))] = entry
    return result


def _registry() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for epoch, dataset_name in DATASET_NAMES.items():
        entries[dataset_name] = {
            "file_name": f"PT-exp2-mm-{epoch}-train-v2.jsonl",
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
    return entries


def _build_epoch(epoch: str, parent: Path, output: Path, epoch_manifest: dict[str, Any]) -> dict[str, Any]:
    expected_rows = int(epoch_manifest["rows"])
    expected_mm_rows = int(epoch_manifest["multimodal_rows"])
    expected_replay_rows = int(epoch_manifest["text_replay_rows"])
    if expected_mm_rows + expected_replay_rows != expected_rows:
        raise ValueError(f"{epoch}: inconsistent row counts in v1 manifest")

    ids: set[str] = set()
    ordered_ids = hashlib.sha256()
    semantic_parent = hashlib.sha256()
    meta_signatures: Counter[str] = Counter()
    count = 0
    mm_rows = 0
    replay_rows = 0
    with parent.open("rb") as source, output.open("wb") as target:
        for line_no, raw in enumerate(source, 1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{parent}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{parent}:{line_no}: expected an object")
            projected = _canonical_projection(row, path=parent, line_no=line_no)
            sample_id = projected["id"]
            if sample_id in ids:
                raise ValueError(f"{parent}:{line_no}: duplicate id {sample_id!r}")
            ids.add(sample_id)
            if line_no <= expected_mm_rows:
                if len(projected["images"]) != 1:
                    raise ValueError(f"{parent}:{line_no}: MM prefix row must contain exactly one image")
                mm_rows += 1
            else:
                if projected["images"]:
                    raise ValueError(f"{parent}:{line_no}: replay suffix row must contain images=[]")
                replay_rows += 1
            meta = row.get("meta")
            if isinstance(meta, dict):
                meta_signatures[",".join(sorted(meta))] += 1
            else:
                meta_signatures[f"<{type(meta).__name__}>"] += 1
            projected_raw = _canonical_bytes(projected)
            target.write(projected_raw)
            semantic_parent.update(projected_raw)
            ordered_ids.update(sample_id.encode("utf-8"))
            ordered_ids.update(b"\n")
            count += 1

    if count != expected_rows or mm_rows != expected_mm_rows or replay_rows != expected_replay_rows:
        raise ValueError(
            f"{epoch}: projected counts {(count, mm_rows, replay_rows)} do not match "
            f"{(expected_rows, expected_mm_rows, expected_replay_rows)}"
        )

    semantic_output = hashlib.sha256()
    output_ids = hashlib.sha256()
    output_count = 0
    with output.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            row = json.loads(raw)
            if list(row) != ["id", "images", "messages"]:
                # sort_keys=True intentionally fixes one uniform top-level schema/order.
                raise ValueError(f"{output}:{line_no}: unexpected serialized field order {list(row)}")
            semantic_output.update(raw)
            output_ids.update(row["id"].encode("utf-8"))
            output_ids.update(b"\n")
            output_count += 1
    if output_count != count:
        raise ValueError(f"{epoch}: output row count changed during verification")
    if semantic_output.hexdigest() != semantic_parent.hexdigest():
        raise ValueError(f"{epoch}: semantic projection digest mismatch")
    if output_ids.hexdigest() != ordered_ids.hexdigest():
        raise ValueError(f"{epoch}: ordered ID digest mismatch")

    return {
        "dataset": DATASET_NAMES[epoch],
        "file": output.name,
        "rows": count,
        "multimodal_rows": mm_rows,
        "text_replay_rows": replay_rows,
        "ordered_id_sha256": ordered_ids.hexdigest(),
        "semantic_projection_sha256": semantic_parent.hexdigest(),
        "sha256": _sha256_file(output),
        "size_bytes": output.stat().st_size,
        "parent": {
            "path": str(parent.resolve()),
            "sha256": _sha256_file(parent),
            "manifest_sha256": epoch_manifest["sha256"],
        },
        "parent_meta_signatures": dict(sorted(meta_signatures.items())),
        "projection_equivalent": True,
    }


def _arrow_materialize(path: Path, cache_dir: Path, expected_rows: int) -> dict[str, Any]:
    import datasets

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
    boundary_indices = sorted({0, 135_050, 135_051, len(dataset) - 1})
    boundary = []
    for index in boundary_indices:
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


def _validate_parent_manifest() -> dict[str, Any]:
    if not V1_MANIFEST.is_file():
        raise FileNotFoundError(V1_MANIFEST)
    actual_manifest_hash = _sha256_file(V1_MANIFEST)
    if actual_manifest_hash != EXPECTED_V1_MANIFEST_SHA256:
        raise ValueError(
            f"v1 manifest drift: {actual_manifest_hash} != {EXPECTED_V1_MANIFEST_SHA256}"
        )
    manifest = _json(V1_MANIFEST)
    if manifest.get("eligible") is not True:
        raise ValueError("v1 manifest is not eligible")
    for epoch, expected_hash in EXPECTED_PARENT_SHA256.items():
        epoch_manifest = manifest.get("epochs", {}).get(epoch, {})
        if epoch_manifest.get("sha256") != expected_hash:
            raise ValueError(f"{epoch}: v1 manifest hash declaration drift")
        parent = V1_ROOT / f"PT-exp2-mm-{epoch}.jsonl"
        actual_hash = _sha256_file(parent)
        if actual_hash != expected_hash:
            raise ValueError(f"{epoch}: v1 parent drift: {actual_hash} != {expected_hash}")
    return manifest


def _verify_existing(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.v2.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _json(manifest_path)
    if manifest.get("eligible") is not True:
        raise ValueError("v2 manifest is not eligible")
    _validate_parent_manifest()
    registry = output / "dataset_info.json"
    if _sha256_file(registry) != manifest["registry"]["sha256"]:
        raise ValueError("v2 dataset registry drift")
    for epoch, info in manifest["epochs"].items():
        dataset = output / info["file"]
        if _sha256_file(dataset) != info["sha256"]:
            raise ValueError(f"{epoch}: v2 dataset drift")
    return manifest


def build(output: Path) -> dict[str, Any]:
    parent_manifest = _validate_parent_manifest()
    if output.exists():
        raise FileExistsError(f"{output} exists; immutable v2 data are never overwritten")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"{temporary} exists; review it before any cleanup")
    temporary.mkdir(parents=True)

    try:
        epochs: dict[str, Any] = {}
        for epoch in ("e1", "e2", "e3"):
            parent = V1_ROOT / f"PT-exp2-mm-{epoch}.jsonl"
            target = temporary / f"PT-exp2-mm-{epoch}-train-v2.jsonl"
            epochs[epoch] = _build_epoch(epoch, parent, target, parent_manifest["epochs"][epoch])

        registry_path = temporary / "dataset_info.json"
        registry_path.write_text(
            json.dumps(_registry(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        with tempfile.TemporaryDirectory(prefix="pt-exp2-mm-v2-arrow-", dir=output.parent) as cache:
            cache_root = Path(cache)
            for epoch, info in epochs.items():
                info["arrow_materialization"] = _arrow_materialize(
                    temporary / info["file"], cache_root / epoch, info["rows"]
                )

        historical_paths = [
            V1_MANIFEST,
            ROOT / "data/dataset_info.json",
            ROOT / "scripts/launch_bricknet_pt_exp2.py",
            ROOT / "tmp_bash/run_pt_exp2_mm_e1_e2_e3.sh",
        ]
        historical_paths.extend(
            ROOT / f"examples/train_lora/qwen35_08b_bricknet_pt_exp2_mm_{epoch}.yaml"
            for epoch in ("e1", "e2", "e3")
        )
        historical_paths.extend(
            ROOT / f"examples/train_lora/qwen35_08b_bricknet_pt_exp2_mm_{epoch}_predict.yaml"
            for epoch in ("e1", "e2", "e3")
        )
        adapter_root = (
            ROOT
            / "saves/Qwen3.5-0.8B-Thinking/lora"
            / "train_PT_exp2_text8m_qwen35_08b_path7698261_steps250k_bs4_gbs32_lora64_len6401_nopack"
        )
        historical_paths.extend(
            adapter_root / name
            for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "train_results.json")
        )

        manifest = {
            "schema_version": 2,
            "family": "PT-exp2-mm-training-projection",
            "data_revision": "v2-no-meta",
            "status": "validated",
            "created_at": datetime.now(UTC).isoformat(),
            "purpose": (
                "Arrow-safe training view; preserve id/messages/images exactly and exclude provenance-only meta"
            ),
            "parent_manifest": {
                "path": str(V1_MANIFEST.resolve()),
                "sha256": EXPECTED_V1_MANIFEST_SHA256,
            },
            "retained_fields": list(RETAINED_FIELDS),
            "removed_fields": ["meta"],
            "training_columns": ["messages", "images"],
            "ordering": parent_manifest.get("ordering"),
            "selection": parent_manifest.get("selection"),
            "tokenizer": parent_manifest.get("tokenizer"),
            "sources": parent_manifest.get("sources"),
            "epochs": epochs,
            "registry": {
                "path": str((output / "dataset_info.json").resolve()),
                "sha256": _sha256_file(registry_path),
                "dataset_names": DATASET_NAMES,
            },
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__)),
            },
            "repository": {
                "root": str(ROOT),
                "head": _git_value("rev-parse", "HEAD"),
                "tracked_diff_sha256": _tracked_diff_sha256(),
                "status": (_git_value("status", "--short") or "").splitlines(),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "historical_entrypoints": _existing_file_hashes(historical_paths),
            "gates": {
                "parent_hashes_match": True,
                "row_counts_match": True,
                "ordered_ids_match": True,
                "semantic_projection_match": True,
                "uniform_top_level_schema": True,
                "full_arrow_materialization": True,
            },
            "eligible": True,
        }
        manifest_path = temporary / "manifest.v2.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        # Preserve the incomplete directory for diagnosis; never silently clean evidence.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing immutable v2 directory without changing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    manifest = _verify_existing(output) if args.verify_only else build(output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
