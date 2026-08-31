"""CPU-only tests for the immutable PT-exp2 row-balanced data builder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_bricknet_pt_exp2_mm_rowbal_cont3.py"
SPEC = importlib.util.spec_from_file_location("rowbal_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_rows(path: Path, rows: list[dict], *, array: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array:
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )


def _row(sample_id: str, *, multimodal: bool, with_meta: bool = True) -> dict:
    row = {
        "id": sample_id,
        "images": [f"images/{sample_id}.png"] if multimodal else [],
        "messages": [
            {"role": "system", "content": "You are a LEGO planner."},
            {
                "role": "user",
                "content": "<image>\nInventory:" if multimodal else "Build it.",
            },
            {"role": "assistant", "content": "a Brick 1 x 1 | Red"},
        ],
    }
    if with_meta:
        row["meta"] = {"source": sample_id, "variable": len(sample_id)}
    return row


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stable_selection_matches_frozen_rank_rule() -> None:
    ids = ["text-0", "text-1", "text-2", "text-3", "text-4"]
    expected = sorted(ids, key=lambda sample_id: (builder._stable_rank(sample_id, 42), sample_id))[:3]
    assert builder.select_text_ids(ids, count=3, seed=42) == expected


def test_projection_removes_meta_and_preserves_loader_key_order(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    row = _row("mm-0", multimodal=True)
    projected = builder._project_row(row, path=source, line_no=1, modality="multimodal")
    assert list(projected) == ["id", "images", "messages"]
    assert "meta" not in projected
    assert json.loads(builder._canonical_bytes(projected)) == projected


def test_text_projection_rejects_media_and_selection_rejects_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "text.jsonl"
    bad = _row("text-0", multimodal=True)
    with pytest.raises(ValueError, match=r"text row must have images=\[\]"):
        builder._project_row(bad, path=source, line_no=1, modality="text")
    with pytest.raises(ValueError, match="unique"):
        builder.select_text_ids(["text-0", "text-0"], count=1)


def test_small_fixture_build_verify_roundtrip_is_single_file_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mm_source = tmp_path / "BrickNet-MM_PT.json"
    text_source = tmp_path / "BrickNet-PT_text.jsonl"
    output = tmp_path / "bricknet_pt_exp2_mm_rowbal_cont3"
    mm_rows = [_row("mm-0", multimodal=True), _row("mm-1", multimodal=True)]
    text_rows = [_row("text-0", multimodal=False), _row("text-1", multimodal=False), _row("text-2", multimodal=False)]
    _write_rows(mm_source, mm_rows, array=True)
    _write_rows(text_source, text_rows, array=False)

    arrow_report = {
        "eligible": True,
        "rows": 4,
        "columns": ["id", "images", "messages"],
        "fixture": True,
    }
    old_report = {"eligible": True, "available": True, "fixture": True, "epochs": {}}
    monkeypatch.setattr(builder, "_arrow_materialize", lambda *args, **kwargs: dict(arrow_report))
    monkeypatch.setattr(builder, "_old_replay_coverage", lambda *args, **kwargs: dict(old_report))
    args = SimpleNamespace(
        mm_source=mm_source,
        text_source=text_source,
        old_root=tmp_path / "unused-old-root",
        output=output,
        expected_mm_rows=2,
        expected_text_rows=3,
        expected_mm_sha256=_sha256(mm_source),
        expected_text_sha256=_sha256(text_source),
        skip_token_audit=True,
        local_files_only=True,
        token_batch_size=8,
    )

    manifest = builder.build(args)
    assert manifest["eligible"] is True
    assert manifest["rows"] == 4
    assert manifest["multimodal_rows"] == 2
    assert manifest["text_rows"] == 2
    assert manifest["training_reuse"]["num_epochs"] == 3
    assert manifest["training_reuse"]["same_dataset_each_epoch"] is True
    assert manifest["selection"]["selected_rows"] == 2
    assert [path.name for path in output.glob("*.jsonl") if path.name == builder.OUTPUT_FILE_NAME] == [
        builder.OUTPUT_FILE_NAME
    ]
    assert (output / builder.MANIFEST_NAME).is_file()
    assert (output / "dataset_info.json").is_file()
    assert (output / builder.SELECTED_TEXT_IDS_FILE).is_file()
    rows = [json.loads(line) for line in (output / builder.OUTPUT_FILE_NAME).read_text().splitlines()]
    assert all(list(row) == ["id", "images", "messages"] for row in rows)
    assert all("meta" not in row for row in rows)
    assert [len(row["images"]) for row in rows] == [1, 1, 0, 0]
    assert manifest["schema"]["top_level_key_signatures"] == {"id,images,messages": 4}
    assert manifest["schema"]["message_role_sequences"] == {"system,user,assistant": 4}

    verified = builder.verify_existing(args)
    assert verified["sha256"] == manifest["sha256"]
    assert verified["ordered_id_sha256"] == manifest["ordered_id_sha256"]
    with pytest.raises(FileExistsError):
        builder.build(args)


def test_scan_output_rejects_extra_top_level_field(tmp_path: Path) -> None:
    path = tmp_path / "output.jsonl"
    row = _row("mm-0", multimodal=True, with_meta=False)
    row["extra"] = "must not reach the view"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output fields"):
        builder._scan_output(path, expected_rows=1, expected_mm_rows=1)
