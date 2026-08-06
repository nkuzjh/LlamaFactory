#!/usr/bin/env python3
"""Build a held-out BrickNet-MM evaluation set that matches the MM-PT prompt protocol."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/BrickNet-MM_VAL.json"
DEFAULT_OUTPUT = ROOT / "data/BrickNet-MM_PT_VAL.json"
CAPTION_MARKER = "Caption:\n"
INVENTORY_MARKER = "\n\nInventory of parts:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clear_caption(content: str) -> tuple[str, str]:
    before, marker, remainder = content.partition(CAPTION_MARKER)
    if not marker:
        raise ValueError(f"missing prompt marker {CAPTION_MARKER!r}")

    caption, marker, after = remainder.partition(INVENTORY_MARKER)
    if not marker:
        raise ValueError(f"missing prompt marker {INVENTORY_MARKER!r}")

    return before + CAPTION_MARKER + INVENTORY_MARKER + after, caption.strip()


def convert_record(record: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    converted = deepcopy(record)
    messages = converted.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"row {index}: expected at least two messages")
    if messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
        raise ValueError(f"row {index}: expected a non-empty final assistant response")

    user_messages = [message for message in messages if message.get("role") == "user"]
    if len(user_messages) != 1 or not isinstance(user_messages[0].get("content"), str):
        raise ValueError(f"row {index}: expected exactly one textual user message")

    user_messages[0]["content"], caption = clear_caption(user_messages[0]["content"])
    images = converted.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"row {index}: expected at least one image")

    return converted, caption


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{input_path}: expected a non-empty JSON array")

    converted_records = []
    captions = []
    ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"row {index}: expected a JSON object")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError(f"row {index}: missing or duplicate id {sample_id!r}")
        ids.add(sample_id)
        converted, caption = convert_record(record, index)
        converted_records.append(converted)
        captions.append(caption)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    nonempty_captions = sum(bool(caption) for caption in captions)
    print(f"Wrote {len(converted_records):,} MM-PT evaluation rows to {output_path}")
    print(f"Cleared {nonempty_captions:,} non-empty captions; retained images, inventories, ids, and references.")


if __name__ == "__main__":
    main()
