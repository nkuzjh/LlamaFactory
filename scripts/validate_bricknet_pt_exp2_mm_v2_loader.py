#!/usr/bin/env python3
"""Validate PT-exp2 MM v2 with LlamaFactory's real SFT data pipeline.

The validator fully materializes each configured JSONL through the same
``get_dataset`` path used by training, then preprocesses a small prefix with
the real Qwen multimodal processor.  It never loads model weights, creates an
adapter, writes a tokenized dataset, or starts an optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams.parser import _parse_train_args
from llamafactory.model import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "data/bricknet_pt_exp2_mm_v2"
DEFAULT_OUTPUT = V2_ROOT / "loader_validation_report.json"
DEFAULT_CONFIGS = [
    ROOT / f"examples/train_lora/qwen35_08b_bricknet_pt_exp2_mm_{epoch}_v2.yaml"
    for epoch in ("e1", "e2", "e3")
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(config_path: Path, max_samples: int) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.update(
        {
            "max_samples": max_samples,
            "preprocessing_num_workers": min(4, int(raw.get("preprocessing_num_workers", 1))),
            "dataloader_num_workers": 0,
            "tokenized_path": None,
            "output_dir": str(ROOT / ".llamafactory_cache/validation-only/pt-exp2-mm-v2"),
            "report_to": "none",
            "plot_loss": False,
        }
    )
    # The public get_train_args intentionally rejects direct library calls that
    # are not launched through llamafactory-cli/torchrun.  This validator never
    # trains, so use the same typed parser and call the real get_dataset path
    # without initializing a distributed runtime.
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(raw)
    training_args.remove_unused_columns = False
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    train_dataset = dataset_module.get("train_dataset")
    if train_dataset is None or len(train_dataset) != max_samples:
        raise ValueError(f"{config_path}: preprocessed rows do not match max_samples={max_samples}")
    columns = list(train_dataset.column_names)
    required = {"input_ids", "attention_mask", "labels"}
    if not required.issubset(columns):
        raise ValueError(f"{config_path}: missing tokenized columns {sorted(required - set(columns))}")
    first = train_dataset[0]
    if not first["input_ids"] or len(first["input_ids"]) != len(first["attention_mask"]):
        raise ValueError(f"{config_path}: invalid first tokenized sample")
    return {
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "dataset": data_args.dataset,
        "dataset_dir": data_args.dataset_dir,
        "media_dir": data_args.media_dir,
        "raw_full_materialization": True,
        "preprocessed_rows": len(train_dataset),
        "columns": columns,
        "first_input_tokens": len(first["input_ids"]),
        "first_supervised_tokens": sum(label != -100 for label in first["labels"]),
        "processor": type(tokenizer_module["processor"]).__name__,
        "eligible": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    configs = [path.expanduser().resolve() for path in (args.config or DEFAULT_CONFIGS)]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} exists; validation evidence is never overwritten")
    report = {
        "schema_version": 1,
        "validation": "LlamaFactory get_dataset SFT loader smoke",
        "created_at": datetime.now(UTC).isoformat(),
        "max_samples_per_dataset": args.max_samples,
        "model_weights_loaded": False,
        "optimizer_started": False,
        "v2_manifest": str((V2_ROOT / "manifest.v2.json").resolve()),
        "v2_manifest_sha256": _sha256(V2_ROOT / "manifest.v2.json"),
        "results": [_validate(path, args.max_samples) for path in configs],
    }
    report["eligible"] = all(item["eligible"] for item in report["results"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"{temporary} exists; review incomplete validation evidence")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
