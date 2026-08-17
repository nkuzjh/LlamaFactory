#!/usr/bin/env python3
"""Build or migrate a LlamaFactory tokenized cache with a persisted length column.

The script supports PT and SFT YAML files.  If ``--source-cache`` is supplied,
it migrates that existing cache to the configured or explicitly selected output.
Otherwise it reuses the configured cache when present, or invokes LlamaFactory's
normal preprocessing pipeline when absent.
The final cache is written through a sibling temporary directory and renamed
only after schema, row-count, and sampled length checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "length_cache_manifest.json"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return config


def _cache_schema(path: Path) -> dict[str, list[str]]:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, Dataset):
        return {"train": dataset.column_names}
    return {split: dataset[split].column_names for split in dataset}


def check_cache(path: Path, length_column: str, input_column: str = "input_ids") -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "length_column": length_column,
        "input_column": input_column,
        "eligible": False,
    }
    if not path.is_dir():
        result["error"] = "CACHE_MISSING"
        return result

    try:
        schema = _cache_schema(path)
    except Exception as exc:
        result["error"] = f"CACHE_LOAD_FAILED: {exc}"
        return result

    result["schema"] = schema
    if "train" not in schema:
        result["error"] = "TRAIN_SPLIT_MISSING"
    elif input_column not in schema["train"]:
        result["error"] = "INPUT_COLUMN_MISSING"
    elif length_column not in schema["train"]:
        result["error"] = "LENGTH_COLUMN_MISSING"
    else:
        result["eligible"] = True
    return result


def _add_length(
    dataset: Dataset,
    length_column: str,
    input_column: str,
    batch_size: int,
    num_proc: int,
    cache_file: Path,
) -> Dataset:
    if input_column not in dataset.column_names:
        raise ValueError(f"input column {input_column!r} is missing from {dataset.column_names}")
    if length_column in dataset.column_names:
        return dataset

    features = Features({**dataset.features, length_column: Value("int32")})

    def compute_lengths(batch: dict[str, list[Any]]) -> dict[str, np.ndarray]:
        return {length_column: np.fromiter((len(ids) for ids in batch[input_column]), dtype=np.int32)}

    return dataset.map(
        compute_lengths,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        features=features,
        cache_file_name=str(cache_file),
        desc=f"Adding {length_column} column",
    )


def _length_stats(dataset: Dataset, length_column: str) -> dict[str, int | float]:
    column = dataset.data.column(length_column)
    bounds = pc.min_max(column).as_py()
    total = pc.sum(column).as_py()
    return {
        "rows": len(dataset),
        "min": int(bounds["min"]),
        "max": int(bounds["max"]),
        "mean": float(total / len(dataset)) if len(dataset) else 0.0,
    }


def _validate_lengths(dataset: Dataset, length_column: str, input_column: str) -> list[int]:
    if not len(dataset):
        return []
    indices = sorted({0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4, len(dataset) - 1})
    for index in indices:
        row = dataset[index]
        if int(row[length_column]) != len(row[input_column]):
            raise ValueError(f"length mismatch at row {index}")
    return indices


def _parse_build_args(config: dict[str, Any], output: Path) -> tuple[Any, Any, Any, Any]:
    from transformers import HfArgumentParser

    from llamafactory.hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
        TrainingArguments,
    )

    build_config = dict(config)
    build_config.update(
        {
            "tokenized_path": str(output),
            "do_train": False,
            "do_eval": False,
            "do_predict": False,
            "report_to": "none",
        }
    )
    # The project train parser intentionally rejects a non-distributed process.
    # Cache construction must remain single-process at the launcher level, so
    # parse the same dataclasses without applying train-launch-only validation.
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments)
    )
    model_args, data_args, training_args, finetuning_args, _ = parser.parse_dict(build_config)
    return model_args, data_args, training_args, finetuning_args


def _build_base_cache(config: dict[str, Any], output: Path) -> None:
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer

    model_args, data_args, training_args, finetuning_args = _parse_build_args(config, output)
    if finetuning_args.stage not in {"pt", "sft"}:
        raise ValueError(f"only PT and SFT are supported, got stage={finetuning_args.stage!r}")

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage=finetuning_args.stage,
        **tokenizer_module,
    )
    if not output.is_dir():
        raise RuntimeError(f"LlamaFactory did not create tokenized cache: {output}")


def build_with_length(
    config_path: Path,
    source: Path | None,
    output: Path,
    length_column: str,
    input_column: str,
    batch_size: int,
    num_proc: int,
) -> dict[str, Any]:
    if output.exists():
        check = check_cache(output, length_column, input_column)
        if check["eligible"]:
            return {"action": "reuse", **check}
        raise FileExistsError(f"output exists but is not eligible: {json.dumps(check, ensure_ascii=False)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.name}.building-{os.getpid()}"
    base = work / "base"
    maps = work / "maps"
    final = work / "final"
    if work.exists():
        raise FileExistsError(f"temporary work directory already exists: {work}")
    work.mkdir()
    maps.mkdir()

    try:
        if source is None:
            _build_base_cache(_load_config(config_path), base)
            source = base
        elif not source.is_dir():
            raise FileNotFoundError(f"source cache does not exist: {source}")

        source_data = load_from_disk(str(source))
        source_dict = DatasetDict({"train": source_data}) if isinstance(source_data, Dataset) else source_data
        output_dict = DatasetDict(
            {
                split: _add_length(
                    dataset,
                    length_column,
                    input_column,
                    batch_size,
                    num_proc,
                    maps / f"{split}.arrow",
                )
                for split, dataset in source_dict.items()
            }
        )
        stats = {split: _length_stats(dataset, length_column) for split, dataset in output_dict.items()}
        samples = {
            split: _validate_lengths(dataset, length_column, input_column) for split, dataset in output_dict.items()
        }
        output_dict.save_to_disk(str(final), num_proc=num_proc)

        check = check_cache(final, length_column, input_column)
        if not check["eligible"]:
            raise RuntimeError(f"new cache failed validation: {check}")
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "config": str(config_path),
            "source_cache": str(source),
            "output_cache": str(output),
            "input_column": input_column,
            "length_column": length_column,
            "length_dtype": "int32",
            "stats": stats,
            "sampled_validation_rows": samples,
        }
        (final / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        final.rename(output)
        return {"action": "build", "eligible": True, **manifest}
    finally:
        if work.exists():
            shutil.rmtree(work)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="PT or SFT training YAML")
    parser.add_argument("--source-cache", type=Path, help="existing cache to migrate; omit to build from YAML")
    parser.add_argument(
        "--output-cache",
        type=Path,
        help="override YAML tokenized_path; use a new path when migrating the configured cache",
    )
    parser.add_argument("--length-column", help="override YAML length_column_name")
    parser.add_argument("--input-column", default="input_ids")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.num_proc < 1:
        raise SystemExit("--num-proc must be positive")
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_value = args.output_cache or config.get("tokenized_path")
    if not output_value:
        raise SystemExit("tokenized_path is missing; set it in YAML or pass --output-cache")
    length_column = args.length_column or config.get("length_column_name")
    if not length_column:
        raise SystemExit("length_column_name is missing; set it in YAML or pass --length-column")

    output = _resolve(output_value)
    if not args.check_only and args.source_cache and _resolve(args.source_cache).resolve() == output.resolve():
        raise SystemExit("--source-cache and output cache must differ; pass --output-cache with a new path")
    if args.check_only:
        result = check_cache(output, str(length_column), args.input_column)
    else:
        source = _resolve(args.source_cache) if args.source_cache else None
        result = build_with_length(
            config_path=config_path,
            source=source,
            output=output,
            length_column=str(length_column),
            input_column=args.input_column,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("eligible"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
