#!/usr/bin/env python3
"""Audit BrickNet Stage-1 datasets with the exact LlamaFactory MM SFT processor.

The script loads tokenizer/processor and expands real image tokens, but never
loads model weights and never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_BRICKNET_ROOT = Path("/home/jiahao/task/BrickNet")
DEFAULT_MODEL = (
    Path("/home/jiahao/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots")
    / "2fc06364715b967f1860aea9cf38778875588b17"
)
_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=JSONL",
        help="Repeat for paired datasets.",
    )
    parser.add_argument("--bricknet-root", type=Path, default=DEFAULT_BRICKNET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--template", default="qwen3_5_nothink")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--image-max-pixels", type=int, default=589824)
    parser.add_argument("--image-min-pixels", type=int, default=1024)
    parser.add_argument("--video-max-pixels", type=int, default=65536)
    parser.add_argument("--video-min-pixels", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--audit-purpose",
        default="stage1_full_pool",
        help="Free-form provenance label written to the report.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Per-dataset smoke prefix only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_dataset_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    seen = set()
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --dataset {spec!r}; expected NAME=JSONL")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe dataset name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate dataset name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        parsed.append((name, path))
    return parsed


def _init_worker(config: dict[str, Any]) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(
        model_name_or_path=config["model"],
        trust_remote_code=True,
        image_max_pixels=config["image_max_pixels"],
        image_min_pixels=config["image_min_pixels"],
        video_max_pixels=config["video_max_pixels"],
        video_min_pixels=config["video_min_pixels"],
    )
    data_args = DataArguments(
        template=config["template"],
        dataset_dir=config["media_dir"],
        media_dir=config["media_dir"],
        cutoff_len=config["cutoff_len"],
        train_on_prompt=False,
        mask_history=False,
        packing=False,
        enable_thinking=False,
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    if processor is None:
        raise RuntimeError("model did not provide a multimodal processor")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_attr = DatasetAttr(load_from="file", dataset_name="stage1-jsonl")
    dataset_attr.join(
        {
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
    )
    converter = SharegptDatasetConverter(dataset_attr=dataset_attr, data_args=data_args)
    sft_processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )
    _STATE.update(
        converter=converter,
        sft_processor=sft_processor,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
        image_expansion_cache={},
        image_metadata_cache=OrderedDict(),
        runtime={
            "processor": type(processor).__name__,
            "image_processor": type(processor.image_processor).__name__,
            "mm_plugin": type(template.mm_plugin).__name__,
        },
    )


def _raw_counts(aligned: dict[str, Any]) -> tuple[int, int, int, bool]:
    """Run MM expansion once and count the exact untruncated encoded pairs."""
    template = _STATE["template"]
    tokenizer = _STATE["tokenizer"]
    processor = _STATE["processor"]
    images = aligned["_images"] or []
    videos = aligned["_videos"] or []
    audios = aligned["_audios"] or []
    media_markers = ("<image>", "<video>", "<audio>")
    if any(marker in message["content"] for message in aligned["_response"] for marker in media_markers):
        raise ValueError("assistant response unexpectedly contains a media placeholder")
    prompt = aligned["_prompt"]
    image_placeholders = sum(message["content"].count("<image>") for message in prompt)
    if len(images) == 1 and not videos and not audios and image_placeholders == 1:
        image_path = str(images[0])
        metadata_cache: OrderedDict = _STATE["image_metadata_cache"]
        image_metadata = metadata_cache.get(image_path)
        if image_metadata is None:
            with Image.open(image_path) as image:
                image_metadata = (image.width, image.height, image.mode)
            metadata_cache[image_path] = image_metadata
            metadata_cache.move_to_end(image_path)
            if len(metadata_cache) > 64:
                metadata_cache.popitem(last=False)
        else:
            metadata_cache.move_to_end(image_path)

        expansion_cache: dict = _STATE["image_expansion_cache"]
        image_expansion = expansion_cache.get(image_metadata)
        cache_hit = image_expansion is not None
        if image_expansion is None:
            processed_prompt = template.mm_plugin.process_messages(prompt, images, videos, audios, processor)
            for original, processed in zip(prompt, processed_prompt, strict=True):
                if "<image>" not in original["content"]:
                    continue
                before, separator, after = original["content"].partition("<image>")
                if (
                    not separator
                    or not processed["content"].startswith(before)
                    or not processed["content"].endswith(after)
                ):
                    raise RuntimeError("could not isolate processor-generated image token expansion")
                end = len(processed["content"]) - len(after) if after else len(processed["content"])
                image_expansion = processed["content"][len(before) : end]
                break
            if not image_expansion:
                raise RuntimeError("processor did not expand the image placeholder")
            expansion_cache[image_metadata] = image_expansion
        messages = [
            {"role": message["role"], "content": message["content"].replace("<image>", image_expansion)}
            for message in prompt
        ]
    else:
        messages = template.mm_plugin.process_messages(prompt, images, videos, audios, processor)
        cache_hit = False
    messages.extend(aligned["_response"])
    prefix_ids, prefix_labels = template.mm_plugin.process_token_ids(
        [], [], images, videos, audios, tokenizer, processor
    )
    prefix_labels = prefix_labels or []
    encoded_pairs = template.encode_multiturn(
        tokenizer,
        messages,
        aligned["_system"],
        aligned["_tools"],
        False,
    )
    prompt_count = sum(label == -100 for label in prefix_labels)
    label_count = sum(label != -100 for label in prefix_labels)
    total_count = len(prefix_ids)
    for source_ids, target_ids in encoded_pairs:
        prompt_count += len(source_ids)
        label_count += len(target_ids)
        total_count += len(source_ids) + len(target_ids)
    if template.efficient_eos:
        label_count += 1
        total_count += 1
    if prompt_count + label_count != total_count:
        raise RuntimeError("raw prompt/label mask counts do not sum to raw total")
    return prompt_count, label_count, total_count, cache_hit


def _audit_one(payload: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    dataset_name, line_no, row = payload
    sample_id = row.get("id")
    try:
        aligned = _STATE["converter"](row)
        if len(aligned["_prompt"]) % 2 != 1 or len(aligned["_response"]) != 1:
            raise ValueError("converter did not produce one valid supervised turn")
        raw_prompt, raw_label, raw_total, mm_prompt_cache_hit = _raw_counts(aligned)
        cutoff_len = _STATE["data_args"].cutoff_len
        if raw_total <= cutoff_len:
            effective_prompt, effective_label, effective_total = raw_prompt, raw_label, raw_total
        else:
            images = aligned["_images"] or []
            videos = aligned["_videos"] or []
            audios = aligned["_audios"] or []
            input_ids, labels = _STATE["sft_processor"]._encode_data_example(
                prompt=aligned["_prompt"],
                response=aligned["_response"],
                system=aligned["_system"],
                tools=aligned["_tools"],
                images=images,
                videos=videos,
                audios=audios,
            )
            effective_prompt = sum(token == -100 for token in labels)
            effective_label = sum(token != -100 for token in labels)
            effective_total = len(input_ids)
        return {
            "ok": True,
            "dataset": dataset_name,
            "line": line_no,
            "id": sample_id,
            "raw_prompt_tokens": raw_prompt,
            "raw_label_tokens": raw_label,
            "raw_total_tokens": raw_total,
            "effective_prompt_tokens": effective_prompt,
            "effective_label_tokens": effective_label,
            "effective_total_tokens": effective_total,
            "truncated": effective_total != raw_total,
            "mm_prompt_cache_hit": mm_prompt_cache_hit,
            **_STATE["runtime"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "dataset": dataset_name,
            "line": line_no,
            "id": sample_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _iter_rows(path: Path, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        emitted = 0
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and emitted >= limit:
                break
            yield line_no, json.loads(line)
            emitted += 1


def _iter_payloads(datasets: list[tuple[str, Path]], limit: int | None) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """Interleave paired rows so a worker can reuse identical MM prompt expansion."""
    iterators = [(name, _iter_rows(path, limit)) for name, path in datasets]
    active = True
    while active:
        active = False
        for name, iterator in iterators:
            try:
                line_no, row = next(iterator)
            except StopIteration:
                continue
            active = True
            yield name, line_no, row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def nearest_rank(percent: float) -> int:
        return ordered[max(0, math.ceil(percent * len(ordered)) - 1)]

    return {
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    datasets = _parse_dataset_specs(args.dataset)
    if args.workers < 1 or args.chunksize < 1:
        raise ValueError("--workers and --chunksize must be positive")
    if args.cutoff_len < 1:
        raise ValueError("--cutoff-len must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not Path(args.model).exists() and args.model == str(DEFAULT_MODEL):
        raise FileNotFoundError(f"frozen local model snapshot is missing: {args.model}")

    output_dir = (
        args.output_dir or args.bricknet_root / "outputs_preprocess/BrickNet-MM-Reasoning/reports/token_audit"
    ).resolve()
    report_path = output_dir / "BrickNet-MM-Reasoning_token_audit_report.json"
    sidecar_paths = {name: output_dir / f"{name}_token_audit.jsonl" for name, _ in datasets}
    for path in [report_path, *sidecar_paths.values()]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace the audit")
        path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "media_dir": str(args.bricknet_root.resolve()),
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "video_max_pixels": args.video_max_pixels,
        "video_min_pixels": args.video_min_pixels,
        "train_on_prompt": False,
        "mask_history": False,
        "packing": False,
        "enable_thinking": False,
        "image_token_expansion": "Qwen processor once per (width,height,mode) per worker; exact expansion reused",
        "image_metadata_read_per_record": True,
    }
    payloads = _iter_payloads(datasets, args.limit)
    if args.workers > 1:
        executor = ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(config,))
        results = executor.map(_audit_one, payloads, chunksize=args.chunksize)
    else:
        executor = None
        _init_worker(config)
        results = map(_audit_one, payloads)

    stats = {
        name: {
            "count": 0,
            "errors": 0,
            "truncated": 0,
            "mm_prompt_cache_hits": 0,
            "raw_prompt_tokens": [],
            "raw_label_tokens": [],
            "raw_total_tokens": [],
            "id_digest": hashlib.sha256(),
            "runtime": None,
        }
        for name, _ in datasets
    }
    handles = {name: path.open("w", encoding="utf-8") for name, path in sidecar_paths.items()}
    try:
        for result in results:
            name = result["dataset"]
            handles[name].write(json.dumps(result, ensure_ascii=False) + "\n")
            item = stats[name]
            item["count"] += 1
            item["id_digest"].update((str(result.get("id")) + "\n").encode("utf-8"))
            if not result["ok"]:
                item["errors"] += 1
                continue
            item["truncated"] += int(result["truncated"])
            item["mm_prompt_cache_hits"] += int(result["mm_prompt_cache_hit"])
            for key in ("raw_prompt_tokens", "raw_label_tokens", "raw_total_tokens"):
                item[key].append(result[key])
            item["runtime"] = {key: result[key] for key in ("processor", "image_processor", "mm_plugin")}
    finally:
        for handle in handles.values():
            handle.close()
        if executor is not None:
            executor.shutdown()

    summaries = {}
    for name, _ in datasets:
        item = stats[name]
        summaries[name] = {
            "count": item["count"],
            "errors": item["errors"],
            "truncated": item["truncated"],
            "mm_prompt_cache_hits": item["mm_prompt_cache_hits"],
            "raw_prompt_tokens": _percentiles(item["raw_prompt_tokens"]),
            "raw_label_tokens": _percentiles(item["raw_label_tokens"]),
            "raw_total_tokens": _percentiles(item["raw_total_tokens"]),
            "ordered_id_sha256": item["id_digest"].hexdigest(),
            "runtime": item["runtime"],
            "sidecar": str(sidecar_paths[name]),
            "sidecar_sha256": _sha256_file(sidecar_paths[name]),
        }
    paired = (
        len({summary["count"] for summary in summaries.values()}) == 1
        and len({summary["ordered_id_sha256"] for summary in summaries.values()}) == 1
    )
    report = {
        "stage": args.stage,
        "audit": "real_llamafactory_multimodal_processor",
        "audit_purpose": args.audit_purpose,
        "config": config,
        "limit": args.limit,
        "is_full_pool": args.limit is None,
        "datasets": {
            name: {"path": str(path), "sha256": _sha256_file(path), **summaries[name]} for name, path in datasets
        },
        "paired_order_and_ids": paired,
        "zero_errors": all(summary["errors"] == 0 for summary in summaries.values()),
        "zero_truncation": all(summary["truncated"] == 0 for summary in summaries.values()),
        "training_eligible": (
            args.limit is None
            and paired
            and all(summary["errors"] == 0 and summary["truncated"] == 0 for summary in summaries.values())
        ),
        "percentile_method": "nearest-rank",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["zero_errors"] or not report["zero_truncation"] or not paired:
        raise SystemExit("Stage-1 token audit gate failed; inspect the report and sidecars")


if __name__ == "__main__":
    main()
