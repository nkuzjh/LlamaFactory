#!/usr/bin/env python3
"""Audit Stage-8 message loss schema and actual supervised Qwen tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--token-mix-approved", action="store_true")
    return parser.parse_args()


def validate_raw(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"{path}:{line_no}: messages must be a non-empty list")
            if messages[-1].get("role") != "assistant":
                raise SystemExit(f"{path}:{line_no}: final message must be assistant")
            targets = 0
            for index, message in enumerate(messages):
                role = message.get("role")
                loss = message.get("loss", role == "assistant")
                if not isinstance(loss, bool):
                    raise SystemExit(f"{path}:{line_no}: message {index} loss is not boolean")
                if role != "assistant" and loss:
                    raise SystemExit(f"{path}:{line_no}: message {index} non-assistant loss=true")
                if role == "assistant":
                    content = message.get("content")
                    if not isinstance(content, str) or not content.endswith("\n"):
                        raise SystemExit(f"{path}:{line_no}: assistant action must retain trailing LF")
                    if any(tag in content for tag in ("<think>", "</think>", "<action>", "</action>")):
                        raise SystemExit(f"{path}:{line_no}: assistant action is not raw path text")
                    targets += int(loss)
            if not targets:
                raise SystemExit(f"{path}:{line_no}: no supervised assistant target")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    raw_count = validate_raw(args.dataset)
    config = OmegaConf.to_container(OmegaConf.load(args.config))
    config["dataset"] = args.dataset_name
    model_args, data_args, training_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    configured_cutoff = data_args.cutoff_len
    data_args.cutoff_len = 10**9
    data_args.tokenized_path = None
    module = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)
    dataset = module["train_dataset"]
    processed_count = len(dataset)
    if processed_count != raw_count:
        raise SystemExit(f"converter silently dropped rows: raw={raw_count}, processed={processed_count}")
    supervised_tokens = 0
    max_length = 0
    cutoff_hits = 0
    zero_target = 0
    for row in dataset:
        length = len(row["input_ids"])
        targets = sum(token != IGNORE_INDEX for token in row["labels"])
        max_length = max(max_length, length)
        cutoff_hits += int(length > configured_cutoff)
        zero_target += int(targets == 0)
        supervised_tokens += targets
    matched = {}
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text())
        baseline_budget = 3 * int(baseline["supervised_assistant_tokens"])
        steps_per_epoch = math.ceil(raw_count / 16)
        tokens_per_step = supervised_tokens / steps_per_epoch
        matched_steps = math.ceil(baseline_budget / tokens_per_step)
        projected = matched_steps * tokens_per_step
        matched = {
            "matched_supervised_token_budget": baseline_budget,
            "matched_max_steps": matched_steps,
            "projected_supervised_tokens": projected,
            "projected_relative_error": abs(projected - baseline_budget) / baseline_budget,
        }
    eligible = processed_count == raw_count and cutoff_hits == 0 and zero_target == 0
    if "R1-C" in args.dataset.name or "R1-B" in args.dataset.name:
        eligible = eligible and args.token_mix_approved and bool(matched)
    report = {
        "schema_version": "bricknet-stage8-act-token-audit-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "input_count": raw_count,
        "processor_output_count": processed_count,
        "supervised_assistant_tokens": supervised_tokens,
        "max_input_tokens": max_length,
        "cutoff_len": configured_cutoff,
        "cutoff_hits": cutoff_hits,
        "zero_supervised_target_rows": zero_target,
        "token_mix_approved": args.token_mix_approved,
        "training_eligible": eligible,
        **matched,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
