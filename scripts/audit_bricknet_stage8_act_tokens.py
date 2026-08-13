#!/usr/bin/env python3
"""Fail-closed Stage-8 schema, token-mix, and window-boundary audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processor import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.stage8_gate import STAGE8_TOKEN_MIX_TARGETS, evaluate_supervised_token_mix
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


SCHEMA_VERSION = "bricknet-stage8-act-token-audit-v2"
BOUNDARY_SCHEMA_VERSION = "bricknet-stage8-window-boundary-plan-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--boundary-plan",
        type=Path,
        help="Default: REPORT with .boundary_plan.jsonl suffix.",
    )
    return parser.parse_args()


def _run_name(dataset_name: str) -> str:
    matches = [run for run in STAGE8_TOKEN_MIX_TARGETS if run in dataset_name]
    if len(matches) != 1:
        raise SystemExit(f"cannot infer exactly one R1 run from dataset name: {dataset_name}")
    return matches[0]


def _validate_raw(path: Path, expected_run: str) -> list[dict[str, Any]]:
    rows = []
    allowed_types = set(STAGE8_TOKEN_MIX_TARGETS[expected_run])
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            record_id = row.get("id")
            trajectory_type = row.get("trajectory_type")
            messages = row.get("messages")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise SystemExit(f"{path}:{line_no}: id must be a unique non-empty string")
            if trajectory_type not in allowed_types:
                raise SystemExit(
                    f"{path}:{line_no}: trajectory_type {trajectory_type!r} is not allowed for {expected_run}"
                )
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"{path}:{line_no}: messages must be a non-empty list")
            if messages[-1].get("role") != "assistant":
                raise SystemExit(f"{path}:{line_no}: final message must be assistant")
            targets = 0
            assistant_message_indexes = []
            supervised_ordinal = 0
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
                    assistant_message_indexes.append(
                        {
                            "message_index": index,
                            "loss": loss,
                            "supervised_turn_ordinal": supervised_ordinal if loss else None,
                        }
                    )
                    targets += int(loss)
                    supervised_ordinal += int(loss)
            if not targets:
                raise SystemExit(f"{path}:{line_no}: no supervised assistant target")
            seen_ids.add(record_id)
            rows.append(
                {
                    "row": row,
                    "line_no": line_no,
                    "id": record_id,
                    "trajectory_type": trajectory_type,
                    "assistant_messages": assistant_message_indexes,
                    "supervised_turns": targets,
                }
            )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_attr(config_dataset_dir: str, dataset_name: str, dataset: Path) -> DatasetAttr:
    info_path = Path(config_dataset_dir) / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if dataset_name not in info:
        raise SystemExit(f"{dataset_name!r} is absent from {info_path}")
    entry = info[dataset_name]
    if entry.get("formatting") != "sharegpt":
        raise SystemExit(f"{dataset_name}: Stage-8 audit requires sharegpt formatting")
    attr = DatasetAttr("file", dataset_name=str(dataset.resolve()))
    attr.join(entry)
    return attr


def _convert_and_encode(
    raw_row: dict[str, Any],
    converter,
    processor: SupervisedDatasetProcessor,
) -> tuple[list[int], list[int], list[dict[str, int | bool]]]:
    converted = converter(raw_row)
    prompt = converted["_prompt"]
    response = converted["_response"]
    if len(prompt) % 2 != 1 or len(response) != 1:
        raise ValueError("converter rejected or changed the Stage-8 conversation shape")
    return processor._encode_data_example_with_turn_boundaries(
        prompt=prompt,
        response=response,
        system=converted["_system"],
        tools=converted["_tools"],
        images=converted["_images"] or [],
        videos=converted["_videos"] or [],
        audios=converted["_audios"] or [],
    )


def _materialize_window(row: dict[str, Any], message_start: int, message_end: int) -> dict[str, Any]:
    """Apply the exact message-boundary contract consumed by BrickNet.

    Every window retains the original system/user/image prompt. For later
    windows, the reset observation is removed from the original user and
    replaced by the full accepted raw-action prefix plus the frozen verified
    observation immediately following the previous supervised action.
    """
    messages = row["messages"]
    if len(messages) < 3 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("Stage-8 window materialization requires leading system and user messages")
    if messages[message_end - 1]["role"] != "assistant" or not messages[message_end - 1].get("loss", True):
        raise ValueError("window must end at a supervised assistant boundary")
    window = deepcopy(row)
    system = deepcopy(messages[0])
    user = deepcopy(messages[1])
    if message_start == 1:
        body_start = 2
    else:
        context = messages[message_start]
        if context.get("role") != "observation" or context.get("loss", False):
            raise ValueError("later window must start from a masked observation context")
        observation_marker = "\n\n<observation>\nstatus=ready\nreason=reset\nstep=0\n"
        marker_index = user["content"].rfind(observation_marker)
        if marker_index < 0 or not user["content"].endswith("</observation>"):
            raise ValueError("original user does not end with the frozen ready/reset observation")
        accepted_prefix = "".join(
            message["content"]
            for index, message in enumerate(messages)
            if index < message_start and message.get("role") == "assistant" and message.get("loss", True)
        )
        if not accepted_prefix:
            raise ValueError("later window requires a non-empty accepted BrickNet prefix")
        user["content"] = (
            user["content"][:marker_index]
            + "\n\nAccepted BrickNet prefix:\n"
            + accepted_prefix
            + "\n"
            + context["content"]
        )
        body_start = message_start + 1
    window["messages"] = [system, user, *deepcopy(messages[body_start:message_end])]
    return window


def _canonical_messages_sha256(messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_plan(
    item: dict[str, Any],
    full_length: int,
    cutoff_len: int,
    encode_length: Callable[[dict[str, Any]], int],
) -> tuple[list[dict[str, Any]], bool]:
    row = item["row"]
    supervised = [entry for entry in item["assistant_messages"] if entry["loss"]]
    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(supervised):
        message_start = 1 if start == 0 else int(supervised[start - 1]["message_index"]) + 1
        low, high = start + 1, len(supervised)
        best: tuple[int, int] | None = None
        while low <= high:
            end = (low + high) // 2
            message_end = int(supervised[end - 1]["message_index"]) + 1
            if start == 0 and end == len(supervised):
                length = full_length
            else:
                length = encode_length(_materialize_window(row, message_start, message_end))
            if length <= cutoff_len:
                best = (end, length)
                low = end + 1
            else:
                high = end - 1
        if best is None:
            message_end = int(supervised[start]["message_index"]) + 1
            materialized = _materialize_window(row, message_start, message_end)
            length = encode_length(materialized)
            windows.append(
                {
                    "start_supervised_turn": start,
                    "end_supervised_turn": start + 1,
                    "message_start": message_start,
                    "message_end": message_end,
                    "processed_input_tokens": length,
                    "canonical_messages_sha256": _canonical_messages_sha256(materialized["messages"]),
                }
            )
            return windows, False
        end, length = best
        materialized = _materialize_window(row, message_start, int(supervised[end - 1]["message_index"]) + 1)
        windows.append(
            {
                "start_supervised_turn": start,
                "end_supervised_turn": end,
                "message_start": message_start,
                "message_end": int(supervised[end - 1]["message_index"]) + 1,
                "processed_input_tokens": length,
                "canonical_messages_sha256": _canonical_messages_sha256(materialized["messages"]),
            }
        )
        start = end
    allocated = [ordinal for window in windows for ordinal in range(window["start_supervised_turn"], window["end_supervised_turn"])]
    return windows, allocated == list(range(len(supervised))) and all(
        window["processed_input_tokens"] <= cutoff_len for window in windows
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run = _run_name(args.dataset_name)
    raw_rows = _validate_raw(args.dataset, run)
    dataset_sha256 = sha256_file(args.dataset)
    config_payload = OmegaConf.to_container(OmegaConf.load(args.config))
    config_payload["dataset"] = args.dataset_name
    model_args, data_args, _, _, _ = get_train_args(config_payload)
    if data_args.packing or data_args.mask_history or data_args.train_on_prompt:
        raise SystemExit("Stage-8 boundary audit requires packing=false, mask_history=false, train_on_prompt=false")
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    configured_cutoff = data_args.cutoff_len
    data_args.cutoff_len = 10**9
    data_args.tokenized_path = None
    dataset_attr = _dataset_attr(str(data_args.dataset_dir), args.dataset_name, args.dataset)
    converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module["processor"],
        data_args=data_args,
    )

    token_counts: Counter[str] = Counter()
    supervised_tokens = 0
    max_length = 0
    cutoff_hits = 0
    zero_target = 0
    boundary_rows = []
    window_plan_failures = 0
    for item in raw_rows:
        try:
            input_ids, labels, turn_boundaries = _convert_and_encode(item["row"], converter, processor)
        except Exception as exc:
            raise SystemExit(f"{args.dataset}:{item['line_no']}: processor rejected row: {exc}") from exc
        if len(turn_boundaries) != len(item["assistant_messages"]):
            raise SystemExit(f"{args.dataset}:{item['line_no']}: assistant turn alignment changed during processing")
        targets = sum(token != IGNORE_INDEX for token in labels)
        length = len(input_ids)
        max_length = max(max_length, length)
        cutoff_hits += int(length > configured_cutoff)
        zero_target += int(targets == 0)
        supervised_tokens += targets
        token_counts[item["trajectory_type"]] += targets
        assistant_turns = []
        for raw_turn, encoded_turn in zip(item["assistant_messages"], turn_boundaries):
            assistant_turns.append(
                {
                    "assistant_turn_ordinal": int(encoded_turn["assistant_turn_ordinal"]),
                    "supervised_turn_ordinal": raw_turn["supervised_turn_ordinal"],
                    "message_index": raw_turn["message_index"],
                    "loss": raw_turn["loss"],
                    "target_token_start": int(encoded_turn["target_token_start"]),
                    "target_token_end": int(encoded_turn["target_token_end"]),
                    "target_token_count": int(encoded_turn["target_token_count"]),
                    "supervised_token_count": int(encoded_turn["supervised_token_count"]),
                }
            )

        def encode_length(window_row: dict[str, Any]) -> int:
            window_input_ids, _, _ = _convert_and_encode(window_row, converter, processor)
            return len(window_input_ids)

        windows, plan_ok = _window_plan(item, length, configured_cutoff, encode_length)
        window_plan_failures += int(not plan_ok)
        boundary_rows.append(
            {
                "schema_version": BOUNDARY_SCHEMA_VERSION,
                "source_record_id": item["id"],
                "dataset_path": str(args.dataset.resolve()),
                "dataset_sha256": dataset_sha256,
                "trajectory_type": item["trajectory_type"],
                "cutoff_len": configured_cutoff,
                "untruncated_input_tokens": length,
                "assistant_turns": assistant_turns,
                "windows": windows,
                "window_plan_eligible": plan_ok,
                "requires_window_materialization": length > configured_cutoff,
            }
        )

    mix = evaluate_supervised_token_mix(token_counts, run)
    matched: dict[str, Any] = {}
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        if baseline.get("schema_version") != SCHEMA_VERSION or not baseline.get("training_eligible"):
            raise SystemExit("baseline report is not an eligible Stage-8 v2 token audit")
        baseline_budget = 3 * int(baseline["supervised_assistant_tokens"])
        steps_per_epoch = math.ceil(len(raw_rows) / 16)
        tokens_per_step = supervised_tokens / steps_per_epoch
        matched_steps = math.ceil(baseline_budget / tokens_per_step)
        projected = matched_steps * tokens_per_step
        matched = {
            "baseline_report": str(args.baseline_report.resolve()),
            "baseline_report_sha256": sha256_file(args.baseline_report),
            "matched_supervised_token_budget": baseline_budget,
            "matched_max_steps": matched_steps,
            "projected_supervised_tokens": projected,
            "projected_relative_error": abs(projected - baseline_budget) / baseline_budget,
        }
    matched_eligible = run == "R1-S" or bool(matched)
    eligible = (
        len(raw_rows) > 0
        and cutoff_hits == 0
        and zero_target == 0
        and window_plan_failures == 0
        and mix["token_mix_eligible"]
        and matched_eligible
    )
    boundary_plan = args.boundary_plan or args.report.with_suffix(".boundary_plan.jsonl")
    _write_jsonl_atomic(boundary_plan, boundary_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "input_count": len(raw_rows),
        "processor_output_count": len(boundary_rows),
        "supervised_assistant_tokens": supervised_tokens,
        "supervised_assistant_tokens_by_trajectory_type": dict(sorted(token_counts.items())),
        "max_input_tokens": max_length,
        "cutoff_len": configured_cutoff,
        "cutoff_hits": cutoff_hits,
        "zero_supervised_target_rows": zero_target,
        "boundary_plan": str(boundary_plan.resolve()),
        "boundary_plan_sha256": sha256_file(boundary_plan),
        "boundary_plan_failures": window_plan_failures,
        "canonical_messages_sha256_rule": (
            "sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, "
            "separators=(',', ':')).encode('utf-8'))"
        ),
        "requires_window_materialization_rows": cutoff_hits,
        "window_materialization_rule": (
            "retain system+user+images; for later windows remove the ready/reset observation from user, append "
            "'\\n\\nAccepted BrickNet prefix:\\n' + all prior loss=true raw actions + '\\n' + the message_start "
            "five-field observation, then append messages through message_end; canonical messages use sorted-key "
            "compact UTF-8 JSON; rerun this audit on BrickNet-materialized rows"
        ),
        **mix,
        **matched,
        "training_eligible": eligible,
    }
    _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
