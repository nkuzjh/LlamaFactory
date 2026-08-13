#!/usr/bin/env python3
"""Audit a fresh Stage-8 LoRA before any optimizer or training step is created."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf

from llamafactory.extras.stage8_gate import (
    adapter_artifact_sha256,
    evaluate_initialization_gate,
    validate_adapter_chain,
)
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


SCHEMA_VERSION = "bricknet-stage8-initialization-audit-v1"
DEFAULT_PROMPT = "BrickNet Stage-8 initialization equivalence probe.\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser.parse_args()


def _forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        return model(input_ids=input_ids.to(device), use_cache=False, return_dict=True).logits.detach().cpu()


def main() -> None:
    args = parse_args()
    payload = OmegaConf.to_container(OmegaConf.load(args.config))
    model_args, _, _, finetuning_args, _ = get_train_args(payload)
    if finetuning_args.finetuning_type != "lora" or not finetuning_args.create_new_adapter:
        raise SystemExit("Stage-8 initialization gate requires finetuning_type=lora and create_new_adapter=true")
    if finetuning_args.additional_target:
        raise SystemExit("Stage-8 initialization gate forbids additional_target base parameters")
    adapter_paths = validate_adapter_chain(model_args.adapter_name_or_path or [])
    hashes_before = {str(path): adapter_artifact_sha256(path) for path in adapter_paths}
    tokenizer_module = load_tokenizer(deepcopy(model_args))
    tokenizer = tokenizer_module["tokenizer"]
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    input_sha256 = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()

    torch.manual_seed(42)
    reference_model = load_model(
        tokenizer,
        deepcopy(model_args),
        deepcopy(finetuning_args),
        is_trainable=False,
    )
    reference_logits = _forward_logits(reference_model, input_ids)
    del reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(42)
    candidate_model = load_model(
        tokenizer,
        deepcopy(model_args),
        deepcopy(finetuning_args),
        is_trainable=True,
    )
    candidate_logits = _forward_logits(candidate_model, input_ids)
    named_parameters = list(candidate_model.named_parameters())
    hashes_after = {str(path): adapter_artifact_sha256(path) for path in adapter_paths}
    checks = evaluate_initialization_gate(
        reference_logits,
        candidate_logits,
        named_parameters,
        hashes_before,
        hashes_after,
        atol=args.atol,
        rtol=args.rtol,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "model_name_or_path": model_args.model_name_or_path,
        "adapter_chain": [str(path) for path in adapter_paths],
        "probe_input_sha256": input_sha256,
        "probe_input_tokens": input_ids.numel(),
        **checks,
    }
    _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["initialization_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
