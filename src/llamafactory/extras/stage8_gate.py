"""Pure helpers for BrickNet Stage-8 fail-closed initialization gates."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch


EXPECTED_STAGE8_ADAPTER_SUFFIXES = (
    "train_PT_exp1_qwen35_08b_bricknet_text270k_mmpt135k_ep1_bs2_ga8_lora64",
    "train_exp4_2_qwen35_08b_mixedpt_stage2_nonthinking_control_10k_ep3_bs1_ga16_lora64_len16384",
)
ADAPTER_ARTIFACT_NAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
STAGE8_TOKEN_MIX_TOLERANCE = Fraction(1, 200)
STAGE8_TOKEN_MIX_TARGETS = {
    "R1-S": {"R1-S": Fraction(1, 1)},
    "R1-C": {"R1-S": Fraction(4, 5), "R1-C": Fraction(1, 5)},
    "R1-B": {"R1-S": Fraction(7, 10), "R1-C": Fraction(1, 5), "R1-B": Fraction(1, 10)},
}
STAGE8_MAIN_EVAL_MODES = {
    "R1-S": "a0-act-feedback",
    "R1-C": "a0-act-feedback",
    "R1-B": "a1-feedback-search",
}


def adapter_artifact_sha256(path: Path) -> str:
    """Hash the immutable adapter config and exactly one supported weight artifact."""
    files = [path / name for name in ADAPTER_ARTIFACT_NAMES if (path / name).is_file()]
    if path / "adapter_config.json" not in files or len(files) != 2:
        raise ValueError(f"incomplete or ambiguous adapter artifacts: {path}")
    digest = hashlib.sha256()
    for artifact in sorted(files):
        digest.update(artifact.name.encode("utf-8"))
        digest.update(b"\0")
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter_chain(adapter_paths: Iterable[str | Path]) -> list[Path]:
    paths = [Path(path).resolve() for path in adapter_paths]
    suffixes = tuple(path.name for path in paths)
    if suffixes != EXPECTED_STAGE8_ADAPTER_SUFFIXES:
        raise ValueError(
            "Stage-8 must start from the frozen PT-exp1 + exp4_2 adapter chain; "
            f"got {suffixes!r}"
        )
    return paths


def evaluate_supervised_token_mix(counts: Counter[str], run: str) -> dict[str, Any]:
    """Apply the frozen exact-rational token-share rule for one Stage-8 run."""
    target = STAGE8_TOKEN_MIX_TARGETS[run]
    total = sum(counts.values())
    unexpected = sorted(set(counts) - set(target))
    errors: dict[str, float] = {}
    eligible = total > 0 and not unexpected
    for trajectory_type, expected in target.items():
        actual = Fraction(counts[trajectory_type], total) if total else Fraction(0, 1)
        error = abs(actual - expected)
        errors[trajectory_type] = float(error)
        eligible = eligible and error <= STAGE8_TOKEN_MIX_TOLERANCE

    return {
        "target_supervised_token_mix": {key: float(value) for key, value in target.items()},
        "actual_supervised_token_mix": {key: counts[key] / total if total else 0.0 for key in target},
        "token_mix_absolute_error": errors,
        "token_mix_tolerance": float(STAGE8_TOKEN_MIX_TOLERANCE),
        "token_mix_rule": "all absolute share errors <= 1/200 (0.5 percentage point)",
        "unexpected_trajectory_types": unexpected,
        "token_mix_eligible": eligible,
    }


def evaluate_stage8_build_report(
    report: dict[str, Any],
    *,
    run: str,
    expected_size: int,
    dataset_path: Path,
    dataset_sha256: str,
    dataset_count: int,
    dataset_window_materialized: bool,
) -> dict[str, Any]:
    """Bind one BrickNet build report variant to the exact audited dataset."""
    blockers: list[str] = []
    if report.get("schema_version") != "bricknet-stage8-act-sft-build-report-v1":
        blockers.append("BUILD_REPORT_SCHEMA_MISMATCH")
    if report.get("size") != expected_size:
        blockers.append("BUILD_REPORT_SIZE_MISMATCH")
    stage5_gate = report.get("stage5_replay_gate")
    if not isinstance(stage5_gate, dict) or stage5_gate.get("stage5_replay_gate_passed") is not True:
        blockers.append("STAGE5_REPLAY_GATE_NOT_PASSED")
    variants = report.get("variants")
    variant = variants.get(run) if isinstance(variants, dict) else None
    if not isinstance(variant, dict):
        blockers.append("BUILD_REPORT_VARIANT_MISSING")
    else:
        if variant.get("count") != dataset_count:
            blockers.append("BUILD_REPORT_VARIANT_COUNT_MISMATCH")
        try:
            reported_dataset = Path(variant.get("dataset", "")).resolve()
        except TypeError:
            reported_dataset = Path("/")
        if reported_dataset != dataset_path.resolve():
            blockers.append("BUILD_REPORT_VARIANT_PATH_MISMATCH")
        if variant.get("dataset_sha256") != dataset_sha256:
            blockers.append("BUILD_REPORT_VARIANT_SHA256_MISMATCH")
        if variant.get("window_materialized") is not dataset_window_materialized:
            blockers.append("BUILD_REPORT_WINDOW_MATERIALIZATION_MISMATCH")
        type_counts = variant.get("trajectory_type_counts")
        if (
            not isinstance(type_counts, dict)
            or not all(isinstance(value, int) and value >= 0 for value in type_counts.values())
            or sum(type_counts.values()) != dataset_count
        ):
            blockers.append("BUILD_REPORT_TRAJECTORY_COUNTS_MISMATCH")
    return {
        "build_report_gate_passed": not blockers,
        "build_report_blockers": blockers,
    }


def resolve_stage8_eval_mode(run: str, requested_mode: str | None, *, ablation: bool) -> tuple[str, bool]:
    """Freeze main evaluation modes and require an explicit ablation marker for mismatches."""
    main_mode = STAGE8_MAIN_EVAL_MODES[run]
    mode = requested_mode or main_mode
    mismatch = mode != main_mode
    if mismatch and not ablation:
        raise ValueError(f"{run} main evaluation requires {main_mode}; pass --ablation for {mode}")
    return mode, mismatch


def resolve_stage5_report_binding(build_report: dict[str, Any]) -> dict[str, str]:
    """Resolve the immutable Stage-5 report recorded by the formal Stage-8 build."""
    if build_report.get("schema_version") != "bricknet-stage8-act-sft-build-report-v1":
        raise ValueError("Stage-8 build report schema mismatch")
    stage5 = build_report.get("stage5_replay_gate")
    if not isinstance(stage5, dict) or stage5.get("stage5_replay_gate_passed") is not True:
        raise ValueError("Stage-8 build did not pass the Stage-5 replay gate")
    path_value = stage5.get("path")
    expected_sha256 = stage5.get("sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(expected_sha256, str):
        raise ValueError("Stage-8 build report lacks the fixed Stage-5 report path/hash")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"fixed Stage-5 report is missing: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("fixed Stage-5 report hash differs from the Stage-8 build binding")
    return {"path": str(path), "sha256": actual_sha256}


def evaluate_initialization_gate(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    named_parameters: Iterable[tuple[str, Any]],
    adapter_hashes_before: dict[str, str],
    adapter_hashes_after: dict[str, str],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Verify zero-impact fresh LoRA initialization and parameter isolation."""
    if reference_logits.shape != candidate_logits.shape:
        max_abs = float("inf")
        max_rel = float("inf")
        logits_match = False
    else:
        reference = reference_logits.detach().float().cpu()
        candidate = candidate_logits.detach().float().cpu()
        difference = (candidate - reference).abs()
        max_abs = difference.max().item() if difference.numel() else 0.0
        denominator = reference.abs().clamp_min(torch.finfo(torch.float32).eps)
        max_rel = (difference / denominator).max().item() if difference.numel() else 0.0
        logits_match = bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol))

    trainable = sorted(name for name, parameter in named_parameters if parameter.requires_grad)
    frozen = sorted(name for name, parameter in named_parameters if not parameter.requires_grad)
    only_new_lora_trainable = bool(trainable) and all(
        "lora_" in name and ".default." in name for name in trainable
    )
    base_and_merged_chain_frozen = bool(frozen) and all("lora_" not in name for name in frozen)
    adapter_artifacts_unchanged = adapter_hashes_before == adapter_hashes_after
    eligible = (
        logits_match
        and only_new_lora_trainable
        and base_and_merged_chain_frozen
        and adapter_artifacts_unchanged
    )
    return {
        "logits_match": logits_match,
        "logits_atol": atol,
        "logits_rtol": rtol,
        "logits_max_abs_error": max_abs,
        "logits_max_relative_error": max_rel,
        "trainable_parameter_count": len(trainable),
        "trainable_parameter_names": trainable,
        "frozen_parameter_count": len(frozen),
        "only_new_default_lora_trainable": only_new_lora_trainable,
        "base_and_merged_adapter_chain_frozen": base_and_merged_chain_frozen,
        "adapter_artifacts_unchanged": adapter_artifacts_unchanged,
        "adapter_artifact_sha256": adapter_hashes_after,
        "initialization_eligible": eligible,
    }
