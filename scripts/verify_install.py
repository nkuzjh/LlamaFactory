#!/usr/bin/env python3
"""Verify the key runtime dependencies for the LLaMA Factory environment.

This script is intentionally defensive:
- it imports packages one by one and reports failures instead of crashing
- it uses the correct Transformers API for flash-linear-attention checks
- it exits non-zero when any required component is missing
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _import_module(module_name: str) -> tuple[Any | None, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - the script reports runtime failures directly
        return None, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}"
    return module, "ok"


def _format_result(result: CheckResult) -> str:
    status = "OK" if result.ok else "FAIL"
    return f"[{status}] {result.name}: {result.detail}"


def build_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    torch_module, torch_status = _import_module("torch")
    results.append(CheckResult("torch", torch_module is not None, f"{_version('torch')} ({torch_status})"))

    transformers_module, transformers_status = _import_module("transformers")
    results.append(
        CheckResult("transformers", transformers_module is not None, f"{_version('transformers')} ({transformers_status})")
    )

    for package_name in ("causal_conv1d", "flash_attn", "fla", "liger_kernel"):
        module, status = _import_module(package_name)
        results.append(CheckResult(package_name, module is not None, f"{_version(package_name)} ({status})"))

    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        cuda_detail = "CUDA available" if cuda_available else "CUDA unavailable"
        if cuda_available:
            cuda_detail += f", device 0: {torch_module.cuda.get_device_name(0)}"
        results.append(CheckResult("torch.cuda", cuda_available, cuda_detail))

    if transformers_module is not None:
        try:
            from transformers.utils import is_flash_attn_2_available

            flash_attn_2_ok = bool(is_flash_attn_2_available())
            results.append(
                CheckResult(
                    "transformers flash-attn-2 probe",
                    flash_attn_2_ok,
                    f"is_flash_attn_2_available() -> {flash_attn_2_ok}",
                )
            )
        except Exception as exc:  # pragma: no cover
            results.append(CheckResult("transformers flash-attn-2 probe", False, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}"))

        try:
            import transformers.utils.import_utils as import_utils

            has_fla_probe = hasattr(import_utils, "is_flash_linear_attention_available")
            if has_fla_probe:
                fla_probe = getattr(import_utils, "is_flash_linear_attention_available")
                fla_ok = bool(fla_probe())
                results.append(
                    CheckResult(
                        "transformers flash-linear-attention probe",
                        fla_ok,
                        f"transformers.utils.import_utils.is_flash_linear_attention_available() -> {fla_ok}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "transformers flash-linear-attention probe",
                        False,
                        "transformers.utils.import_utils.is_flash_linear_attention_available is missing",
                    )
                )

            top_level_has_fla = hasattr(transformers_module.utils, "is_flash_linear_attention_available")
            results.append(
                CheckResult(
                    "transformers.utils top-level FLA export",
                    True,
                    "export exists" if top_level_has_fla else "not exported at top level; use transformers.utils.import_utils",
                )
            )
        except Exception as exc:  # pragma: no cover
            results.append(CheckResult("transformers flash-linear-attention probe", False, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}"))

    if torch_module is not None:
        try:
            from causal_conv1d import causal_conv1d_fn

            results.append(CheckResult("causal_conv1d_fn", True, f"{causal_conv1d_fn}"))
        except Exception as exc:  # pragma: no cover
            results.append(CheckResult("causal_conv1d_fn", False, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}"))

        try:
            from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

            results.append(CheckResult("fla.modules.convolution.causal_conv1d", True, f"{fla_causal_conv1d}"))
        except Exception as exc:  # pragma: no cover
            results.append(
                CheckResult("fla.modules.convolution.causal_conv1d", False, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}")
            )

        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

            results.append(CheckResult("fla.ops.gated_delta_rule.chunk_gated_delta_rule", True, f"{chunk_gated_delta_rule}"))
            results.append(
                CheckResult(
                    "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
                    True,
                    f"{fused_recurrent_gated_delta_rule}",
                )
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                CheckResult("fla.ops.gated_delta_rule", False, f"IMPORT FAIL: {exc.__class__.__name__}: {exc}")
            )

        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

            results.append(
                CheckResult("liger_kernel.transformers.apply_liger_kernel_to_qwen3_5", True, f"{apply_liger_kernel_to_qwen3_5}")
            )
        except Exception as exc:  # pragma: no cover
            results.append(
                CheckResult(
                    "liger_kernel.transformers.apply_liger_kernel_to_qwen3_5",
                    False,
                    f"IMPORT FAIL: {exc.__class__.__name__}: {exc}",
                )
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the installed runtime dependencies.")
    parser.add_argument("--json", action="store_true", help="reserved for future machine-readable output")
    args = parser.parse_args()

    results = build_checks()

    print("Environment verification")
    print("-" * 80)
    for result in results:
        print(_format_result(result))

    if args.json:
        print("--json is reserved for future use; textual output is the supported mode for now.")

    failed = [result for result in results if not result.ok]
    print("-" * 80)
    if failed:
        print(f"{len(failed)} check(s) failed.")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())