#!/usr/bin/env python3
"""Run the four approved BrickNet PT unconditional NS comparisons.

This launcher is intentionally a small, fail-closed orchestration layer.  It
does not import ``torch``, ``transformers`` or BrickNet, and therefore a
preflight/dry-run can never load a model or allocate a GPU.  The only actions
which start an external process are ``smoke``, ``pilot``, ``generate`` and
``evaluate`` with ``--execute``.  Formal generation is staged in ``.building``;
only a successful, independent ``verify --execute`` promotes that directory
to the final output path.

The public experiment keys are deliberately only the four keys in
``EXPERIMENTS``.  They are *not* a general sweep interface: sample counts,
model/adapter pairing, prompt and sampling settings are all frozen here.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRICKNET_ROOT = ROOT.parent / "BrickNet"
LLAMAFACTORY_PYTHON = Path("/home/jiahao/miniconda3/envs/llamafactory/bin/python")
BRICKNET_PYTHON = Path("/home/jiahao/miniconda3/envs/bricknet/bin/python")

GENERATE_SCRIPT = BRICKNET_ROOT / "scripts/generate.py"
EVALUATE_SCRIPT = BRICKNET_ROOT / "scripts/evaluate_unconditional_generation.py"
BRICKNET_DATA = BRICKNET_ROOT / "data/bricknet_datasets"
CATALOG_ROOT = BRICKNET_ROOT / "src/bricknet/_data/v1"
COLLISION_CATALOG_FILES = (
    "part_names.json",
    "color_names.json",
    "labels.json.xz",
    "part_aliases.json.xz",
)
INSET_ROOT_NAME = "inset"
EXPECTED_COLLISION_MESHES = 21_084

# ``OUTPUT_ROOT`` is a parent, rather than a single run directory.  Keeping
# the experiment component below it makes accidental cross-arm promotion
# impossible and also lets tests replace the parent with a temporary path.
OUTPUT_ROOT = BRICKNET_ROOT / "outputs_pt/paper_pt_unconditional"

SCHEMA_VERSION = "bricknet-paper-pt-unconditional-launch-v1"
EVALUATION_SCHEMA_VERSION = "bricknet-unconditional-evaluation-v1"
EXPECTED_PROMPT = "a"
EXPECTED_MAX_NEW_TOKENS = 4_096
EXPECTED_TEMPERATURE = 1.0
EXPECTED_TOP_K = 20
EXPECTED_TOP_P = 0.95
EXPECTED_DTYPE = "bfloat16"
EXPECTED_STOP_AFTER_NEWLINES = 199
EXPECTED_SEED: None = None
EXPECTED_NEWLINE_TOKEN = "Ċ"
EXPECTED_NEWLINE_TOKEN_ID = 198
EXPECTED_PROMPT_TOKEN_IDS = (64,)
EXPECTED_EOS_TOKEN = "<|im_end|>"
EXPECTED_EOS_TOKEN_ID = 151_645
EXPECTED_PAD_TOKEN = "<|endoftext|>"
EXPECTED_GENERATION_EOS_TOKEN_IDS = (151_645, 151_643)
EXPECTED_GENERATION_BOS_TOKEN_ID = 151_643
EXPECTED_GENERATION_PAD_TOKEN_ID = 151_643
EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID = EXPECTED_EOS_TOKEN_ID
EXPECTED_EOS_TOKEN_IDS = EXPECTED_GENERATION_EOS_TOKEN_IDS

# Smoke and pilot sizes are fixed diagnostics, not user-selectable sample
# counts.  Formal N is part of each immutable experiment identity below.
SMOKE_SAMPLES = 1
PILOT_SAMPLES = 8

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# Base weights are part of the immutable run identity.  They are intentionally
# hashed only by an actual preflight/execute path (the unit tests replace them
# with tiny fixtures); a path and byte count alone cannot detect a local
# checkpoint replacement.
BASE_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
BASE_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
QWEN3_CONFIG_FINGERPRINT_KEYS = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "rope_scaling",
    "rope_theta",
    "tie_word_embeddings",
)
# These values are the reviewed Qwen3 base-model semantics.  A path and a new
# content hash are not sufficient to identify a compatible arm: changing the
# model shape or tokenizer vocabulary must fail before generation.
QWEN3_CONFIG_FINGERPRINTS: dict[str, dict[str, Any]] = {
    "Qwen3-0.6B": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
    },
    "Qwen3-1.7B": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
    },
    "Qwen3-8B": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
    },
    "Qwen3-14B": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
    },
}
EXPECTED_QWEN3_CONFIG_FINGERPRINTS = QWEN3_CONFIG_FINGERPRINTS
GPU_SELECTOR_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

RUN_SPEC_NAME = "run_spec.json"
FINAL_MANIFEST_NAME = "manifest.json"
RAW_NAME = "out.jsonl"
SCORED_NAME = "scored.jsonl"
METRICS_NAME = "metrics.json"
REPORT_NAME = "metrics.md"
EVALUATION_MANIFEST_NAME = "evaluation_manifest.json"
EVALUATION_SUBDIR = "evaluation"
EVALUATION_OUTPUTS = (
    SCORED_NAME,
    METRICS_NAME,
    REPORT_NAME,
    EVALUATION_MANIFEST_NAME,
)
STAGED_OUTPUTS = (RUN_SPEC_NAME, RAW_NAME, *EVALUATION_OUTPUTS)
PROMOTABLE_OUTPUTS = (RUN_SPEC_NAME, RAW_NAME, *EVALUATION_OUTPUTS, FINAL_MANIFEST_NAME)
_SHA256_CACHE: dict[tuple[str, int, int, int], str | None] = {}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class GateError(RuntimeError):
    """Raised when an identity, protocol or artifact gate fails."""


@dataclass(frozen=True)
class Experiment:
    """One immutable BrickNet PT unconditional NS arm."""

    key: str
    identity: str
    label: str
    model_name: str
    adapter_name: str
    base_path: Path
    base_revision: str
    adapter_path: Path
    adapter_revision: str
    expected_samples: int
    batch_size: int
    slug: str

    @property
    def n(self) -> int:
        """Short alias used by command and manifest callers."""
        return self.expected_samples


def _hf_snapshot(repo: str, revision: str) -> Path:
    return (
        Path("/home/jiahao/.cache/huggingface/hub")
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / revision
    )


_BASE_06_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
_BASE_17_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_ADAPTER_06_REVISION = "3f5470a0cc45ecf5646d0d4e350704c9dc7e1807"
_ADAPTER_17_REVISION = "0f75d5e5922234f129a3bb42307c14c685f098f2"
_ADAPTER_8_REVISION = "a4c23748aee599577d65bf2581a8aa52e8a6bc40"
_ADAPTER_14_REVISION = "510afeeea5ba20f7202bc4c9c355b57594d07575"


def _local_base(name: str) -> Path:
    return BRICKNET_ROOT / "hf_checkpoints/Qwen" / name


# Exactly four accepted identities.  The values intentionally carry both the
# base and adapter names so a run_spec cannot silently pair a 14B adapter with
# an 8B base.
EXPERIMENTS: dict[str, Experiment] = {
    "bricknet-pt-qwen3-8b-uncond-ns-n2048-v1": Experiment(
        key="bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
        identity="bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
        label="Qwen3-8B+BrickNet-8B-PT",
        model_name="Qwen3-8B",
        adapter_name="BrickNet-8B-PT",
        base_path=_local_base("Qwen3-8B"),
        base_revision="workspace-local",
        adapter_path=_hf_snapshot("kulits/BrickNet-8B-PT", _ADAPTER_8_REVISION),
        adapter_revision=_ADAPTER_8_REVISION,
        expected_samples=2_048,
        batch_size=1,
        slug="bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
    ),
    "bricknet-pt-qwen3-14b-uncond-ns-n2048-v1": Experiment(
        key="bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
        identity="bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
        label="Qwen3-14B+BrickNet-14B-PT",
        model_name="Qwen3-14B",
        adapter_name="BrickNet-14B-PT",
        base_path=_local_base("Qwen3-14B"),
        base_revision="workspace-local",
        adapter_path=_hf_snapshot("kulits/BrickNet-14B-PT", _ADAPTER_14_REVISION),
        adapter_revision=_ADAPTER_14_REVISION,
        expected_samples=2_048,
        batch_size=1,
        slug="bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
    ),
    "bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1": Experiment(
        key="bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
        identity="bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
        label="Qwen3-0.6B+BrickNet-0.6B-PT",
        model_name="Qwen3-0.6B",
        adapter_name="BrickNet-0.6B-PT",
        base_path=_hf_snapshot("Qwen/Qwen3-0.6B", _BASE_06_REVISION),
        base_revision=_BASE_06_REVISION,
        adapter_path=_hf_snapshot("kulits/BrickNet-0.6B-PT", _ADAPTER_06_REVISION),
        adapter_revision=_ADAPTER_06_REVISION,
        expected_samples=65_536,
        batch_size=8,
        slug="bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
    ),
    "bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1": Experiment(
        key="bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
        identity="bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
        label="Qwen3-1.7B+BrickNet-1.7B-PT",
        model_name="Qwen3-1.7B",
        adapter_name="BrickNet-1.7B-PT",
        base_path=_hf_snapshot("Qwen/Qwen3-1.7B", _BASE_17_REVISION),
        base_revision=_BASE_17_REVISION,
        adapter_path=_hf_snapshot("kulits/BrickNet-1.7B-PT", _ADAPTER_17_REVISION),
        adapter_revision=_ADAPTER_17_REVISION,
        expected_samples=65_536,
        batch_size=4,
        slug="bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
    ),
}

# Deliberately expose common names used by review scripts without adding
# entries to EXPERIMENTS (and therefore without creating additional arms).
RUNS = EXPERIMENTS
EXPERIMENT_IDS = tuple(EXPERIMENTS)
PAPER_EXPERIMENTS = EXPERIMENTS
NS_EXPERIMENT_IDENTITIES = tuple(item.identity for item in EXPERIMENTS.values())
EXPERIMENT_IDENTITIES = NS_EXPERIMENT_IDENTITIES
_EXPERIMENT_ALIASES = {
    value.label: key for key, value in EXPERIMENTS.items()
}
_EXPERIMENT_ALIASES.update(
    {
        "qwen3_8b": "bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
        "qwen3_14b": "bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
        "qwen3_0_6b": "bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
        "qwen3_1_7b": "bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
        "qwen3-8b+bricknet-8b-pt": "bricknet-pt-qwen3-8b-uncond-ns-n2048-v1",
        "qwen3-14b+bricknet-14b-pt": "bricknet-pt-qwen3-14b-uncond-ns-n2048-v1",
        "qwen3-0.6b+bricknet-0.6b-pt": "bricknet-pt-qwen3-06b-uncond-paper-ns-n65536-v1",
        "qwen3-1.7b+bricknet-1.7b-pt": "bricknet-pt-qwen3-17b-uncond-paper-ns-n65536-v1",
    }
)


def _resolve_experiment(value: str | Experiment) -> Experiment:
    if isinstance(value, Experiment):
        return value
    key = _EXPERIMENT_ALIASES.get(value, value)
    try:
        return EXPERIMENTS[key]
    except KeyError as exc:
        raise GateError(
            f"unknown experiment {value!r}; only {', '.join(EXPERIMENTS)} are allowed"
        ) from exc


def _exp_output_root(experiment: str | Experiment) -> Path:
    return OUTPUT_ROOT / _resolve_experiment(experiment).slug


def _output_root(experiment: str | Experiment) -> Path:
    """Compatibility spelling for callers that inspect planned paths."""
    return _exp_output_root(experiment)


def _building_root(experiment: str | Experiment) -> Path:
    root = _exp_output_root(experiment)
    return root.with_name(root.name + ".building")


def _smoke_root(experiment: str | Experiment) -> Path:
    root = _exp_output_root(experiment)
    return root.with_name(root.name + ".smoke")


def _pilot_root(experiment: str | Experiment) -> Path:
    root = _exp_output_root(experiment)
    return root.with_name(root.name + ".pilot")


def _scope_root(experiment: str | Experiment, scope: str) -> Path:
    if scope == "formal":
        return _building_root(experiment)
    if scope == "smoke":
        return _smoke_root(experiment)
    if scope == "pilot":
        return _pilot_root(experiment)
    raise GateError(f"unknown artifact scope: {scope}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) and math.isfinite(float(value))


def _sha256(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if cache_key in _SHA256_CACHE:
        return _SHA256_CACHE[cache_key]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def _valid_sha256(value: Any) -> bool:
    """Return whether a recorded digest is a concrete SHA-256 value."""
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename *source* without replacing an existing destination.

    The launcher deliberately does not fall back to ``os.rename`` or
    ``Path.replace``: both may overwrite a destination in a race.  A platform
    without Linux ``renameat2(RENAME_NOREPLACE)`` therefore fails closed.
    """
    if sys.platform != "linux" or os.name != "posix":
        raise GateError("atomic no-replace rename is only supported on Linux")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise GateError("Linux renameat2(RENAME_NOREPLACE) is unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise GateError(f"refusing to overwrite existing path: {destination}")
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise GateError("Linux renameat2(RENAME_NOREPLACE) is unavailable")
    raise OSError(error, os.strerror(error), str(source))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink() and not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() and not path.exists():
        raise GateError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise GateError(f"blank JSONL row: {path}:{line_number}")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise GateError(f"invalid JSONL row: {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise GateError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except OSError as exc:
        raise GateError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise GateError(f"temporary file already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # ``renameat2(RENAME_NOREPLACE)`` is atomic and keeps a concurrently
        # created destination intact.
        if path.exists() or path.is_symlink():
            raise GateError(f"refusing to overwrite existing file: {path}")
        _rename_noreplace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"cannot update missing run_spec: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise GateError(f"temporary file already exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _update_run_spec(path: Path, **updates: Any) -> dict[str, Any]:
    spec = _load_json(path)
    if spec is None:
        raise GateError(f"invalid run_spec: {path}")
    spec.update(updates)
    _replace_json(path, spec)
    return spec


def _parse_gpu_selector(value: str | None) -> tuple[str, ...]:
    if value is None or not GPU_SELECTOR_PATTERN.fullmatch(value):
        return ()
    selectors = tuple(value.split(","))
    if len(set(selectors)) != len(selectors):
        return ()
    return selectors


def _gpu_value(value: str | None) -> str | None:
    """Normalize one CLI selector while retaining the physical IDs."""
    return value if _parse_gpu_selector(value) else None


def _file_identity(path: Path, names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        item = path / name
        result[name] = {
            "path": str(item),
            "exists": item.is_file(),
            "symlink": item.is_symlink(),
            "bytes": item.stat().st_size if item.is_file() else None,
            "sha256": _sha256(item),
        }
    return result


def _adapter_identity(experiment: str | Experiment) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    return {
        "name": exp.adapter_name,
        "path": str(exp.adapter_path),
        "revision": exp.adapter_revision,
        "files": _file_identity(exp.adapter_path, ADAPTER_FILES),
    }


def _base_weight_files(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for item in sorted(path.glob("*.safetensors")):
        if item.is_file():
            result.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
            )
    return result


def _base_config_semantics(experiment: str | Experiment) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    expected = QWEN3_CONFIG_FINGERPRINTS.get(exp.model_name)
    config = _load_json(exp.base_path / "config.json")
    if expected is None:
        return {
            "expected": None,
            "actual": None,
            "missing": list(QWEN3_CONFIG_FINGERPRINT_KEYS),
            "matches": False,
        }
    actual = {
        key: config.get(key) if config is not None else None
        for key in QWEN3_CONFIG_FINGERPRINT_KEYS
    }
    missing = [key for key in QWEN3_CONFIG_FINGERPRINT_KEYS if config is None or key not in config]
    return {
        "expected": expected,
        "actual": actual,
        "missing": missing,
        "matches": not missing and actual == expected,
    }


def _generation_config_protocol() -> dict[str, Any]:
    return {
        "eos_token_id": list(EXPECTED_GENERATION_EOS_TOKEN_IDS),
        "bos_token_id": EXPECTED_GENERATION_BOS_TOKEN_ID,
        "pad_token_id": EXPECTED_GENERATION_PAD_TOKEN_ID,
        "effective_generation_pad_token_id": EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID,
    }


def _generation_config_source_protocol() -> dict[str, Any]:
    protocol = _generation_config_protocol()
    protocol.pop("effective_generation_pad_token_id")
    return protocol


def _generation_config_semantics(experiment: str | Experiment) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    expected = _generation_config_source_protocol()
    config = _load_json(exp.base_path / "generation_config.json")
    actual = {key: config.get(key) if config is not None else None for key in expected}
    missing = [key for key in expected if config is None or key not in config]
    return {
        "expected": expected,
        "actual": actual,
        "missing": missing,
        "matches": not missing and actual == expected,
        "effective_generation_pad_token_id": EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID,
    }


def _base_identity(experiment: str | Experiment) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    return {
        "name": exp.model_name,
        "path": str(exp.base_path),
        "revision": exp.base_revision,
        "config_semantics": _base_config_semantics(exp),
        "generation_config_semantics": _generation_config_semantics(exp),
        "files": _file_identity(exp.base_path, BASE_IDENTITY_FILES),
        "weight_files": _base_weight_files(exp.base_path),
    }


def _base_identity_hashes_complete(identity: Mapping[str, Any]) -> bool:
    """Require hashes for all mandatory metadata and every recorded shard.

    ``model.safetensors.index.json`` is optional for the known single-shard
    Qwen3-0.6B snapshot.  When an index is present it is part of the identity
    and therefore must itself be readable and hashed.
    """
    semantics = identity.get("config_semantics")
    if not isinstance(semantics, Mapping) or semantics.get("matches") is not True:
        return False
    generation_semantics = identity.get("generation_config_semantics")
    if (
        not isinstance(generation_semantics, Mapping)
        or generation_semantics.get("matches") is not True
        or generation_semantics.get("effective_generation_pad_token_id")
        != EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID
    ):
        return False
    files = identity.get("files")
    if not isinstance(files, Mapping):
        return False
    for name in BASE_REQUIRED_FILES:
        info = files.get(name)
        if not isinstance(info, Mapping) or not info.get("exists") or not _valid_sha256(info.get("sha256")):
            return False
    index_info = files.get("model.safetensors.index.json")
    if not isinstance(index_info, Mapping):
        return False
    if index_info.get("exists") and not _valid_sha256(index_info.get("sha256")):
        return False
    weight_files = identity.get("weight_files")
    if not isinstance(weight_files, list) or not weight_files:
        return False
    return all(
        isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and _valid_sha256(item.get("sha256"))
        for item in weight_files
    )


def _adapter_identity_hashes_complete(identity: Mapping[str, Any]) -> bool:
    files = identity.get("files")
    if not isinstance(files, Mapping):
        return False
    return all(
        isinstance(files.get(name), Mapping)
        and files[name].get("exists")
        and _valid_sha256(files[name].get("sha256"))
        for name in ADAPTER_FILES
    )


def _tokenizer_protocol(experiment: str | Experiment) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    return {
        "path": str(exp.base_path / "tokenizer.json"),
        "config_path": str(exp.base_path / "tokenizer_config.json"),
        "files": {
            "tokenizer.json": _sha256(exp.base_path / "tokenizer.json"),
            "tokenizer_config.json": _sha256(exp.base_path / "tokenizer_config.json"),
        },
        "prompt": EXPECTED_PROMPT,
        "prompt_token_ids": list(EXPECTED_PROMPT_TOKEN_IDS),
        "newline_token": EXPECTED_NEWLINE_TOKEN,
        "newline_token_id": EXPECTED_NEWLINE_TOKEN_ID,
        "eos_token": EXPECTED_EOS_TOKEN,
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "pad_token": EXPECTED_PAD_TOKEN,
        "pad_token_id": EXPECTED_GENERATION_PAD_TOKEN_ID,
        "source_pad_token": EXPECTED_PAD_TOKEN,
        "source_pad_token_id": EXPECTED_GENERATION_PAD_TOKEN_ID,
        "effective_generation_pad_token_id": EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID,
        "chat_template": False,
        "raw_prompt": True,
    }


def _check_tokenizer(experiment: str | Experiment) -> tuple[dict[str, Any], list[str]]:
    exp = _resolve_experiment(experiment)
    tokenizer_path = exp.base_path / "tokenizer.json"
    config_path = exp.base_path / "tokenizer_config.json"
    tokenizer = _load_json(tokenizer_path)
    config = _load_json(config_path)
    blockers: list[str] = []
    model = tokenizer.get("model") if tokenizer else None
    vocab = model.get("vocab") if isinstance(model, Mapping) else None
    prompt_id = vocab.get(EXPECTED_PROMPT) if isinstance(vocab, Mapping) else None
    newline_id = vocab.get(EXPECTED_NEWLINE_TOKEN) if isinstance(vocab, Mapping) else None
    prompt_ok = prompt_id == EXPECTED_PROMPT_TOKEN_IDS[0]
    newline_ok = newline_id == EXPECTED_NEWLINE_TOKEN_ID

    added_tokens = tokenizer.get("added_tokens") if tokenizer else None
    pad_added_ids = [
        item.get("id")
        for item in added_tokens
        if isinstance(item, Mapping) and item.get("content") == EXPECTED_PAD_TOKEN
    ] if isinstance(added_tokens, list) else []
    eos_added_ids = [
        item.get("id")
        for item in added_tokens
        if isinstance(item, Mapping) and item.get("content") == EXPECTED_EOS_TOKEN
    ] if isinstance(added_tokens, list) else []
    eos_added_ok = EXPECTED_EOS_TOKEN_ID in eos_added_ids
    eos_config = config.get("eos_token") if config else None
    pad_config = config.get("pad_token") if config else None
    decoder = config.get("added_tokens_decoder") if config else None
    decoder_entry = decoder.get(str(EXPECTED_EOS_TOKEN_ID)) if isinstance(decoder, Mapping) else None
    pad_decoder_entry = decoder.get(str(EXPECTED_GENERATION_PAD_TOKEN_ID)) if isinstance(decoder, Mapping) else None
    decoder_content = decoder_entry.get("content") if isinstance(decoder_entry, Mapping) else None
    pad_decoder_content = pad_decoder_entry.get("content") if isinstance(pad_decoder_entry, Mapping) else None
    eos_config_ok = eos_config == EXPECTED_EOS_TOKEN and decoder_content == EXPECTED_EOS_TOKEN
    pad_ok = pad_added_ids == [EXPECTED_GENERATION_PAD_TOKEN_ID] and pad_decoder_content == EXPECTED_PAD_TOKEN
    pad_config_ok = pad_config == EXPECTED_PAD_TOKEN and pad_decoder_content == EXPECTED_PAD_TOKEN

    if tokenizer is None:
        blockers.append("TOKENIZER_FILE_MISSING_OR_INVALID:tokenizer.json")
    if config is None:
        blockers.append("TOKENIZER_FILE_MISSING_OR_INVALID:tokenizer_config.json")
    if not prompt_ok:
        blockers.append("TOKENIZER_PROMPT_TOKEN_ID_MISMATCH")
    if not newline_ok:
        blockers.append("TOKENIZER_NEWLINE_TOKEN_ID_MISMATCH")
    if not pad_ok:
        blockers.append("TOKENIZER_PAD_TOKEN_ID_MISMATCH")
    if not eos_added_ok:
        blockers.append("TOKENIZER_EOS_TOKEN_MISMATCH")
    if not eos_config_ok:
        blockers.append("TOKENIZER_EOS_CONFIG_MISMATCH")
    if not pad_config_ok:
        blockers.append("TOKENIZER_PAD_CONFIG_MISMATCH")
    checks = {
        "paths": {
            "tokenizer": str(tokenizer_path),
            "config": str(config_path),
        },
        "prompt_id": prompt_id,
        "newline_id": newline_id,
        "pad_id": pad_added_ids[0] if len(pad_added_ids) == 1 else None,
        "source_pad_token_id": pad_added_ids[0] if len(pad_added_ids) == 1 else None,
        "effective_generation_pad_token_id": EXPECTED_EFFECTIVE_GENERATION_PAD_TOKEN_ID,
        "pad_added_ids": pad_added_ids,
        "eos_added_ids": eos_added_ids,
        "eos_config": eos_config,
        "pad_config": pad_config,
        "eos_decoder_content": decoder_content,
        "pad_decoder_content": pad_decoder_content,
        "expected": _tokenizer_protocol(exp),
        "eligible": not blockers,
    }
    return checks, blockers


def _check_base(experiment: str | Experiment) -> tuple[dict[str, Any], list[str]]:
    exp = _resolve_experiment(experiment)
    path = exp.base_path
    blockers: list[str] = []
    checks = _base_identity(exp)
    checks["generation_config"] = checks["generation_config_semantics"]
    checks["directory"] = {
        "exists": path.is_dir(),
        "symlink": path.is_symlink(),
        "revision_path_match": exp.base_revision == "workspace-local" or path.name == exp.base_revision,
    }
    if not path.is_dir() or path.is_symlink():
        blockers.append("BASE_DIRECTORY_MISSING_OR_SYMLINK")
    if not checks["directory"]["revision_path_match"]:
        blockers.append("BASE_REVISION_PATH_MISMATCH")
    if not checks["config_semantics"]["matches"]:
        blockers.append("BASE_CONFIG_SEMANTIC_FINGERPRINT_MISMATCH")
    if not checks["generation_config_semantics"]["matches"]:
        blockers.append("BASE_GENERATION_CONFIG_SEMANTICS_MISMATCH")
    for name in BASE_REQUIRED_FILES:
        info = checks["files"][name]
        if not info["exists"]:
            blockers.append(f"BASE_FILE_MISSING:{name}")
        elif not _valid_sha256(info["sha256"]):
            blockers.append(f"BASE_FILE_UNHASHABLE:{name}")
    if not checks["weight_files"]:
        blockers.append("BASE_WEIGHT_FILES_MISSING")
    for info in checks["weight_files"]:
        if not _valid_sha256(info.get("sha256")):
            blockers.append(f"BASE_WEIGHT_FILE_UNHASHABLE:{info.get('name')}")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map") if index else None
        referenced = (
            sorted({name for name in weight_map.values() if isinstance(name, str)})
            if isinstance(weight_map, Mapping)
            else []
        )
        missing = [name for name in referenced if not (path / name).is_file()]
        checks["weight_index"] = {
            "path": str(index_path),
            "valid": isinstance(weight_map, Mapping) and bool(referenced),
            "referenced": referenced,
            "missing": missing,
        }
        if not _valid_sha256(checks["files"]["model.safetensors.index.json"]["sha256"]):
            blockers.append("BASE_FILE_UNHASHABLE:model.safetensors.index.json")
        if not isinstance(weight_map, Mapping) or not referenced:
            blockers.append("BASE_WEIGHT_INDEX_INVALID")
        blockers.extend(f"BASE_WEIGHT_FILE_MISSING:{name}" for name in missing)
    else:
        # Qwen3-0.6B is a valid single-shard snapshot with no index file.
        names = [item["name"] for item in checks["weight_files"]]
        checks["weight_index"] = {
            "path": str(index_path),
            "valid": names == ["model.safetensors"],
            "referenced": names,
            "missing": [],
            "single_shard_without_index": True,
        }
        if names != ["model.safetensors"]:
            blockers.append("BASE_SINGLE_SHARD_LAYOUT_INVALID")
    checks["eligible"] = not blockers
    return checks, blockers


def _check_adapter(experiment: str | Experiment) -> tuple[dict[str, Any], list[str]]:
    exp = _resolve_experiment(experiment)
    path = exp.adapter_path
    blockers: list[str] = []
    checks = _adapter_identity(exp)
    checks["directory"] = {
        "exists": path.is_dir(),
        "symlink": path.is_symlink(),
        "revision_path_match": path.name == exp.adapter_revision,
    }
    if not path.is_dir() or path.is_symlink():
        blockers.append("ADAPTER_DIRECTORY_MISSING_OR_SYMLINK")
    if path.name != exp.adapter_revision:
        blockers.append("ADAPTER_REVISION_PATH_MISMATCH")
    for name, info in checks["files"].items():
        if not info["exists"]:
            blockers.append(f"ADAPTER_FILE_MISSING:{name}")
        elif not _valid_sha256(info["sha256"]):
            blockers.append(f"ADAPTER_FILE_UNHASHABLE:{name}")
    config = _load_json(path / "adapter_config.json")
    targets = config.get("target_modules") if config else None
    target_ok = (
        isinstance(targets, list)
        and all(isinstance(item, str) for item in targets)
        and frozenset(targets) == EXPECTED_TARGET_MODULES
        and len(targets) == len(EXPECTED_TARGET_MODULES)
    )
    config_ok = (
        config is not None
        and config.get("base_model_name_or_path") == f"Qwen/{exp.model_name}"
        and config.get("peft_type") == "LORA"
        and config.get("r") == 32
        and config.get("lora_alpha") == 32
        and target_ok
    )
    checks["adapter_config"] = {
        "base_model_name_or_path": config.get("base_model_name_or_path") if config else None,
        "peft_type": config.get("peft_type") if config else None,
        "r": config.get("r") if config else None,
        "lora_alpha": config.get("lora_alpha") if config else None,
        "target_modules": targets,
        "target_modules_ok": target_ok,
        "eligible": config_ok,
    }
    if not config_ok:
        blockers.append("ADAPTER_CONFIG_CONTRACT_FAILED")
    checks["eligible"] = not blockers
    return checks, blockers


def _check_runtime() -> tuple[dict[str, Any], list[str]]:
    paths = {
        "llamafactory_python": LLAMAFACTORY_PYTHON,
        "bricknet_python": BRICKNET_PYTHON,
        "generate_script": GENERATE_SCRIPT,
        "evaluate_script": EVALUATE_SCRIPT,
    }
    checks = {name: {"path": str(path), "exists": path.is_file()} for name, path in paths.items()}
    blockers = [f"RUNTIME_MISSING:{name}" for name, info in checks.items() if not info["exists"]]
    code = _code_identity()
    checks["code"] = code
    if not _code_identity_complete(code):
        blockers.append("CODE_IDENTITY_INCOMPLETE")
    checks["eligible"] = not blockers
    return checks, blockers


def _check_output_state(experiment: str | Experiment, action: str) -> tuple[dict[str, Any], list[str]]:
    exp = _resolve_experiment(experiment)
    final = _exp_output_root(exp)
    building = _building_root(exp)
    smoke = _smoke_root(exp)
    pilot = _pilot_root(exp)
    blockers: list[str] = []
    checks = {
        "final": {"path": str(final), "absent": not final.exists() and not final.is_symlink()},
        "building": {"path": str(building), "absent": not building.exists() and not building.is_symlink()},
        "smoke": {"path": str(smoke), "absent": not smoke.exists() and not smoke.is_symlink()},
        "pilot": {"path": str(pilot), "absent": not pilot.exists() and not pilot.is_symlink()},
    }
    if action in {"smoke", "pilot", "generate"}:
        name = action
        path_check = checks[name]
        if not path_check["absent"]:
            blockers.append(f"{name.upper()}_OUTPUT_ALREADY_EXISTS")
        if not checks["final"]["absent"]:
            blockers.append("FINAL_OUTPUT_ALREADY_EXISTS")
        if not checks["building"]["absent"]:
            blockers.append("FORMAL_BUILDING_DIRECTORY_ALREADY_EXISTS")
    elif action == "evaluate":
        if not checks["final"]["absent"]:
            blockers.append("FINAL_OUTPUT_ALREADY_EXISTS")
        checks["building"]["exists"] = building.is_dir() and not building.is_symlink()
        if not checks["building"]["exists"]:
            blockers.append("FORMAL_BUILDING_DIRECTORY_MISSING")
        else:
            evaluation_dir = building / EVALUATION_SUBDIR
            checks["evaluation_staging"] = {
                "path": str(evaluation_dir),
                "absent": not evaluation_dir.exists() and not evaluation_dir.is_symlink(),
            }
            if not checks["evaluation_staging"]["absent"]:
                blockers.append("EVALUATION_STAGING_ALREADY_EXISTS")
            for name in EVALUATION_OUTPUTS:
                if (building / name).exists() or (building / name).is_symlink():
                    blockers.append(f"EVALUATION_OUTPUT_ALREADY_EXISTS:{name}")
    elif action == "verify":
        if not checks["final"]["absent"]:
            blockers.append("FINAL_OUTPUT_ALREADY_EXISTS")
        checks["building"]["exists"] = building.is_dir() and not building.is_symlink()
        if not checks["building"]["exists"]:
            blockers.append("FORMAL_BUILDING_DIRECTORY_MISSING")
    checks["eligible"] = not blockers
    return checks, blockers


def _tree_identity(root: Path) -> dict[str, Any]:
    """Return a compact, deterministic identity for a file tree.

    The returned aggregate hashes ``relative POSIX path + file content
    SHA256`` in sorted order.  Per-file records are deliberately not retained
    in run metadata: the inset mesh tree is large, and count/bytes/aggregate
    are sufficient to detect any content or membership drift when recomputed.
    """
    exists = root.is_dir() and not root.is_symlink()
    records: list[tuple[str, int, str | None]] = []
    complete = exists
    if exists:
        try:
            candidates = sorted(
                (item for item in root.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(root).as_posix(),
            )
        except OSError:
            candidates = []
            complete = False
        for item in candidates:
            relative = item.relative_to(root).as_posix()
            try:
                size = item.stat().st_size
            except OSError:
                size = 0
                complete = False
            digest = _sha256(item)
            if not _valid_sha256(digest):
                complete = False
            records.append((relative, size, digest))
    total_bytes = sum(size for _, size, _ in records)
    aggregate = None
    if complete:
        payload = "".join(
            f"{relative}\t{digest}\n" for relative, _, digest in records
        ).encode("utf-8")
        aggregate = hashlib.sha256(payload).hexdigest()
    return {
        "path": str(root),
        "exists": exists,
        "count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": aggregate,
        "complete": complete and _valid_sha256(aggregate),
    }


def _collision_assets_identity() -> dict[str, Any]:
    return {
        "catalog": _file_identity(CATALOG_ROOT, COLLISION_CATALOG_FILES),
        "inset": _tree_identity(BRICKNET_DATA / INSET_ROOT_NAME),
    }


def _collision_assets_identity_complete(identity: Mapping[str, Any]) -> bool:
    catalog = identity.get("catalog")
    if not isinstance(catalog, Mapping):
        return False
    for name in COLLISION_CATALOG_FILES:
        info = catalog.get(name)
        if not isinstance(info, Mapping) or not info.get("exists") or not _valid_sha256(info.get("sha256")):
            return False
    inset = identity.get("inset")
    return (
        isinstance(inset, Mapping)
        and inset.get("exists") is True
        and _is_int(inset.get("count"))
        and inset.get("count") == EXPECTED_COLLISION_MESHES
        and _is_int(inset.get("total_bytes"))
        and inset.get("total_bytes") >= 0
        and inset.get("complete") is True
        and _valid_sha256(inset.get("manifest_sha256"))
    )


def _check_collision_assets() -> tuple[dict[str, Any], list[str]]:
    # The reviewed mesh count and stable inset identity are checked at each
    # gate and carried into run_spec/final manifest so data cannot drift.
    identity = _collision_assets_identity()
    catalog = identity["catalog"]
    inset_identity = identity["inset"]
    checks = {
        "bricknet_data": str(BRICKNET_DATA),
        "inset": inset_identity,
        "catalog": catalog,
        "identity": identity,
    }
    blockers: list[str] = []
    if not inset_identity.get("exists"):
        blockers.append("COLLISION_MESH_DIRECTORY_MISSING")
    elif not _collision_assets_identity_complete(identity):
        if inset_identity.get("count") != EXPECTED_COLLISION_MESHES:
            blockers.append("COLLISION_MESH_COUNT_MISMATCH")
        else:
            blockers.append("COLLISION_MESH_MANIFEST_UNHASHABLE_OR_EMPTY")
    for name, info in catalog.items():
        if not info["exists"]:
            blockers.append(f"CATALOG_FILE_MISSING:{name}")
        elif not _valid_sha256(info.get("sha256")):
            blockers.append(f"CATALOG_FILE_UNHASHABLE:{name}")
    checks["eligible"] = not blockers
    return checks, blockers


def _protocol(experiment: str | Experiment, *, samples: int | None = None) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    count = exp.expected_samples if samples is None else samples
    return {
        "mode": "NS",
        "base_model": exp.model_name,
        "adapter": exp.adapter_name,
        "base_path": str(exp.base_path),
        "base_revision": exp.base_revision,
        "adapter_path": str(exp.adapter_path),
        "adapter_revision": exp.adapter_revision,
        "prompt": EXPECTED_PROMPT,
        "raw_prompt": True,
        "chat_template": False,
        "num_samples": count,
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "temperature": EXPECTED_TEMPERATURE,
        "top_k": EXPECTED_TOP_K,
        "top_p": EXPECTED_TOP_P,
        "dtype": EXPECTED_DTYPE,
        "stop_after_newlines": EXPECTED_STOP_AFTER_NEWLINES,
        "seed": EXPECTED_SEED,
        "generation_config": _generation_config_protocol(),
    }


def _code_identity_paths() -> dict[str, Path]:
    return {
        "launcher": Path(__file__).resolve(),
        "generator": GENERATE_SCRIPT,
        "evaluator": EVALUATE_SCRIPT,
        "score": BRICKNET_ROOT / "src/bricknet/score.py",
        "collision": BRICKNET_ROOT / "src/bricknet/collision.py",
        "core": BRICKNET_ROOT / "src/bricknet/core.py",
        "graph": BRICKNET_ROOT / "src/bricknet/graph.py",
        "tree": BRICKNET_ROOT / "src/bricknet/tree.py",
    }


def _code_identity() -> dict[str, Any]:
    paths = _code_identity_paths()
    return {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }


def _code_identity_complete(identity: Any) -> bool:
    """Require every recorded code dependency to exist and have a real hash."""
    paths = _code_identity_paths()
    if not isinstance(identity, Mapping) or set(identity) != set(paths):
        return False
    return all(
        isinstance(item, Mapping)
        and item.get("path") == str(paths[name])
        and _valid_sha256(item.get("sha256"))
        for name, item in identity.items()
    )


def _environment(gpus: str | None) -> dict[str, Any]:
    selectors = _parse_gpu_selector(gpus)
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "launcher_python": str(LLAMAFACTORY_PYTHON),
        "evaluator_python": str(BRICKNET_PYTHON),
        "gpus": list(selectors),
        "gpu": gpus,
        "world_size": len(selectors),
        "parent_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _token_metadata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise GateError("cannot summarize token metadata for zero rows")
    counts = [row["generated_token_count"] for row in rows]
    newlines = [row["generated_newline_tokens"] for row in rows]
    stop_count = sum(row["stop_target_reached"] for row in rows)
    max_count = sum(row["hit_max_new_tokens"] for row in rows)
    # The paper's full-length gate is the newline stop contract.  A sample
    # that reaches the target on token 4096 may also report
    # ``hit_max_new_tokens=true``; that flag is diagnostic and must not by
    # itself turn an otherwise valid sample into a failure.
    full_count = sum(
        row["generated_newline_tokens"] == EXPECTED_STOP_AFTER_NEWLINES
        and row["stop_target_reached"]
        for row in rows
    )
    return {
        "samples": len(rows),
        "max_new_tokens": EXPECTED_MAX_NEW_TOKENS,
        "stop_after_newlines": EXPECTED_STOP_AFTER_NEWLINES,
        "generated_token_count": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "full_length_count": full_count,
            "full_length_rate": full_count / len(counts),
        },
        "generated_newline_tokens": {
            "min": min(newlines),
            "max": max(newlines),
            "mean": sum(newlines) / len(newlines),
        },
        "stop_target_reached_count": stop_count,
        "stop_target_reached_rate": stop_count / len(rows),
        "hit_max_new_tokens_count": max_count,
        "hit_max_new_tokens_rate": max_count / len(rows),
        # These aliases make the full-length gate easy to consume without
        # losing the source field names emitted by BrickNet's generator.
        "token_full_length_count": full_count,
        "token_full_length_rate": full_count / len(rows),
    }


def _validate_token_metadata(
    rows: Sequence[Mapping[str, Any]],
    expected_samples: int,
    *,
    require_full_length: bool = False,
) -> dict[str, Any]:
    if len(rows) != expected_samples:
        raise GateError(f"expected {expected_samples} generated rows, found {len(rows)}")
    for index, row in enumerate(rows):
        prefix = f"generated row {index}"
        for field in (
            "generated_token_count",
            "generated_newline_tokens",
            "stop_target_reached",
            "hit_max_new_tokens",
        ):
            if field not in row:
                raise GateError(f"{prefix} is missing {field}")
        token_count = row["generated_token_count"]
        newline_count = row["generated_newline_tokens"]
        if not _is_int(token_count) or not 0 <= token_count <= EXPECTED_MAX_NEW_TOKENS:
            raise GateError(f"{prefix} has invalid generated_token_count")
        if not _is_int(newline_count) or not 0 <= newline_count <= token_count:
            raise GateError(f"{prefix} has invalid generated_newline_tokens")
        if not isinstance(row["stop_target_reached"], bool):
            raise GateError(f"{prefix} has non-boolean stop_target_reached")
        if not isinstance(row["hit_max_new_tokens"], bool):
            raise GateError(f"{prefix} has non-boolean hit_max_new_tokens")
        if row["stop_target_reached"] and newline_count != EXPECTED_STOP_AFTER_NEWLINES:
            raise GateError(f"{prefix} claims stop target with {newline_count} instead of {EXPECTED_STOP_AFTER_NEWLINES} newlines")
        if not row["stop_target_reached"] and newline_count >= EXPECTED_STOP_AFTER_NEWLINES:
            raise GateError(f"{prefix} reaches stop target but flag is false")
        if row["hit_max_new_tokens"] != (token_count == EXPECTED_MAX_NEW_TOKENS):
            raise GateError(f"{prefix} hit_max_new_tokens disagrees with token count")
        if require_full_length and not (
            newline_count == EXPECTED_STOP_AFTER_NEWLINES
            and row["stop_target_reached"]
        ):
            raise GateError(f"{prefix} does not satisfy the strict token full-length gate")
    return _token_metadata_summary(rows)


def _validate_raw_rows(
    path_or_rows: Path | str | Sequence[Mapping[str, Any]], expected_samples: int
) -> list[dict[str, Any]]:
    if isinstance(path_or_rows, (Path, str)):
        rows = _read_jsonl(Path(path_or_rows))
    else:
        rows = [dict(row) for row in path_or_rows]
    if len(rows) != expected_samples:
        raise GateError(f"expected {expected_samples} generated rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if "path" in row:
            raise GateError(f"generated row {index} contains forbidden path field")
        if row.get("id") != index or not _is_int(row.get("id")):
            raise GateError(f"generated row {index} has invalid ordered integer id")
        if row.get("sample") != 0 or not _is_int(row.get("sample")):
            raise GateError(f"generated row {index} has invalid sample")
        if row.get("source") != "" or row.get("caption") != "":
            raise GateError(f"generated row {index} requires empty source and caption")
        text_value = row.get("text")
        if not isinstance(text_value, str) or not text_value.strip() or not text_value.startswith(EXPECTED_PROMPT):
            raise GateError(f"generated row {index} has invalid raw prompt/text")
    _validate_token_metadata(rows, expected_samples)
    return rows


def _validate_scored_rows(
    raw_rows: Sequence[Mapping[str, Any]], scored_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(scored_rows) != len(raw_rows):
        raise GateError(f"expected {len(raw_rows)} scored rows, found {len(scored_rows)}")
    for index, (raw, scored) in enumerate(zip(raw_rows, scored_rows)):
        required = ("id", "sample", "text", "n_actions", "invalid", "collisions")
        missing = [field for field in required if field not in scored]
        if missing:
            raise GateError(f"scored row {index} is missing required fields: {missing}")
        if scored["id"] != raw["id"] or scored["sample"] != raw["sample"]:
            raise GateError(f"scored row {index} id/sample mismatch")
        if scored["text"] != raw["text"]:
            raise GateError(f"scored row {index} text mismatch")
        if not _is_int(scored["n_actions"]) or scored["n_actions"] < 0:
            raise GateError(f"scored row {index} has invalid n_actions")
        invalid = scored["invalid"]
        if invalid is not None and (not _is_int(invalid) or invalid < 0):
            raise GateError(f"scored row {index} has invalid invalid index")
        collisions = scored["collisions"]
        if not isinstance(collisions, list) or any(not _is_int(item) or item < 0 for item in collisions):
            raise GateError(f"scored row {index} has invalid collisions")
        if "connectivity" in scored:
            value = scored["connectivity"]
            if not ((isinstance(value, bool)) or (_is_number(value) and 0 <= float(value) <= 1)):
                raise GateError(f"scored row {index} has invalid connectivity")


def _aggregate_scored_metrics(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute parsable/clean/connectivity/collision means from rows."""
    if not scored_rows:
        raise GateError("cannot aggregate an empty scored bundle")
    parsable_count = sum(row["invalid"] is None for row in scored_rows)
    clean_count = sum(row["invalid"] is None and not row["collisions"] for row in scored_rows)
    failures: list[int] = []
    for row in scored_rows:
        invalid = row["n_actions"] if row["invalid"] is None else row["invalid"]
        failures.append(min([invalid, *row["collisions"]]))
    count = len(scored_rows)
    # Connectivity is a continuous prefix metric in the paper evaluator:
    # complete parses contribute ``n_actions`` and parse failures contribute
    # their first invalid action index.  The parsable rate remains a separate
    # binary metric.  Collision prefix is the minimum of that connectivity
    # prefix and every collision index (including parse failure), rather than
    # a collision-only mean.
    connectivity_mean = sum(
        row["n_actions"] if row["invalid"] is None else row["invalid"]
        for row in scored_rows
    ) / count
    collision_mean = sum(failures) / count
    return {
        "samples": count,
        "parsable_count": parsable_count,
        "parsable_rate": parsable_count / count,
        "clean_count": clean_count,
        "clean_rate": clean_count / count,
        "connectivity_count": parsable_count,
        "connectivity_mean": connectivity_mean,
        "connectivity_rate": parsable_count / count,
        "mean_actions_before_connectivity_failure": connectivity_mean,
        "collision_mean": collision_mean,
        "mean_actions_before_collision_failure": collision_mean,
        "collision": collision_mean,
        "mean_actions_before_first_failure": collision_mean,
    }


# Short aliases retained for callers of earlier launcher prototypes.
_recompute_metrics = _aggregate_scored_metrics


def _metric_at(metrics: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = metrics
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _validate_metrics(metrics: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    checks = (
        ("parsable_rate", ("parsable_rate", "fully_parsable_rate", "parsable.rate"), expected["parsable_rate"]),
        ("clean_rate", ("clean_rate", "parsable_and_collision_free_rate", "clean.rate"), expected["clean_rate"]),
        (
            "connectivity_mean",
            (
                "connectivity_mean",
                "mean_actions_before_connectivity_failure",
                "connectivity",
                "connectivity.mean",
            ),
            expected["connectivity_mean"],
        ),
        (
            "collision_mean",
            (
                "collision_mean",
                "mean_actions_before_collision_failure",
                "mean_actions_before_first_failure",
                "collision",
            ),
            expected["collision_mean"],
        ),
    )
    for name, aliases, target in checks:
        actual = _metric_at(metrics, *aliases)
        if actual is None or not _is_number(actual) or not math.isclose(float(actual), float(target), rel_tol=1e-12, abs_tol=1e-12):
            raise GateError(f"evaluation metric mismatch for {name}: expected {target!r}, found {actual!r}")
    count_checks = (
        ("parsable_count", ("parsable_count", "fully_parsable", "parsable.count"), expected["parsable_count"]),
        ("clean_count", ("clean_count", "parsable_and_collision_free", "clean.count"), expected["clean_count"]),
    )
    for name, aliases, target in count_checks:
        actual = _metric_at(metrics, *aliases)
        if actual is not None and actual != target:
            raise GateError(f"evaluation metric mismatch for {name}: expected {target!r}, found {actual!r}")
    sample_value = _metric_at(metrics, "samples", "sample_count")
    if sample_value is not None and sample_value != expected["samples"]:
        raise GateError("evaluation metric sample count mismatch")


def _validate_evaluation_manifest(
    manifest: Mapping[str, Any], *, expected_samples: int, stable_root: Path
) -> None:
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise GateError("evaluation manifest schema mismatch")
    if manifest.get("status") != "complete":
        raise GateError("evaluation manifest is not complete")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise GateError("evaluation manifest lacks evaluation contract")
    if evaluation.get("expected_samples") != expected_samples:
        raise GateError("evaluation manifest expected_samples mismatch")
    if evaluation.get("expected_prompt") != EXPECTED_PROMPT:
        raise GateError("evaluation manifest expected_prompt mismatch")
    if evaluation.get("require_generated_newlines") != EXPECTED_STOP_AFTER_NEWLINES:
        raise GateError("evaluation manifest newline protocol mismatch")
    if evaluation.get("token_level_full_length_gate") is not True:
        raise GateError("evaluation manifest token full-length gate is not enabled")
    input_info = manifest.get("input")
    scored_info = manifest.get("scored")
    if not isinstance(input_info, Mapping) or not isinstance(scored_info, Mapping):
        raise GateError("evaluation manifest lacks input/scored identity")
    if input_info.get("path") != str(stable_root / RAW_NAME):
        raise GateError("evaluation manifest input path mismatch")
    if scored_info.get("path") != str(stable_root / SCORED_NAME):
        raise GateError("evaluation manifest scored path mismatch")
    if input_info.get("rows") != expected_samples or scored_info.get("rows") != expected_samples:
        raise GateError("evaluation manifest row count mismatch")


def _validate_evaluation_bundle(root: Path, raw_rows: Sequence[Mapping[str, Any]], *, stable_root: Path) -> dict[str, Any]:
    scored_path = root / SCORED_NAME
    metrics_path = root / METRICS_NAME
    report_path = root / REPORT_NAME
    manifest_path = root / EVALUATION_MANIFEST_NAME
    scored_rows = _read_jsonl(scored_path)
    _validate_scored_rows(raw_rows, scored_rows)
    metrics = _load_json(metrics_path)
    manifest = _load_json(manifest_path)
    if metrics is None or manifest is None or not report_path.is_file():
        raise GateError("evaluation metrics/report/manifest missing or invalid")
    if metrics.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise GateError("unexpected evaluator schema")
    generation_gate = metrics.get("generation_gate")
    if not isinstance(generation_gate, Mapping):
        raise GateError("evaluation generation gate is missing")
    if generation_gate.get("enabled") is not True:
        raise GateError("evaluation generation gate is not enabled")
    if generation_gate.get("unit") != "generated_token":
        raise GateError("evaluation generation gate unit mismatch")
    if generation_gate.get("required_generated_newlines") != EXPECTED_STOP_AFTER_NEWLINES:
        raise GateError("evaluation generation gate target mismatch")
    if (
        generation_gate.get("passed_count") != len(raw_rows)
        or generation_gate.get("failed_count") != 0
    ):
        raise GateError("evaluation generation gate does not cover all samples")
    expected = _aggregate_scored_metrics(scored_rows)
    _validate_metrics(metrics, expected)
    metric_input_hash = metrics.get("input_sha256")
    metric_scored_hash = metrics.get("scored_sha256")
    if metric_input_hash not in (_sha256(root / RAW_NAME), _sha256(stable_root / RAW_NAME)):
        raise GateError("evaluation metrics input hash mismatch")
    if metric_scored_hash not in (_sha256(scored_path), _sha256(stable_root / SCORED_NAME)):
        raise GateError("evaluation metrics scored hash mismatch")
    if manifest.get("metrics") is not None and manifest.get("metrics") != metrics:
        raise GateError("evaluation manifest metrics content mismatch")
    _validate_evaluation_manifest(manifest, expected_samples=len(raw_rows), stable_root=stable_root)
    input_hash = _metric_at(manifest, "input.sha256", "sha256.input")
    scored_hash = _metric_at(manifest, "scored.sha256", "sha256.scored")
    if input_hash is None or input_hash != _sha256(root / RAW_NAME):
        raise GateError("evaluation manifest input hash mismatch")
    if scored_hash is None or scored_hash != _sha256(scored_path):
        raise GateError("evaluation manifest scored hash mismatch")
    metrics_hash = _sha256(metrics_path)
    report_hash = _sha256(report_path)
    if manifest.get("metrics_path") != str(stable_root / METRICS_NAME) or manifest.get("metrics_sha256") != metrics_hash:
        raise GateError("evaluation manifest metrics hash/path mismatch")
    if manifest.get("report_path") != str(stable_root / REPORT_NAME) or manifest.get("report_sha256") != report_hash:
        raise GateError("evaluation manifest report hash/path mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GateError("evaluation manifest artifact hashes are missing")
    expected_artifacts = {
        "input": (stable_root / RAW_NAME, _sha256(root / RAW_NAME)),
        "scored": (stable_root / SCORED_NAME, _sha256(scored_path)),
        "metrics": (stable_root / METRICS_NAME, metrics_hash),
        "report": (stable_root / REPORT_NAME, report_hash),
    }
    for name, (expected_path, expected_hash) in expected_artifacts.items():
        item = artifacts.get(name)
        if not isinstance(item, Mapping) or item.get("path") != str(expected_path) or item.get("sha256") != expected_hash:
            raise GateError(f"evaluation manifest artifact hash/path mismatch: {name}")
    return {
        "metrics": metrics,
        "manifest": manifest,
        "scored_rows": len(scored_rows),
        "recomputed": expected,
        "token_metadata": _token_metadata_summary(raw_rows),
    }


def _rewrite_evaluation_manifest_paths(evaluation_dir: Path, stable_root: Path) -> None:
    path = evaluation_dir / EVALUATION_MANIFEST_NAME
    manifest = _load_json(path)
    if manifest is None:
        raise GateError(f"invalid evaluation manifest: {path}")
    input_info = manifest.get("input")
    scored_info = manifest.get("scored")
    if isinstance(input_info, dict):
        input_info["path"] = str(stable_root / RAW_NAME)
    if isinstance(scored_info, dict):
        scored_info["path"] = str(stable_root / SCORED_NAME)
    manifest["output_dir"] = str(stable_root)
    manifest["metrics_path"] = str(stable_root / METRICS_NAME)
    manifest["report_path"] = str(stable_root / REPORT_NAME)
    manifest["metrics_sha256"] = _sha256(evaluation_dir / METRICS_NAME)
    manifest["report_sha256"] = _sha256(evaluation_dir / REPORT_NAME)
    manifest["artifacts"] = {
        "input": {"path": str(stable_root / RAW_NAME), "sha256": _sha256(evaluation_dir.parent / RAW_NAME)},
        "scored": {"path": str(stable_root / SCORED_NAME), "sha256": _sha256(evaluation_dir / SCORED_NAME)},
        "metrics": {"path": str(stable_root / METRICS_NAME), "sha256": _sha256(evaluation_dir / METRICS_NAME)},
        "report": {"path": str(stable_root / REPORT_NAME), "sha256": _sha256(evaluation_dir / REPORT_NAME)},
    }
    _replace_json(path, manifest)


def _move_evaluation_bundle(building: Path, *, stable_root: Path) -> None:
    evaluation_dir = building / EVALUATION_SUBDIR
    if not evaluation_dir.is_dir() or evaluation_dir.is_symlink():
        raise GateError(f"evaluator output directory missing: {evaluation_dir}")
    entries = {path.name for path in evaluation_dir.iterdir()}
    if entries != set(EVALUATION_OUTPUTS):
        raise GateError(f"evaluator output set mismatch: expected {sorted(EVALUATION_OUTPUTS)}, found {sorted(entries)}")
    if any(not (evaluation_dir / name).is_file() or (evaluation_dir / name).is_symlink() for name in EVALUATION_OUTPUTS):
        raise GateError("evaluator outputs must be regular files")
    _rewrite_evaluation_manifest_paths(evaluation_dir, stable_root)
    for name in EVALUATION_OUTPUTS:
        destination = building / name
        if destination.exists() or destination.is_symlink():
            raise GateError(f"refusing to overwrite evaluation output: {destination}")
        _rename_noreplace(evaluation_dir / name, destination)
    evaluation_dir.rmdir()


def _generation_command(
    experiment: str | Experiment,
    output_path: Path | str,
    *,
    samples: int | None = None,
    gpus: str | None = None,
    batch_size: int | None = None,
) -> list[str]:
    exp = _resolve_experiment(experiment)
    count = exp.expected_samples if samples is None else samples
    if count not in {exp.expected_samples, SMOKE_SAMPLES, PILOT_SAMPLES}:
        raise GateError("sample count is not one of the frozen formal/smoke/pilot counts")
    effective_batch_size = exp.batch_size if batch_size is None else batch_size
    if effective_batch_size != exp.batch_size:
        raise GateError("batch size is frozen per experiment and cannot be customized")
    selectors = _parse_gpu_selector(gpus)
    if gpus is not None and not selectors:
        raise GateError("invalid GPU selector")
    if len(selectors) > 1:
        # ``generate.py`` uses Accelerate's process index to place one full
        # model copy per process.  Merely exporting CUDA_VISIBLE_DEVICES with
        # several IDs would still start world-size=1 and make the manifest
        # misleading, so multi-GPU commands explicitly launch Accelerate.
        command = [
            str(LLAMAFACTORY_PYTHON),
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_machines",
            "1",
            "--mixed_precision",
            "no",
            "--dynamo_backend",
            "no",
            "--main_process_port",
            "0",
            "--num_processes",
            str(len(selectors)),
            "--gpu_ids",
            "all",
            str(GENERATE_SCRIPT),
        ]
    else:
        command = [str(LLAMAFACTORY_PYTHON), str(GENERATE_SCRIPT)]
    command.extend([
        "--model",
        str(exp.base_path),
        "--lora",
        str(exp.adapter_path),
        "--output",
        str(Path(output_path)),
        "--prompt",
        EXPECTED_PROMPT,
        "--num_samples",
        str(count),
        "--batch_size",
        str(effective_batch_size),
        "--max_new_tokens",
        str(EXPECTED_MAX_NEW_TOKENS),
        "--temperature",
        str(EXPECTED_TEMPERATURE),
        "--top_k",
        str(EXPECTED_TOP_K),
        "--top_p",
        str(EXPECTED_TOP_P),
        "--dtype",
        EXPECTED_DTYPE,
        "--stop_after_newlines",
        str(EXPECTED_STOP_AFTER_NEWLINES),
    ])
    return command


def _evaluation_command(
    input_path: Path | str,
    output_dir: Path | str,
    *,
    expected_samples: int,
) -> list[str]:
    if expected_samples <= 0:
        raise GateError("expected_samples must be positive")
    return [
        str(BRICKNET_PYTHON),
        str(EVALUATE_SCRIPT),
        "--input",
        str(Path(input_path)),
        "--output-dir",
        str(Path(output_dir)),
        "--require-generated-newlines",
        str(EXPECTED_STOP_AFTER_NEWLINES),
        "--expected-samples",
        str(expected_samples),
        "--expected-prompt",
        EXPECTED_PROMPT,
        "--workers",
        "1",
        "--execute",
    ]


def _command_env(gpus: str | None) -> dict[str, str]:
    selectors = _parse_gpu_selector(gpus)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BRICKNET_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["BRICKNET_DATA"] = str(BRICKNET_DATA)
    env["BRICKNET_CATALOG"] = "v1"
    if selectors:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(selectors)
    return env


def _run_command(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], label: str) -> None:
    print(f"[{label}]\n$ {shlex.join([str(item) for item in command])}", flush=True)
    result = subprocess.run([str(item) for item in command], cwd=str(cwd), env=dict(env), check=False)
    if result.returncode:
        raise GateError(f"{label} failed with exit code {result.returncode}")


def _create_staging(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise GateError(f"refusing to reuse existing staging directory: {root}")
    if root.parent.exists() and not root.parent.is_dir():
        raise GateError(f"staging parent is not a directory: {root.parent}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()


def _run_spec(
    experiment: str | Experiment,
    *,
    scope: str,
    status: str,
    gpus: str | None,
    samples: int | None = None,
) -> dict[str, Any]:
    exp = _resolve_experiment(experiment)
    count = exp.expected_samples if samples is None else samples
    environment = _environment(gpus)
    scope_root = _scope_root(exp, scope)
    generation_argv = _generation_command(
        exp,
        scope_root / RAW_NAME,
        samples=count,
        gpus=gpus,
    )
    evaluation_argv = _evaluation_command(
        scope_root / RAW_NAME,
        scope_root / EVALUATION_SUBDIR,
        expected_samples=count,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp.identity,
        "experiment_key": exp.key,
        "status": status,
        "scope": scope,
        "created_at": datetime.now(UTC).isoformat(),
        "base": _base_identity(exp),
        "adapter": _adapter_identity(exp),
        "collision_assets": _collision_assets_identity(),
        "protocol": _protocol(exp, samples=count),
        "tokenizer": _tokenizer_protocol(exp),
        "evaluation": {
            "evaluator": str(EVALUATE_SCRIPT),
            "expected_samples": count,
            "expected_prompt": EXPECTED_PROMPT,
            "require_generated_newlines": EXPECTED_STOP_AFTER_NEWLINES,
            "collision": True,
        },
        "world_size": environment["world_size"],
        "gpus": environment["gpus"],
        "outputs": {
            "final_root": str(_exp_output_root(exp)),
            "staging_root": str(_scope_root(exp, scope)),
        },
        "code": _code_identity(),
        "environment": environment,
        "commands": {
            "generation": {"argv": generation_argv},
            "evaluation": {"argv": evaluation_argv},
        },
    }


def _validate_run_spec_environment(spec: Mapping[str, Any]) -> None:
    """Ensure the recorded GPU selector and world size agree internally."""
    world_size = spec.get("world_size")
    gpus = spec.get("gpus")
    environment = spec.get("environment")
    if not _is_int(world_size) or world_size < 0:
        raise GateError("run_spec world_size is invalid")
    if not isinstance(gpus, list) or any(not isinstance(value, str) for value in gpus):
        raise GateError("run_spec gpus is invalid")
    if len(set(gpus)) != len(gpus) or any(not _parse_gpu_selector(value) for value in gpus):
        raise GateError("run_spec gpus contains an invalid selector")
    if not isinstance(environment, Mapping):
        raise GateError("run_spec environment is missing")
    expected_gpu = ",".join(gpus) if gpus else None
    if (
        world_size != len(gpus)
        or environment.get("world_size") != world_size
        or environment.get("gpus") != gpus
        or environment.get("gpu") != expected_gpu
    ):
        raise GateError("run_spec world-size/GPU provenance mismatch")


def _validate_run_spec(
    spec: Mapping[str, Any], experiment: str | Experiment, *, scope: str | None = None
) -> None:
    exp = _resolve_experiment(experiment)
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise GateError("run_spec schema mismatch")
    if spec.get("experiment_id") != exp.identity or spec.get("experiment_key") != exp.key:
        raise GateError("run_spec experiment identity mismatch")
    spec_scope = spec.get("scope")
    if spec_scope not in {"formal", "smoke", "pilot"}:
        raise GateError("run_spec scope is invalid")
    if scope is not None and spec_scope != scope:
        raise GateError("run_spec scope mismatch")
    protocol = spec.get("protocol")
    expected = _protocol(exp, samples=exp.expected_samples if scope == "formal" else None)
    if not isinstance(protocol, Mapping):
        raise GateError("run_spec protocol is missing")
    # Small diagnostic runs use a different N, but all other protocol fields
    # must remain exactly the frozen arm contract.
    actual_protocol = dict(protocol)
    actual_protocol.pop("num_samples", None)
    expected_protocol = dict(expected)
    expected_protocol.pop("num_samples", None)
    if actual_protocol != expected_protocol:
        raise GateError("run_spec protocol drift")
    count = protocol.get("num_samples")
    allowed_counts = {exp.expected_samples, SMOKE_SAMPLES, PILOT_SAMPLES}
    if count not in allowed_counts:
        raise GateError("run_spec sample count is not frozen")
    expected_scope_counts = {
        "formal": exp.expected_samples,
        "smoke": SMOKE_SAMPLES,
        "pilot": PILOT_SAMPLES,
    }
    if count != expected_scope_counts[spec_scope]:
        if spec_scope == "formal":
            raise GateError("formal run_spec sample count must equal the experiment N")
        raise GateError(f"{spec_scope} run_spec sample count is not frozen")
    _validate_run_spec_environment(spec)
    commands = spec.get("commands")
    expected_gpus = ",".join(spec["gpus"]) if spec["gpus"] else None
    expected_commands = {
        "generation": {
            "argv": _generation_command(
                exp,
                _scope_root(exp, spec_scope) / RAW_NAME,
                samples=count,
                gpus=expected_gpus,
            )
        },
        "evaluation": {
            "argv": _evaluation_command(
                _scope_root(exp, spec_scope) / RAW_NAME,
                _scope_root(exp, spec_scope) / EVALUATION_SUBDIR,
                expected_samples=count,
            )
        },
    }
    if commands != expected_commands:
        raise GateError("run_spec command argv drift")
    if spec.get("tokenizer") != _tokenizer_protocol(exp):
        raise GateError("run_spec tokenizer protocol drift")
    base = spec.get("base")
    adapter = spec.get("adapter")
    current_base = _base_identity(exp)
    current_adapter = _adapter_identity(exp)
    if not isinstance(base, Mapping) or base != current_base:
        raise GateError("run_spec base identity mismatch")
    if not isinstance(base, Mapping) or not _base_identity_hashes_complete(base):
        raise GateError("run_spec base identity hashes are incomplete")
    if not isinstance(adapter, Mapping) or adapter != current_adapter:
        raise GateError("run_spec adapter identity mismatch")
    if not isinstance(adapter, Mapping) or not _adapter_identity_hashes_complete(adapter):
        raise GateError("run_spec adapter identity hashes are incomplete")
    collision_assets = spec.get("collision_assets")
    current_collision_assets = _collision_assets_identity()
    if not isinstance(collision_assets, Mapping) or collision_assets != current_collision_assets:
        raise GateError("run_spec collision asset identity mismatch")
    if not _collision_assets_identity_complete(collision_assets):
        raise GateError("run_spec collision asset identity is incomplete")
    recorded_code = spec.get("code")
    if not _code_identity_complete(recorded_code):
        raise GateError("run_spec code identity is incomplete")
    current_code = _code_identity()
    if not _code_identity_complete(current_code):
        raise GateError("current code identity is incomplete")
    if recorded_code != current_code:
        raise GateError("run_spec code identity drift")
    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("expected_samples") != count or evaluation.get("expected_prompt") != EXPECTED_PROMPT or evaluation.get("require_generated_newlines") != EXPECTED_STOP_AFTER_NEWLINES:
        raise GateError("run_spec evaluation protocol drift")


def _check_building_entries(building: Path, *, promotable: bool = False) -> None:
    if not building.is_dir() or building.is_symlink():
        raise GateError(f"formal building directory is missing or symlinked: {building}")
    expected = set(PROMOTABLE_OUTPUTS if promotable else STAGED_OUTPUTS)
    entries = {path.name for path in building.iterdir()}
    if entries != expected:
        missing = sorted(expected - entries)
        unexpected = sorted(entries - expected)
        raise GateError(f"building entry set mismatch: missing={missing}, unexpected={unexpected}")
    bad = [name for name in expected if not (building / name).is_file() or (building / name).is_symlink()]
    if bad:
        raise GateError(f"building entries must be regular files: {sorted(bad)}")


def preflight(
    action: str,
    experiment: str | Experiment,
    gpus: str | None = None,
) -> dict[str, Any]:
    """Perform read-only checks and return a machine-readable report."""
    exp = _resolve_experiment(experiment)
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    for function, key in (
        (_check_base, "base"),
        (_check_adapter, "adapter"),
        (_check_tokenizer, "tokenizer"),
        (lambda _exp: _check_runtime(), "runtime"),
        (lambda _exp: _check_collision_assets(), "collision_assets"),
    ):
        item, current = function(exp)  # type: ignore[arg-type]
        checks[key] = item
        blockers.extend(current)
    if action in {"preflight", "smoke", "pilot", "generate"}:
        selectors = _parse_gpu_selector(gpus)
        checks["gpu"] = {"value": gpus, "selectors": list(selectors), "world_size": len(selectors), "eligible": bool(selectors)}
        if not selectors:
            blockers.append("GPU_SELECTOR_REQUIRED_OR_INVALID")
    if action in {"smoke", "pilot", "generate", "evaluate", "verify"}:
        output_checks, output_blockers = _check_output_state(exp, action)
        checks["outputs"] = output_checks
        blockers.extend(output_blockers)
    # For evaluate and verify, inspect the existing formal run_spec without
    # executing anything.  This catches wrong-arm pairing before subprocess.
    if action in {"evaluate", "verify"}:
        building = _building_root(exp)
        spec = _load_json(building / RUN_SPEC_NAME)
        checks["run_spec"] = {"exists": spec is not None, "status": spec.get("status") if spec else None}
        required_status = "generated" if action == "evaluate" else "evaluated"
        if spec is None or spec.get("status") != required_status:
            blockers.append("RUN_SPEC_STATUS_OR_IDENTITY_INVALID")
        else:
            try:
                _validate_run_spec(spec, exp, scope="formal")
            except GateError as exc:
                checks["run_spec"]["error"] = str(exc)
                blockers.append("RUN_SPEC_PROTOCOL_OR_IDENTITY_MISMATCH")
    deduped = list(dict.fromkeys(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "experiment": exp.identity,
        "experiment_key": exp.key,
        "eligible": not deduped,
        "blockers": deduped,
        "checks": checks,
        "output_root": str(_exp_output_root(exp)),
        "building_root": str(_building_root(exp)),
        "smoke_root": str(_smoke_root(exp)),
        "pilot_root": str(_pilot_root(exp)),
    }


def _validate_generation_artifact(exp: Experiment, root: Path, *, samples: int) -> list[dict[str, Any]]:
    rows = _validate_raw_rows(root / RAW_NAME, samples)
    return rows


def _run_generation(
    experiment: str | Experiment,
    *,
    scope: str,
    gpus: str,
    samples: int,
) -> Path:
    exp = _resolve_experiment(experiment)
    root = _scope_root(exp, scope)
    _create_staging(root)
    spec_path = root / RUN_SPEC_NAME
    _write_json_new(spec_path, _run_spec(exp, scope=scope, status="planned", gpus=gpus, samples=samples))
    _update_run_spec(spec_path, status="running", started_at=datetime.now(UTC).isoformat())
    raw_path = root / RAW_NAME
    try:
        _run_command(
            _generation_command(exp, raw_path, samples=samples, gpus=gpus),
            cwd=BRICKNET_ROOT,
            env=_command_env(gpus),
            label=f"{scope} generation {exp.key}",
        )
        rows = _validate_generation_artifact(exp, root, samples=samples)
        _update_run_spec(
            spec_path,
            status="generated",
            generated_at=datetime.now(UTC).isoformat(),
            raw_rows=len(rows),
            raw_sha256=_sha256(raw_path),
            token_metadata=_token_metadata_summary(rows),
        )
    except Exception as exc:
        try:
            _update_run_spec(spec_path, status="generation_failed", error=str(exc))
        except GateError:
            pass
        raise
    return raw_path


def _run_evaluation(exp: Experiment, *, gpus: str | None, root: Path, samples: int) -> dict[str, Any]:
    spec_path = root / RUN_SPEC_NAME
    spec = _load_json(spec_path)
    if spec is None or spec.get("status") != "generated":
        raise GateError("run_spec is not in generated state")
    _validate_run_spec(spec, exp, scope="formal" if root == _building_root(exp) else None)
    raw_path = root / RAW_NAME
    raw_rows = _validate_raw_rows(raw_path, samples)
    # Formal evaluation invokes BrickNet's strict token-level gate.  Mirror it
    # before starting that subprocess so a malformed artifact cannot consume
    # expensive scoring work; smoke/pilot use the same gate for parity.
    _validate_token_metadata(raw_rows, samples, require_full_length=True)
    stable_root = _exp_output_root(exp) if root == _building_root(exp) else root
    evaluation_dir = root / EVALUATION_SUBDIR
    if evaluation_dir.exists() or evaluation_dir.is_symlink():
        raise GateError(f"evaluation staging already exists: {evaluation_dir}")
    try:
        _run_command(
            _evaluation_command(raw_path, evaluation_dir, expected_samples=samples),
            cwd=BRICKNET_ROOT,
            env=_command_env(gpus),
            label=f"evaluation {exp.key}",
        )
        _move_evaluation_bundle(root, stable_root=stable_root)
        bundle = _validate_evaluation_bundle(
            root,
            raw_rows,
            stable_root=stable_root,
        )
        _update_run_spec(
            spec_path,
            status="evaluated",
            evaluated_at=datetime.now(UTC).isoformat(),
            scored_rows=bundle["scored_rows"],
            scored_sha256=_sha256(root / SCORED_NAME),
            metrics_sha256=_sha256(root / METRICS_NAME),
            evaluation_manifest_sha256=_sha256(root / EVALUATION_MANIFEST_NAME),
            outputs={
                "final_root": str(_exp_output_root(exp)),
                "staging_root": str(root),
                "raw": str(stable_root / RAW_NAME),
                "scored": str(stable_root / SCORED_NAME),
                "metrics": str(stable_root / METRICS_NAME),
                "report": str(stable_root / REPORT_NAME),
                "evaluation_manifest": str(stable_root / EVALUATION_MANIFEST_NAME),
            },
            token_metadata=bundle["token_metadata"],
        )
        return bundle
    except Exception as exc:
        if spec_path.is_file():
            try:
                _update_run_spec(spec_path, status="evaluation_failed", error=str(exc))
            except GateError:
                pass
        raise


def _validate_run_spec_artifacts(spec: Mapping[str, Any], root: Path, raw_rows: Sequence[Mapping[str, Any]]) -> None:
    raw_path = root / RAW_NAME
    if spec.get("raw_rows") != len(raw_rows) or spec.get("raw_sha256") != _sha256(raw_path):
        raise GateError("run_spec raw artifact identity mismatch")
    scored_rows = _read_jsonl(root / SCORED_NAME)
    for field, path in (
        ("scored_rows", root / SCORED_NAME),
        ("scored_sha256", root / SCORED_NAME),
        ("metrics_sha256", root / METRICS_NAME),
        ("evaluation_manifest_sha256", root / EVALUATION_MANIFEST_NAME),
    ):
        expected = len(scored_rows) if field == "scored_rows" else _sha256(path)
        if spec.get(field) != expected:
            raise GateError(f"run_spec {field} mismatch")


def _manifest_provenance(spec: Mapping[str, Any], gpus: str | None) -> dict[str, Any]:
    generation_code = spec.get("code")
    generation_environment = spec.get("environment")
    if not _code_identity_complete(generation_code) or not isinstance(generation_environment, Mapping):
        raise GateError("run_spec generation provenance missing")
    verification_code = _code_identity()
    if not _code_identity_complete(verification_code):
        raise GateError("verification code provenance is incomplete")
    return {
        "generation": {"code": dict(generation_code), "environment": dict(generation_environment)},
        "verification": {"code": verification_code, "environment": _environment(gpus)},
    }


def _verify_and_promote(exp: Experiment, *, gpus: str | None) -> Path:
    final = _exp_output_root(exp)
    building = _building_root(exp)
    if final.exists() or final.is_symlink():
        raise GateError(f"refusing to overwrite final output: {final}")
    _check_building_entries(building)
    spec = _load_json(building / RUN_SPEC_NAME)
    if spec is None or spec.get("status") != "evaluated":
        raise GateError("run_spec is not in evaluated state")
    _validate_run_spec(spec, exp, scope="formal")
    raw_rows = _validate_raw_rows(building / RAW_NAME, exp.expected_samples)
    _validate_run_spec_artifacts(spec, building, raw_rows)
    scored_rows = _read_jsonl(building / SCORED_NAME)
    _validate_scored_rows(raw_rows, scored_rows)
    bundle = _validate_evaluation_bundle(building, raw_rows, stable_root=final)
    recomputed = bundle["recomputed"]
    token_metadata = _validate_token_metadata(
        raw_rows, exp.expected_samples, require_full_length=True
    )
    if spec.get("token_metadata") != token_metadata:
        raise GateError("run_spec token full-length metadata mismatch")

    artifacts: dict[str, Any] = {}
    for name in STAGED_OUTPUTS:
        path = building / name
        artifacts[name] = {
            "path": str(final / name),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp.identity,
        "experiment_key": exp.key,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "base": spec["base"],
        "adapter": spec["adapter"],
        "collision_assets": spec["collision_assets"],
        "protocol": spec["protocol"],
        "tokenizer": spec["tokenizer"],
        "code": spec["code"],
        "environment": spec["environment"],
        "world_size": spec.get("world_size"),
        "gpus": spec.get("gpus"),
        "commands": spec["commands"],
        "evaluation": {
            "metrics": bundle["metrics"],
            "independently_recomputed": recomputed,
            "token_full_length": token_metadata,
            "evaluation_manifest_sha256": _sha256(building / EVALUATION_MANIFEST_NAME),
        },
        "artifacts": artifacts,
        "provenance": _manifest_provenance(spec, gpus),
    }
    _write_json_new(building / FINAL_MANIFEST_NAME, manifest)
    _check_building_entries(building, promotable=True)
    if final.exists() or final.is_symlink():
        raise GateError(f"refusing to overwrite final output: {final}")
    # Sibling-directory rename is atomic and no-replace, including a
    # concurrent creator that appears after the checks above.
    _rename_noreplace(building, final)
    return final


def _planned_commands(action: str, exp: Experiment, gpus: str | None) -> list[str]:
    if action == "smoke":
        root = _smoke_root(exp)
        commands = [
            _generation_command(exp, root / RAW_NAME, samples=SMOKE_SAMPLES, gpus=gpus),
            _evaluation_command(root / RAW_NAME, root / EVALUATION_SUBDIR, expected_samples=SMOKE_SAMPLES),
        ]
    elif action == "pilot":
        root = _pilot_root(exp)
        commands = [
            _generation_command(exp, root / RAW_NAME, samples=PILOT_SAMPLES, gpus=gpus),
            _evaluation_command(root / RAW_NAME, root / EVALUATION_SUBDIR, expected_samples=PILOT_SAMPLES),
        ]
    elif action == "generate":
        commands = [_generation_command(exp, _building_root(exp) / RAW_NAME, samples=exp.expected_samples, gpus=gpus)]
    elif action == "evaluate":
        commands = [_evaluation_command(_building_root(exp) / RAW_NAME, _building_root(exp) / EVALUATION_SUBDIR, expected_samples=exp.expected_samples)]
    else:
        commands = []
    return [shlex.join([str(item) for item in command]) for command in commands]


def _experiment_type(value: str) -> str:
    try:
        return _resolve_experiment(value).key
    except GateError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment",
        "--run",
        dest="experiment",
        required=True,
        type=_experiment_type,
        help="One of the four fixed Qwen3+BrickNet-PT identities.",
    )
    parser.add_argument(
        "--action",
        choices=("preflight", "smoke", "pilot", "generate", "evaluate", "verify"),
        required=True,
    )
    # Accept both the repository's recorded ``--gpus 0 1`` spelling and a
    # compact ``--gpu 0,1`` spelling, but normalize them to one immutable
    # comma-separated selector before any gate is evaluated.
    parser.add_argument("--gpus", nargs="+", dest="gpu_list", help="Physical CUDA IDs (space or comma separated).")
    parser.add_argument("--gpu", dest="gpu_single", help="One physical CUDA ID (alias for --gpus).")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the selected action; all actions default to dry-run.",
    )
    parser.add_argument(
        "--artifacts-approved",
        action="store_true",
        help="Required second gate for executable smoke/pilot/generate actions.",
    )
    args = parser.parse_args(argv)
    if args.gpu_list is not None and args.gpu_single is not None:
        parser.error("use only one of --gpus and --gpu")
    if args.gpu_list is not None:
        pieces: list[str] = []
        for item in args.gpu_list:
            pieces.extend(part for part in item.split(",") if part)
        args.gpus = ",".join(pieces)
    else:
        args.gpus = args.gpu_single
    # A few review helpers use the singular attribute; retaining it does not
    # add a second CLI mode or alter the fixed experiment identity.
    args.gpu = args.gpus
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exp = _resolve_experiment(args.experiment)
    report = preflight(args.action, exp, args.gpus)
    payload = dict(report)
    payload["execute"] = bool(args.execute)
    payload["artifacts_approved"] = bool(args.artifacts_approved)
    planning_blocker: str | None = None
    try:
        payload["planned_commands"] = _planned_commands(args.action, exp, args.gpus)
    except GateError as exc:
        planning_blocker = str(exc)
        payload["planned_commands"] = []
        payload["planning_blocker"] = planning_blocker
        payload["eligible"] = False
        payload["blockers"] = [
            *payload.get("blockers", []),
            f"PLANNING_COMMAND_INVALID:{planning_blocker}",
        ]
    else:
        payload["planning_blocker"] = None
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    # Dry-run is the default for every action.  It performs no subprocess and
    # leaves all existing files untouched.  evaluate/verify are therefore
    # inspectable without --execute, while the executable path below enforces
    # that flag.
    if not args.execute:
        return 0
    if args.action in {"smoke", "pilot", "generate"} and not args.artifacts_approved:
        print(
            "ERROR: --artifacts-approved is required for executable smoke/pilot/generate actions.",
            file=sys.stderr,
        )
        return 2
    if planning_blocker is not None:
        print(f"ERROR: planning blocked: {planning_blocker}", file=sys.stderr)
        return 2
    if not report["eligible"]:
        print("ERROR: preflight blocked execution: " + "; ".join(report["blockers"]), file=sys.stderr)
        return 2
    try:
        if args.action == "smoke":
            root = _smoke_root(exp)
            raw = _run_generation(exp, scope="smoke", gpus=args.gpus, samples=SMOKE_SAMPLES)
            _run_evaluation(exp, gpus=args.gpus, root=root, samples=SMOKE_SAMPLES)
            print(json.dumps({"status": "smoke_complete", "path": str(root), "raw": str(raw)}))
        elif args.action == "pilot":
            root = _pilot_root(exp)
            raw = _run_generation(exp, scope="pilot", gpus=args.gpus, samples=PILOT_SAMPLES)
            _run_evaluation(exp, gpus=args.gpus, root=root, samples=PILOT_SAMPLES)
            print(json.dumps({"status": "pilot_complete", "path": str(root), "raw": str(raw)}))
        elif args.action == "generate":
            raw = _run_generation(exp, scope="formal", gpus=args.gpus, samples=exp.expected_samples)
            print(json.dumps({"status": "generated", "path": str(raw)}))
        elif args.action == "evaluate":
            bundle = _run_evaluation(exp, gpus=args.gpus, root=_building_root(exp), samples=exp.expected_samples)
            print(json.dumps({"status": "evaluated", "samples": bundle["scored_rows"]}))
        elif args.action == "verify":
            destination = _verify_and_promote(exp, gpus=args.gpus)
            print(json.dumps({"status": "promoted", "output_root": str(destination)}))
    except (GateError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
