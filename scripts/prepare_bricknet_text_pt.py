#!/usr/bin/env python3
"""Build a deterministic text-only BrickNet PT subset for mixed SFT-stage training."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRICKNET_DATA = ROOT.parent / "BrickNet" / "data" / "bricknet_datasets"
DEFAULT_OUTPUT = ROOT / "data" / "BrickNet-PT_text_270102_seed42.jsonl"
DEFAULT_SYSTEM = (
    "You generate BrickNet path text build sequences. Output only valid BrickNet path text without explanations, "
    "markdown, or extra text."
)
DEFAULT_PROMPT = "Generate an unconditional connected BrickNet path text build sequence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="+",
        default=[
            DEFAULT_BRICKNET_DATA / "paths_pt.jsonl",
            DEFAULT_BRICKNET_DATA / "paths_sft.jsonl",
        ],
        help="BrickNet path JSONL pools. Defaults to the PT and SFT pools used by the paper's PT stage.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=270_102)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-input-rows",
        type=int,
        default=None,
        help="Stop after this many total source rows. Intended only for smoke tests.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def path_digest(path_text: str, seed: int) -> bytes:
    seed_key = str(seed).encode("ascii")
    return hashlib.blake2b(
        path_text.encode("utf-8"),
        digest_size=16,
        key=seed_key,
        person=b"BrickNetMixPT",
    ).digest()


def normalize_path(raw_path: Any) -> str | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return raw_path if raw_path.endswith("\n") else raw_path + "\n"


def count_path_parts(path_text: str) -> int:
    nonempty_lines = sum(1 for line in path_text.splitlines() if line.strip())
    return (nonempty_lines + 1) // 2


def sample_paths(args: argparse.Namespace) -> tuple[list[tuple[int, bytes, dict[str, Any]]], dict[str, Any]]:
    # A bottom-k hash sample is deterministic, order-independent, and deduplicates exact path text.
    heap: list[tuple[int, bytes, dict[str, Any]]] = []
    selected: dict[bytes, str] = {}
    stats: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    started = time.monotonic()

    stop = False
    for source_path in args.paths:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        source_name = source_path.stem
        with source_path.open("rb") as handle:
            for source_line, raw_line in enumerate(handle, 1):
                if args.max_input_rows is not None and stats["rows_seen"] >= args.max_input_rows:
                    stop = True
                    break

                stats["rows_seen"] += 1
                source_rows[source_name] += 1
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stats["malformed_rows"] += 1
                    continue

                path_text = normalize_path(row.get("path"))
                if path_text is None:
                    stats["empty_paths"] += 1
                    continue

                digest = path_digest(path_text, args.seed)
                score = int.from_bytes(digest, "big")
                if digest in selected:
                    if selected[digest] != path_text:
                        raise RuntimeError(f"BLAKE2 digest collision at {source_path}:{source_line}")
                    stats["selected_duplicates"] += 1
                    continue

                metadata = {
                    "source_pool": source_name,
                    "source_line": source_line,
                    "source": row.get("source"),
                    "round": row.get("round"),
                    "sample_index": row.get("sample_index"),
                    "component_index": row.get("component_index"),
                    "n_parts": (
                        len(row["nodes"]) if isinstance(row.get("nodes"), list) else count_path_parts(path_text)
                    ),
                    "complete_component": row.get("complete_component"),
                    "complete_npz": row.get("complete_npz"),
                    "path": path_text,
                }
                candidate = (-score, digest, metadata)

                if len(heap) < args.num_samples:
                    heapq.heappush(heap, candidate)
                    selected[digest] = path_text
                elif score < -heap[0][0]:
                    _, removed_digest, _ = heapq.heapreplace(heap, candidate)
                    del selected[removed_digest]
                    selected[digest] = path_text

                if args.progress_every > 0 and stats["rows_seen"] % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    rate = stats["rows_seen"] / elapsed if elapsed else 0.0
                    print(
                        f"Scanned {stats['rows_seen']:,} rows; retained {len(heap):,}; {rate:,.0f} rows/s",
                        file=sys.stderr,
                        flush=True,
                    )

        if stop:
            break

    if len(heap) < args.num_samples:
        raise RuntimeError(
            f"Requested {args.num_samples:,} samples, but only retained {len(heap):,} non-empty unique paths."
        )

    selected_rows = sorted(
        [(-negative_score, digest, metadata) for negative_score, digest, metadata in heap],
        key=lambda item: (item[0], item[1]),
    )
    report = {
        "schema_version": 1,
        "sampling": "seeded BLAKE2b bottom-k over normalized path text",
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "retained_samples": len(selected_rows),
        "source_paths": [str(path.resolve()) for path in args.paths],
        "source_rows_seen": dict(source_rows),
        "max_input_rows": args.max_input_rows,
        "stats": dict(stats),
        "system_message": args.system_message,
        "prompt": args.prompt,
    }
    return selected_rows, report


def write_dataset(
    rows: list[tuple[int, bytes, dict[str, Any]]],
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    tmp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    output_sources: Counter[str] = Counter()
    output_stats: Counter[str] = Counter()
    min_parts: int | None = None
    max_parts = 0

    try:
        with tmp_output.open("w", encoding="utf-8") as handle:
            for _, digest, metadata in rows:
                path_text = metadata.pop("path")
                output_sources[metadata["source_pool"]] += 1
                n_parts = int(metadata["n_parts"])
                output_stats["parts_total"] += n_parts
                output_stats["complete_npz"] += metadata["complete_npz"] is True
                min_parts = n_parts if min_parts is None else min(min_parts, n_parts)
                max_parts = max(max_parts, n_parts)
                record = {
                    "id": f"bricknet_pt_text_{digest.hex()}",
                    "messages": [
                        {"role": "system", "content": args.system_message},
                        {"role": "user", "content": args.prompt},
                        {"role": "assistant", "content": path_text},
                    ],
                    "meta": {
                        "dataset": "BrickNet-PT-Text",
                        "split": "PT",
                        "sampling_digest": digest.hex(),
                        **metadata,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        report["output"] = str(args.output.resolve())
        report["output_source_counts"] = dict(output_sources)
        report["output_stats"] = {
            "parts_min": min_parts,
            "parts_mean": output_stats["parts_total"] / len(rows),
            "parts_max": max_parts,
            "complete_npz": output_stats["complete_npz"],
            "complete_npz_rate": output_stats["complete_npz"] / len(rows),
        }
        with tmp_report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        os.replace(tmp_output, args.output)
        os.replace(tmp_report, report_path)
    finally:
        tmp_output.unlink(missing_ok=True)
        tmp_report.unlink(missing_ok=True)

    print(f"Wrote {len(rows):,} samples to {args.output}")
    print(f"Wrote sampling report to {report_path}")


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_input_rows is not None and args.max_input_rows <= 0:
        raise ValueError("--max-input-rows must be positive")
    if not args.system_message.strip() or not args.prompt.strip():
        raise ValueError("--system-message and --prompt must be non-empty")

    rows, report = sample_paths(args)
    write_dataset(rows, report, args)


if __name__ == "__main__":
    main()
