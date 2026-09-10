"""Focused tests for BrickNet dataset-version selection."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import HfArgumentParser

from llamafactory.data.parser import get_dataset_list
from llamafactory.hparams.data_args import DataArguments


V2_ROOT = "/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM_v2/sharegpt"


def test_v1_is_a_noop() -> None:
    args = DataArguments(
        dataset="BrickNet-MM-SFT,identity",
        eval_dataset="BrickNet-MM-VAL",
        tokenized_path="cache/bricknet",
    )

    assert args.dataset == ["BrickNet-MM-SFT", "identity"]
    assert args.eval_dataset == ["BrickNet-MM-VAL"]
    assert args.tokenized_path == "cache/bricknet"


def test_v2_suffixes_train_and_eval_registry_names() -> None:
    args = DataArguments(
        dataset="BrickNet-MM-SFT,identity,BrickNet-Stage2-NonThinking-Control-10k",
        eval_dataset="BrickNet-MM-VAL,BrickNet-Stage8-R1-S-64",
        bricknet_dataset_version="v2",
        tokenized_path="cache/bricknet",
    )

    assert args.dataset == [
        "BrickNet-MM-SFT_v2",
        "identity",
        "BrickNet-Stage2-NonThinking-Control-10k_v2",
    ]
    assert args.eval_dataset == ["BrickNet-MM-VAL_v2", "BrickNet-Stage8-R1-S-64_v2"]
    assert args.tokenized_path == "cache/bricknet_v2"


def test_v2_does_not_duplicate_existing_suffix() -> None:
    args = DataArguments(
        dataset="BrickNet-MM-SFT_v2",
        eval_dataset="BrickNet-MM-VAL_v2",
        bricknet_dataset_version="v2",
        tokenized_path="cache/bricknet_v2",
    )

    assert args.dataset == ["BrickNet-MM-SFT_v2"]
    assert args.eval_dataset == ["BrickNet-MM-VAL_v2"]
    assert args.tokenized_path == "cache/bricknet_v2"


def test_v2_leaves_paths_and_non_bricknet_names_untouched() -> None:
    args = DataArguments(
        dataset="/tmp/BrickNet-MM-SFT.jsonl,org/remote-dataset,alpaca_en_demo",
        eval_dataset="remote_dataset",
        bricknet_dataset_version="v2",
    )

    assert args.dataset == ["/tmp/BrickNet-MM-SFT.jsonl", "org/remote-dataset", "alpaca_en_demo"]
    assert args.eval_dataset == ["remote_dataset"]

    online_args = DataArguments(
        dataset="BrickNet-MM-SFT,identity",
        bricknet_dataset_version="v2",
        dataset_dir="ONLINE",
    )
    assert online_args.dataset == ["BrickNet-MM-SFT", "identity"]


def test_v2_cache_isolated_for_nonempty_paths() -> None:
    assert DataArguments(bricknet_dataset_version="v2").tokenized_path is None
    assert DataArguments(bricknet_dataset_version="v2", tokenized_path="cache").tokenized_path == "cache_v2"
    assert DataArguments(bricknet_dataset_version="v2", tokenized_path="cache_v2").tokenized_path == "cache_v2"


def test_version_field_is_available_to_cli_and_config_parsers() -> None:
    parser = HfArgumentParser(DataArguments)

    from_config = parser.parse_dict({"bricknet_dataset_version": "v2"})[0]
    from_cli = parser.parse_args_into_dataclasses(["--bricknet_dataset_version", "v2"])[0]

    assert from_config.bricknet_dataset_version == "v2"
    assert from_cli.bricknet_dataset_version == "v2"


def test_invalid_config_version_fails_closed() -> None:
    parser = HfArgumentParser(DataArguments)

    try:
        parser.parse_dict({"bricknet_dataset_version": "v3"})
    except ValueError as error:
        assert "must be either 'v1' or 'v2'" in str(error)
    else:
        raise AssertionError("invalid BrickNet dataset version was accepted")


def test_pt_v2_is_absent_and_fails_closed() -> None:
    registry_path = Path(__file__).parents[2] / "data" / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    args = DataArguments(dataset="BrickNet-MM-PT", bricknet_dataset_version="v2")

    assert args.dataset == ["BrickNet-MM-PT_v2"]
    try:
        get_dataset_list(args.dataset, registry)
    except ValueError as error:
        assert str(error) == "Undefined dataset BrickNet-MM-PT_v2 in dataset_info.json."
    else:
        raise AssertionError("BrickNet PT v2 unexpectedly resolved through a v1 registry entry")


def test_registry_entries_and_legacy_bricknet_paths() -> None:
    registry_path = Path(__file__).parents[2] / "data" / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    expected = {
        "BrickNet-MM-SFT_v2": "BrickNet-MM_SFT_v2.jsonl",
        "BrickNet-MM-VAL_v2": "BrickNet-MM_VAL_v2.jsonl",
        "BrickNet-Stage2-NonThinking-Control-VAL511-Train_v2": "BrickNet-Stage2-NonThinking-Control-VAL511-Train_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-VAL511-Train_v2": "BrickNet-Stage2-Thinking-Hard-VAL511-Train_v2.jsonl",
        "BrickNet-Stage2-NonThinking-Control-VAL512-Eval_v2": "BrickNet-Stage2-NonThinking-Control-VAL512-Eval_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-VAL512-Eval_v2": "BrickNet-Stage2-Thinking-Hard-VAL512-Eval_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval_v2": "BrickNet-Stage2-ThinkingHard-V2-LeanState-VAL512-Eval_v2.jsonl",
        "BrickNet-Stage2-NonThinking-Control-10k_v2": "BrickNet-Stage2-NonThinking-Control_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-10k_v2": "BrickNet-Stage2-ThinkingHard_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-V2-LeanState-10k_v2": "BrickNet-Stage2-ThinkingHard-V2-LeanState_v2.jsonl",
        "BrickNet-Stage2-NonThinking-Control-All_v2": "BrickNet-MM-NonThinking-Control_v2.jsonl",
        "BrickNet-Stage8-R1-S-64_v2": "BrickNet-Stage8-R1-S_v2.jsonl",
    }
    optional_expected = {
        "BrickNet-MM-RL_v2": "BrickNet-MM-RL_v2.jsonl",
        "BrickNet-MM-RL-n2000-seed42_v2": "BrickNet-MM-RL_n2000_seed42_v2.jsonl",
        "BrickNet-MM-NonThinking-Control_v2": "BrickNet-MM-NonThinking-Control_v2.jsonl",
        "BrickNet-MM-Thinking-Hard_v2": "BrickNet-MM-Thinking-Hard_v2.jsonl",
        "BrickNet-Stage2-ThinkingHard-V2-LeanState-FullPool_v2": "BrickNet-Stage2-ThinkingHard-V2-LeanState-FullPool_v2.jsonl",
        **{
            f"BrickNet-MM-RL-MINING-shard-{index:03d}_v2": f"BrickNet-MM-RL-MINING-shard-{index:03d}_v2.jsonl"
            for index in range(34)
        },
    }

    for key, filename in expected.items():
        assert key in registry
        entry = registry[key]
        assert entry["file_name"] == f"{V2_ROOT}/{filename}"
        assert Path(entry["file_name"]).is_absolute()
        assert entry["formatting"] == "sharegpt"

        legacy_key = key.removesuffix("_v2")
        assert legacy_key in registry

    for key, filename in optional_expected.items():
        assert key in registry
        assert registry[key]["file_name"] == f"{V2_ROOT}/{filename}"
        assert Path(registry[key]["file_name"]).is_absolute()

    assert "BrickNet-MM-PT_v2" not in registry
    assert "BrickNet-MM-PT-VAL_v2" not in registry
    for key in (
        "BrickNet-Stage2-NonThinking-Control-50k_v2",
        "BrickNet-Stage8-R1-S-10k_v2",
        "BrickNet-Stage8-R1-C-64_v2",
        "BrickNet-Stage8-R1-B-64_v2",
    ):
        assert key not in registry

    assert registry["BrickNet-MM-PT"]["file_name"] == (
        "/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_PT.json"
    )
    assert registry["BrickNet-MM-SFT"]["file_name"] == (
        "/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_SFT.json"
    )
    assert registry["BrickNet-MM-VAL"]["file_name"] == (
        "/home/jiahao/task/BrickNet/outputs_preprocess/BrickNet-MM/sharegpt/BrickNet-MM_VAL.json"
    )
    assert registry["BrickNet-MM-PT-VAL"]["file_name"] == "BrickNet-MM_PT_VAL.json"
