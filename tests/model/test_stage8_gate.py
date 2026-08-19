from collections import Counter

import torch

from llamafactory.extras.stage8_gate import (
    evaluate_initialization_gate,
    evaluate_stage8_build_report,
    evaluate_supervised_token_mix,
    file_sha256,
    resolve_stage5_report_binding,
    resolve_stage8_comparator,
    resolve_stage8_eval_mode,
)


class MockParameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


def test_stage8_initialization_gate_accepts_zero_impact_new_default_lora():
    reference = torch.tensor([[[1.0, 2.0]]])
    candidate = reference.clone()
    parameters = [
        ("base_model.model.layer.weight", MockParameter(False)),
        ("base_model.model.layer.lora_A.default.weight", MockParameter(True)),
        ("base_model.model.layer.lora_B.default.weight", MockParameter(True)),
    ]
    result = evaluate_initialization_gate(
        reference,
        candidate,
        parameters,
        {"old": "same"},
        {"old": "same"},
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert result["initialization_eligible"] is True
    assert result["base_and_merged_adapter_chain_frozen"] is True
    assert result["only_new_default_lora_trainable"] is True


def test_stage8_initialization_gate_rejects_base_training_or_changed_logits():
    parameters = [
        ("base_model.model.layer.weight", MockParameter(True)),
        ("base_model.model.layer.lora_A.default.weight", MockParameter(True)),
    ]
    result = evaluate_initialization_gate(
        torch.tensor([0.0]),
        torch.tensor([0.1]),
        parameters,
        {"old": "before"},
        {"old": "after"},
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert result["initialization_eligible"] is False
    assert result["logits_match"] is False
    assert result["only_new_default_lora_trainable"] is False
    assert result["adapter_artifacts_unchanged"] is False


def test_stage8_token_mix_uses_exact_half_percent_tolerance():
    assert evaluate_supervised_token_mix(Counter({"R1-S": 800, "R1-C": 200}), "R1-C")[
        "token_mix_eligible"
    ]
    assert evaluate_supervised_token_mix(Counter({"R1-S": 805, "R1-C": 195}), "R1-C")[
        "token_mix_eligible"
    ]
    assert not evaluate_supervised_token_mix(Counter({"R1-S": 806, "R1-C": 194}), "R1-C")[
        "token_mix_eligible"
    ]


def build_report(dataset_path, *, stage5_passed=True, dataset_sha256="dataset-sha", window_materialized=True):
    return {
        "schema_version": "bricknet-stage8-act-sft-build-report-v1",
        "size": 64,
        "stage5_replay_gate": {"stage5_replay_gate_passed": stage5_passed},
        "variants": {
            "R1-S": {
                "count": 2,
                "dataset": str(dataset_path),
                "dataset_sha256": dataset_sha256,
                "trajectory_type_counts": {"R1-S": 2},
                "window_materialized": window_materialized,
            }
        },
    }


def test_stage8_build_report_binds_variant_to_current_dataset(tmp_path):
    dataset = tmp_path / "BrickNet-Stage8-R1-S.jsonl"
    dataset.write_text("{}\n{}\n")
    result = evaluate_stage8_build_report(
        build_report(dataset),
        run="R1-S",
        expected_size=64,
        dataset_path=dataset,
        dataset_sha256="dataset-sha",
        dataset_count=2,
        dataset_window_materialized=True,
    )
    assert result == {"build_report_gate_passed": True, "build_report_blockers": []}


def test_stage8_build_report_blocks_smoke_override_and_dataset_drift(tmp_path):
    dataset = tmp_path / "BrickNet-Stage8-R1-S.jsonl"
    dataset.write_text("{}\n{}\n")
    report = build_report(dataset, stage5_passed=False, dataset_sha256="old-sha", window_materialized=False)
    result = evaluate_stage8_build_report(
        report,
        run="R1-S",
        expected_size=64,
        dataset_path=dataset,
        dataset_sha256="new-sha",
        dataset_count=2,
        dataset_window_materialized=True,
    )
    assert result["build_report_gate_passed"] is False
    assert "STAGE5_REPLAY_GATE_NOT_PASSED" in result["build_report_blockers"]
    assert "BUILD_REPORT_VARIANT_SHA256_MISMATCH" in result["build_report_blockers"]
    assert "BUILD_REPORT_WINDOW_MATERIALIZATION_MISMATCH" in result["build_report_blockers"]


def test_stage8_main_eval_modes_and_explicit_ablation_gate():
    assert resolve_stage8_eval_mode("S8-ZS-Greedy", None, ablation=False) == ("a0-act-feedback", False)
    assert resolve_stage8_eval_mode("S8-ZS-DFS", None, ablation=False) == ("a1-feedback-search", False)
    assert resolve_stage8_eval_mode("R1-S", None, ablation=False) == ("a0-act-feedback", False)
    assert resolve_stage8_eval_mode("R1-C", None, ablation=False) == ("a0-act-feedback", False)
    assert resolve_stage8_eval_mode("R1-B", None, ablation=False) == ("a1-feedback-search", False)
    try:
        resolve_stage8_eval_mode("R1-B", "a0-act-feedback", ablation=False)
    except ValueError as exc:
        assert "pass --ablation" in str(exc)
    else:
        raise AssertionError("mode mismatch was not blocked")
    assert resolve_stage8_eval_mode("R1-B", "a0-act-feedback", ablation=True) == ("a0-act-feedback", True)


def test_stage8_trained_policies_resolve_separately_named_comparators():
    assert resolve_stage8_comparator("R1-S") == "S8-ZS-Greedy"
    assert resolve_stage8_comparator("R1-C") == "S8-ZS-Greedy"
    assert resolve_stage8_comparator("R1-B") == "S8-ZS-DFS"
    try:
        resolve_stage8_comparator("S8-ZS-Greedy")
    except ValueError as exc:
        assert "not a Stage-8 trained policy" in str(exc)
    else:
        raise AssertionError("a comparator cannot resolve itself as a trained policy")


def test_stage8_eval_resolves_fixed_stage5_report_and_rejects_hash_drift(tmp_path):
    stage5_report = tmp_path / "stage5.json"
    stage5_report.write_text('{"stage5_replay_gate_passed": true}\n')
    build = {
        "schema_version": "bricknet-stage8-act-sft-build-report-v1",
        "stage5_replay_gate": {
            "path": str(stage5_report),
            "sha256": file_sha256(stage5_report),
            "stage5_replay_gate_passed": True,
        },
    }
    assert resolve_stage5_report_binding(build) == {
        "path": str(stage5_report.resolve()),
        "sha256": file_sha256(stage5_report),
    }
    stage5_report.write_text('{"tampered": true}\n')
    try:
        resolve_stage5_report_binding(build)
    except ValueError as exc:
        assert "hash differs" in str(exc)
    else:
        raise AssertionError("Stage-5 report hash drift was not blocked")
