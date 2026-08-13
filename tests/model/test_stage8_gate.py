from collections import Counter

import torch

from llamafactory.extras.stage8_gate import evaluate_initialization_gate, evaluate_supervised_token_mix


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
