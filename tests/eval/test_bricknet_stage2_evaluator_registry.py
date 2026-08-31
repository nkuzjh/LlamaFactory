import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage2 = _load_script(
    "test_launch_bricknet_stage2_sft_registry",
    "launch_bricknet_stage2_sft.py",
)
evaluator = _load_script(
    "test_evaluate_bricknet_stage2_registry",
    "evaluate_bricknet_stage2.py",
)


@pytest.mark.parametrize(
    ("experiment_id", "variant", "train_config", "predict_config", "eval_output"),
    (
        (
            "exp4_4_3",
            "nonthinking-control",
            "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml",
            "qwen35_08b_bricknet_stage2_exp4_4_3_nonthinking_control_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml",
            "eval_exp4_4_3_PT_exp2_mm_rowbal_cont3_ep3_nonthinking_control_10k_val512_in16384_out16384_p95_t1_k20",
        ),
        (
            "exp4_7_3",
            "thinking-hard-v2-lean-state",
            "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_10k_pt_exp2_mm_rowbal_cont3_ep3.yaml",
            "qwen35_08b_bricknet_stage2_exp4_7_3_thinking_hard_v2_lean_state_predict_pt_exp2_mm_rowbal_cont3_ep3.yaml",
            "eval_exp4_7_3_PT_exp2_mm_rowbal_cont3_ep3_thinking_hard_v2_lean_state_10k_val512_in16384_out16384_p95_t1_k20",
        ),
    ),
)
def test_rowbal_ep3_experiments_are_registered_for_evaluator_only(
    experiment_id: str,
    variant: str,
    train_config: str,
    predict_config: str,
    eval_output: str,
) -> None:
    experiment = evaluator.EXPERIMENTS_BY_ID[experiment_id]

    assert experiment.experiment_id == experiment_id
    assert experiment.variant == variant
    assert experiment.scale == "10k"
    assert experiment.dataset in {
        "BrickNet-Stage2-NonThinking-Control-10k",
        "BrickNet-Stage2-ThinkingHard-V2-LeanState-10k",
    }
    assert experiment.train_config.name == train_config
    assert experiment.predict_config.name == predict_config
    assert experiment.predict_output.name == eval_output
    assert experiment.predict_output.name.startswith(f"eval_{experiment_id}_")
    assert experiment.train_output.name.startswith(f"train_{experiment_id}_")
    assert "PT_exp2_mm_rowbal_cont3_ep3" in experiment.train_output.name
    assert "PT_exp2_mm_rowbal_cont3_ep3" in experiment.predict_output.name
    assert stage2._trace_variant(experiment) == variant


def test_rowbal_ep3_ids_are_accepted_by_evaluator_without_public_train_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_variants = {
        "nonthinking-control",
        "thinking-hard",
        "thinking-hard-v2-lean-state",
    }
    private_keys = {
        key for key, value in stage2.EXPERIMENTS.items() if value.experiment_id in {"exp4_4_3", "exp4_7_3"}
    }
    assert {value.experiment_id for value in evaluator.EXPERIMENTS_BY_ID.values()} >= {
        "exp4_4_3",
        "exp4_7_3",
    }
    assert private_keys == {("exp4_4_3", "10k"), ("exp4_7_3", "10k")}
    assert all(key[0] not in public_variants for key in private_keys)

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_bricknet_stage2.py", "--experiment", "exp4_4_3"],
    )
    assert evaluator.parse_args().experiment == "exp4_4_3"


def test_rowbal_ep3_registration_keeps_frozen_evaluation_contract() -> None:
    assert evaluator.EXPECTED_SAMPLES == 512
    assert evaluator.REWARD_WEIGHTS == {
        "parse_prefix": 0.20,
        "inventory_f1": 0.20,
        "length_score": 0.10,
        "collision_prefix": 0.20,
        "pose_match": 0.30,
    }
    assert evaluator.POSE_TOLERANCES == {
        "translation": 0.5,
        "rotation_degrees": 5.0,
        "success_threshold": 1.0,
    }
