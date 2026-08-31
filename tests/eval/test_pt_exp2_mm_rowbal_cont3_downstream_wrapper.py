"""Static contract tests for the PT-exp2-mm-rowbal-cont3 CUDA0 wrapper.

These tests inspect the shell source and run ``bash -n`` only.  They never
source or execute the wrapper, and therefore cannot start training, prediction,
or evaluation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "tmp_bash/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh"
LAUNCHER = ROOT / "scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py"


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_has_valid_shell_syntax_without_running_it() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_wrapper_has_fixed_cuda0_fail_stop_stage_order() -> None:
    source = _source()
    expected_stages = (
        "run_stage exp4_4_3 train",
        "run_stage exp4_4_3 predict",
        "run_stage exp4_4_3 evaluate",
        "run_stage exp4_7_3 train",
        "run_stage exp4_7_3 predict",
        "run_stage exp4_7_3 evaluate",
    )
    positions = [source.index(stage) for stage in expected_stages]
    assert positions == sorted(positions)
    assert source.count("--gpus 0") == 1
    assert "--gpus\n        0" not in source
    assert "--rowbal-ep3-approved" in source
    assert "--execute" in source
    assert "set -Eeuo pipefail" in source


def test_preflight_only_is_explicit_and_execute_is_conditional() -> None:
    source = _source()
    assert '[[ "${1:-}" == "--preflight-only" ]]' in source
    assert "if ((PREFLIGHT_ONLY == 0)); then" in source
    assert "command+=(--execute)" in source
    assert source.index("command+=(--execute)") > source.index("if ((PREFLIGHT_ONLY == 0)); then")
    assert 'payload.get("mode") != "dry-run"' in source
    assert 'payload.get("executed") is not False' in source


def test_wrapper_has_lock_pid_log_disk_and_gpu_gates_without_kill() -> None:
    source = _source()
    for token in (
        "LOCK_FILE=",
        "LOG_FILE=",
        "PID_FILE=",
        "MIN_FREE_KIB=$((40 * 1024 * 1024))",
        "flock -n 9",
        "check_disk",
        "check_gpu0",
        'printf \'%s\\n\' "$$" >"${PID_FILE}"',
        'rm -f -- "${PID_FILE}"',
        "nvidia-smi --id=0",
    ):
        assert token in source

    # Existing processes are a hard blocker; this wrapper must not kill or
    # reset anything on the shared GPU.
    assert not re.search(r"\b(?:kill|pkill|killall)\b", source)
    assert "--gpu-reset" not in source


def test_preflight_gate_accepts_only_dependency_wait_blockers() -> None:
    source = _source()
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    assert '"train": set()' in source
    assert '"predict": {"WAIT_FINAL_SFT_TRAIN_ADAPTER"}' in source
    assert '"WAIT_FINAL_SFT_TRAIN_ADAPTER",' in source and '"WAIT_FINAL_SFT_PREDICTION",' in source
    assert 'blockers.append("WAIT_FINAL_SFT_TRAIN_ADAPTER")' in launcher_source
    assert 'blockers.append("WAIT_FINAL_SFT_PREDICTION")' in launcher_source
    assert "not actual.issubset(allowed)" in source
