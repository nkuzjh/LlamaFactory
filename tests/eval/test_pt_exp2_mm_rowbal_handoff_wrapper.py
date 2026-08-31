"""Static regression for the Text250k -> rowbal handoff JSON gate.

The test extracts and runs only the embedded JSON parser from the handoff
wrapper.  It deliberately never invokes either launcher, CUDA, or an
experiment process.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "tmp_bash/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.sh"
EXPECTED_RUN = "exp4_4_2"


def _embedded_gate_parser() -> str:
    source = WRAPPER.read_text(encoding="utf-8")
    match = re.search(
        r'''reason="\$\("\$\{PYTHON\}" - "\$\{result_file\}" "\$\{experiment\}" <<'PY'\n(.*?)\nPY\n\)"; then''',
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "handoff wrapper JSON gate heredoc disappeared"
    return match.group(1)


def _payload(*, experiment: str = EXPECTED_RUN, checks_run: str = EXPECTED_RUN) -> dict:
    return {
        "experiment": experiment,
        "action": "evaluate",
        "mode": "dry-run",
        "gpus": "1",
        "ready": True,
        "already_complete": True,
        "checks": {"run": checks_run, "output_complete": True},
        "blockers": [],
        "executed": False,
    }


def _run_parser(tmp_path: Path, payload: dict) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "dry_run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", _embedded_gate_parser(), str(path), EXPECTED_RUN],
        text=True,
        capture_output=True,
        check=False,
    )


def test_correct_payload_is_accepted_without_real_launcher(tmp_path: Path) -> None:
    result = _run_parser(tmp_path, _payload())
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize(
    "payload",
    [
        # The old/incorrect shape had no top-level ``experiment`` field.
        {key: value for key, value in _payload().items() if key != "experiment"},
        # A top-level experiment mismatch must not be rescued by checks.run.
        _payload(experiment="exp4_7_2"),
        # Both locations are bound to the requested run.
        _payload(checks_run="exp4_7_2"),
    ],
    ids=("missing_top_level_experiment", "wrong_top_level_experiment", "wrong_checks_run"),
)
def test_malformed_or_mismatched_payload_is_rejected_without_real_launcher(
    tmp_path: Path, payload: dict
) -> None:
    result = _run_parser(tmp_path, payload)
    assert result.returncode != 0
    assert result.stdout.strip() == "EVALUATE_GATE_NOT_STRICTLY_COMPLETE"
