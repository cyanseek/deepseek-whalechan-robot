import json
import subprocess
import sys

import numpy as np
import pytest

from whalechan_training.audit_data import audit


def samples(directory, commands, dtype=np.float32):
    actor = np.zeros((1, len(commands), 78), dtype=dtype)
    actor[..., -3:] = commands
    np.savez(
        directory / "on-policy-samples.npz",
        actor=actor,
        critic=np.zeros((1, len(commands), 81), dtype=dtype),
        action_sampled=np.zeros((1, len(commands), 23), dtype=dtype),
        action_applied=np.zeros((1, len(commands), 23), dtype=dtype),
    )
    # Whole-run counters are independent of this retained sample.
    (directory / "data-summary.json").write_text(json.dumps({
        "transitions": 100, "zero_command": 20,
        "command_observation_mismatch": 0, "nonfinite_next_actor": 0,
    }))


@pytest.mark.parametrize("command", [
    [-.05, 0, 0], [.001, 0, 0], [.005, 0, 0], [.024, 0, 0], [.081, 0, 0],
    [0, .01, 0], [0, 0, .1], [float("nan"), 0, 0],
])
def test_invalid_commands_are_not_counted_as_standing(tmp_path, command):
    samples(tmp_path, [[0, 0, 0], [.05, 0, 0], command])
    report = audit(tmp_path)
    assert not report["passed"]
    assert report["sampled_zero_fraction"] == pytest.approx(1 / 3)
    assert report["sampled_invalid_command_count"] == 1
    assert report["sampled_forward_range"] == pytest.approx([.05, .05])
    assert report["whole_run_zero_fraction"] == .2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_and_forward_range_endpoints_are_valid(tmp_path, dtype):
    samples(tmp_path, [[0, 0, 0], [.025, 0, 0], [.08, 0, 0]], dtype)
    report = audit(tmp_path)
    assert report["passed"] and not report["errors"]
    assert report["sampled_invalid_command_count"] == 0
    assert report["sampled_zero_fraction"] == pytest.approx(1 / 3)
    assert report["sampled_forward_range"] == pytest.approx([.025, .08])


def test_cli_saves_rejected_command_report(tmp_path):
    samples(tmp_path, [[-.05, 0, 0]])
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, "-m", "whalechan_training.audit_data", str(tmp_path), "--out", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert not report["passed"] and report["sampled_invalid_command_count"] == 1
    assert "command range" in report["errors"]
