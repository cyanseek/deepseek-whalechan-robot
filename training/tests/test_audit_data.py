import json
import subprocess
import sys

import numpy as np
import pytest

from whalechan_training.audit_data import audit


def samples(directory, commands, dtype=np.float32, dimensions=(78, 81, 23)):
    actor_dim, critic_dim, action_dim = dimensions
    actor = np.zeros((1, len(commands), actor_dim), dtype=dtype)
    actor[..., -3:] = commands
    np.savez(
        directory / "on-policy-samples.npz",
        actor=actor,
        critic=np.zeros((1, len(commands), critic_dim), dtype=dtype),
        action_sampled=np.zeros((1, len(commands), action_dim), dtype=dtype),
        action_applied=np.zeros((1, len(commands), action_dim), dtype=dtype),
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


def contract(directory, actor_dim=124, critic_dim=150, action_dim=23, **overrides):
    value = {
        "actor_dim": actor_dim, "critic_dim": critic_dim,
        "action_joint_names": [f"joint_{i}" for i in range(action_dim)],
        "observation_order": ["base_ang_vel", "projected_gravity", "joint_pos",
                              "joint_vel", "actions", "motor_state", "command"],
    }
    value.update(overrides)
    (directory / "contract.json").write_text(json.dumps(value), encoding="utf8")


@pytest.mark.parametrize("dimensions", [(78, 81, 23), (124, 150, 23), (147, 150, 23)])
def test_recorded_contract_dimensions_and_command_metrics(tmp_path, dimensions):
    samples(tmp_path, [[0, 0, 0], [.05, 0, 0]], dimensions=dimensions)
    contract(tmp_path, *dimensions)
    report = audit(tmp_path)
    assert report["passed"], report["errors"]
    assert report["dimension_source"] == "contract.json"
    assert report["expected_dimensions"] == dict(zip(
        ("actor", "critic", "action"), dimensions
    ))
    assert report["sampled_transitions"] == 2
    assert report["sampled_zero_fraction"] == .5
    assert report["sampled_invalid_command_count"] == 0
    assert report["sampled_forward_range"] == pytest.approx([.05, .05])


def test_original_contract_without_dimension_fields_keeps_legacy_layout(tmp_path):
    samples(tmp_path, [[0, 0, 0]])
    contract(tmp_path)
    path = tmp_path / "contract.json"
    value = json.loads(path.read_text())
    del value["actor_dim"], value["critic_dim"]
    value["observation_order"].remove("motor_state")
    path.write_text(json.dumps(value))
    report = audit(tmp_path)
    assert report["passed"]
    assert report["dimension_source"] == "legacy 78/81/23"


@pytest.mark.parametrize("overrides", [
    {"actor_dim": None}, {"critic_dim": None}, {"actor_dim": True},
    {"actor_dim": 2}, {"critic_dim": 0}, {"actor_dim": "124"},
    {"action_joint_names": []}, {"action_joint_names": ["same", "same"]},
    {"observation_order": ["command", "motor_state"]},
])
def test_invalid_contract_is_not_silently_replaced_by_legacy_dimensions(tmp_path, overrides):
    samples(tmp_path, [[0, 0, 0]])
    contract(tmp_path, **overrides)
    with pytest.raises(ValueError, match="contract"):
        audit(tmp_path)


def test_partial_dimension_contract_is_rejected(tmp_path):
    samples(tmp_path, [[0, 0, 0]])
    contract(tmp_path)
    path = tmp_path / "contract.json"
    value = json.loads(path.read_text())
    del value["critic_dim"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contract"):
        audit(tmp_path)


@pytest.mark.parametrize("field", ["actor", "critic", "action_sampled", "action_applied"])
def test_actual_dimension_must_match_contract(tmp_path, field):
    samples(tmp_path, [[0, 0, 0]], dimensions=(124, 150, 23))
    contract(tmp_path)
    path = tmp_path / "on-policy-samples.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays[field] = arrays[field][..., :1]
    np.savez(path, **arrays)
    report = audit(tmp_path)
    assert not report["passed"]
    assert "dimension:" + field in report["errors"]


def test_action_clipping_does_not_broadcast_different_sample_shapes(tmp_path):
    samples(tmp_path, [[0, 0, 0], [.05, 0, 0]])
    path = tmp_path / "on-policy-samples.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["action_sampled"] = arrays["action_sampled"][:, :1]
    np.savez(path, **arrays)
    report = audit(tmp_path)
    assert not report["passed"]
    assert "sample shape:action_sampled" in report["errors"]


def test_cli_reads_contract_and_preserves_existing_output(tmp_path):
    samples(tmp_path, [[0, 0, 0], [.05, 0, 0]], dimensions=(124, 150, 23))
    contract(tmp_path)
    output = tmp_path / "audit.json"
    command = [sys.executable, "-m", "whalechan_training.audit_data", str(tmp_path), "--out", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    before = output.read_bytes()
    assert json.loads(before)["expected_dimensions"]["actor"] == 124
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
