import json
import mujoco
import numpy as np
import pytest
import torch
from whalechan_training.data import CommandCoverage
from whalechan_training.task import C, NAMES, get_spec, make_cfg
from whalechan_training.audit_data import audit


def test_body_and_control_contract():
    model = get_spec().compile()
    assert model.nq == 30 and model.nv == 29
    assert sum(model.body_mass) == pytest.approx(1.267)
    assert model.ngeom == 30
    for joint in NAMES:
        assert model.joint(joint).id >= 0
    assert all(
        model.geom(name).type[0] == mujoco.mjtGeom.mjGEOM_BOX
        for name in ("left_sole", "right_sole")
    )
    cfg = make_cfg(4)
    assert cfg.decimation * cfg.sim.mujoco.timestep == 0.02
    assert len(cfg.actions["joint_pos"].actuator_names) == 23
    assert cfg.commands["twist"].init_velocity_prob == 0
    assert cfg.commands["twist"].rel_forward_envs == 0
    assert cfg.commands["twist"].ranges.lin_vel_x == (0.025, 0.08)
    assert "base_lin_vel" not in cfg.observations["actor"].terms
    assert "base_lin_vel" in cfg.observations["critic"].terms
    assert cfg.observations["actor"].enable_corruption
    assert not make_cfg(1, play=True).observations["actor"].enable_corruption
    motors = cfg.scene.entities["robot"].articulation.actuators
    assert all(m.delay_min_lag == m.delay_max_lag == 2 for m in motors)
    assert len(C["kp"]) == len(C["caps"]) == 23


def test_coverage_attributes_reset_to_previous_command():
    stats = CommandCoverage(2, "cpu", 0.02, ["fall", "timeout"])
    command = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    zero = torch.zeros(2, dtype=torch.bool)
    stats.observe(
        command,
        command,
        torch.ones(2),
        zero,
        zero,
        torch.ones(2, 23),
        {"fall": zero, "timeout": zero},
    )
    # World 0 falls while standing, then resets into a walk command.
    # World 1 changes command without ending its episode.
    stats.observe(
        command,
        command.flip(0),
        torch.ones(2),
        torch.tensor([1, 0]),
        zero,
        torch.zeros(2, 23),
        {"fall": torch.tensor([True, False]), "timeout": zero},
    )
    result = stats.report()
    stand, walk = (result["groups"][x] for x in ("standing", "walking"))
    assert stand["episode_ends"] == 1 and walk["episode_ends"] == 0
    assert stand["termination_counts"]["fall"] == 1
    assert stand["nonzero_reset_action_history"] == 0
    assert walk["resamples_without_reset"] == 1
    assert stand["mean_closed_segment_s"] == pytest.approx(0.04)
    assert result["ongoing_segment_count"] == 0


def test_timeout_and_overlapping_causes_are_not_double_counted():
    stats = CommandCoverage(1, "cpu", 0.02, ["fall", "timeout"])
    cmd = torch.zeros(1, 3)
    true = torch.ones(1, dtype=torch.bool)
    stats.observe(
        cmd,
        cmd,
        torch.ones(1),
        true,
        true,
        torch.ones(1, 23),
        {"fall": true, "timeout": true},
    )
    row = stats.report()["groups"]["standing"]
    assert row["episode_ends"] == row["timeouts"] == 1
    assert row["nonzero_reset_action_history"] == 1
    assert row["closed_segment_ticks"] == 1


def test_all_standing_samples_are_auditable(tmp_path):
    np.savez(
        tmp_path / "on-policy-samples.npz",
        actor=np.zeros((1, 2, 78)),
        critic=np.zeros((1, 2, 81)),
        action_applied=np.zeros((1, 2, 23)),
        action_sampled=np.zeros((1, 2, 23)),
    )
    (tmp_path / "data-summary.json").write_text(
        json.dumps(
            {
                "transitions": 2,
                "zero_command": 2,
                "command_observation_mismatch": 0,
                "nonfinite_next_actor": 0,
            }
        )
    )
    result = audit(tmp_path)
    assert result["passed"]
    assert result["sampled_forward_range"] is None
    assert result["whole_run_zero_fraction"] == 1
