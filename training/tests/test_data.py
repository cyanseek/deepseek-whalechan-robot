import json
from types import SimpleNamespace

import pytest
import torch

from whalechan_training.data import CommandCoverage, RecordedEnv


class CounterEnv:
    """Two CPU worlds for testing the recorder without GPU physics."""

    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.max_episode_length = 100
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(is_finite_horizon=False)
        self.command = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
        self.obs = {"actor": torch.zeros(2, 78), "critic": torch.zeros(2, 81)}
        self.obs["actor"][:, -3:] = self.command
        self.obs["actor"][0, -1] = 0.1  # One command mismatch to count.
        self.command_manager = SimpleNamespace(get_command=lambda _: self.command)
        self.observation_manager = SimpleNamespace(compute=lambda: self.obs)
        self.action_manager = SimpleNamespace(
            total_action_dim=23, action=torch.zeros(2, 23)
        )
        self.timeout = torch.tensor([True, False])
        self.termination_manager = SimpleNamespace(
            active_terms=["timeout"], get_term=lambda _: self.timeout
        )

    def reset(self):
        return self.obs, {}

    def step(self, actions):
        self.action_manager.action = actions
        next_obs = {key: value.clone() for key, value in self.obs.items()}
        next_obs["actor"][0, 0] = float("nan")  # One invalid value to count.
        return (
            next_obs,
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            self.timeout,
            {},
        )


@pytest.mark.parametrize("previous_count", [2**24, 2**32])
def test_recorded_counts_keep_unit_increments_in_long_runs(tmp_path, previous_count):
    recorder = RecordedEnv(CounterEnv(), tmp_path)
    recorder.stats[:] = previous_count
    recorder.stats[0] = 2 * previous_count
    recorder.step(torch.zeros(2, 23))
    recorder.flush()

    report = json.loads((tmp_path / "data-summary.json").read_text())
    assert report == {
        "transitions": 2 * previous_count + 2,
        "zero_command": previous_count + 1,
        "walking_command": previous_count + 1,
        "command_observation_mismatch": previous_count + 1,
        "episode_ends": previous_count + 1,
        "timeouts": previous_count + 1,
        "nonfinite_next_actor": previous_count + 1,
    }
    assert report["transitions"] == report["zero_command"] + report["walking_command"]


def test_termination_counts_keep_unit_increments_in_long_runs():
    coverage = CommandCoverage(2, "cpu", 0.02, ["fall"])
    previous_count = 2**24
    coverage.terminations[:] = previous_count
    command = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    fallen = torch.tensor([True, False])
    coverage.observe(
        command,
        command,
        torch.ones(2),
        fallen,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, 23),
        {"fall": fallen},
    )
    report = coverage.report()["groups"]
    assert report["standing"]["termination_counts"]["fall"] == previous_count + 1
    assert report["walking"]["termination_counts"]["fall"] == previous_count
