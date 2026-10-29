"""Bounded provenance samples and command-conditioned online coverage."""

import json
from pathlib import Path
import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper


class CommandCoverage:
    """Attribute transitions to pre-step commands; censor ongoing segments."""

    columns = (
        "transitions",
        "episode_ends",
        "timeouts",
        "closed_segments",
        "closed_segment_ticks",
        "nonzero_reset_action_history",
        "reward_sum",
        "resamples_without_reset",
    )

    def __init__(self, num_envs, device, step_dt, termination_names):
        self.step_dt = step_dt
        self.termination_names = list(termination_names)
        self.ticks = torch.zeros(num_envs, device=device)
        self.counts = torch.zeros(
            (2, len(self.columns)), device=device, dtype=torch.float64
        )
        self.terminations = torch.zeros(
            (2, len(termination_names)), device=device, dtype=torch.int64
        )

    def observe(
        self, command, next_command, reward, done, timeout, reset_actions, causes
    ):
        idle = torch.linalg.vector_norm(command, dim=-1) < 0.005
        ended = done.bool()
        switched = (command != next_command).any(-1)
        closed = ended | switched
        self.ticks += 1
        nonzero_history = reset_actions.abs().amax(-1) != 0
        for group, mask in enumerate((idle, ~idle)):
            selected = mask & closed
            self.counts[group] += torch.stack(
                (
                    mask.sum(),
                    (mask & ended).sum(),
                    (mask & timeout).sum(),
                    selected.sum(),
                    self.ticks[selected].sum(),
                    (mask & ended & nonzero_history).sum(),
                    reward[mask].sum(),
                    (mask & switched & ~ended).sum(),
                )
            )
            for index, name in enumerate(self.termination_names):
                self.terminations[group, index] += (mask & causes[name]).sum()
        self.ticks[closed] = 0

    def report(self):
        groups = {}
        for i, name in enumerate(("standing", "walking")):
            row = dict(zip(self.columns, self.counts[i].cpu().tolist()))
            row["mean_closed_segment_s"] = (
                row["closed_segment_ticks"]
                * self.step_dt
                / max(1, row["closed_segments"])
            )
            row["termination_counts"] = dict(
                zip(self.termination_names, self.terminations[i].cpu().tolist())
            )
            groups[name] = row
        return {
            "groups": groups,
            "ongoing_segment_count": int((self.ticks > 0).sum()),
            "note": "Pre-step command groups; segments end at command changes or resets. "
            "Ongoing durations are censored. Termination causes may overlap.",
        }


class RecordedEnv(RslRlVecEnvWrapper):
    def __init__(self, env, out):
        super().__init__(env, clip_actions=3.0)
        self.out = Path(out)
        self.calls = 0
        self.samples = []
        # Float32 counters lose unit increments after 2**24 transitions.
        self.stats = torch.zeros(7, device=self.device, dtype=torch.int64)
        self.last = self.get_observations()
        self.coverage = CommandCoverage(
            self.num_envs,
            self.device,
            env.step_dt,
            env.termination_manager.active_terms,
        )

    def get_observations(self):
        # Keep exactly the noisy observation passed to PPO, including its first tick.
        obs = super().get_observations()
        self.last = obs
        return obs

    def step(self, actions):
        sample = self.calls % 200 == 0
        if sample:
            row = {
                "actor": self.last["actor"][:4].cpu().numpy().copy(),
                "critic": self.last["critic"][:4].cpu().numpy().copy(),
                "action_sampled": actions[:4].cpu().numpy().copy(),
                "action_applied": actions[:4].clamp(-3, 3).cpu().numpy().copy(),
                "tick": self.calls,
            }
        command = self.env.command_manager.get_command("twist").clone()
        idle = torch.linalg.vector_norm(command, dim=-1) < 0.005
        self.stats[0] += len(command)
        self.stats[1] += idle.sum()
        self.stats[2] += (~idle).sum()
        self.stats[3] += (self.last["actor"][:, -3:] != command).any(-1).sum()
        obs, reward, done, extras = super().step(actions)
        self.coverage.observe(
            command,
            self.env.command_manager.get_command("twist"),
            reward,
            done,
            extras["time_outs"],
            self.env.action_manager.action,
            {
                name: self.env.termination_manager.get_term(name)
                for name in self.coverage.termination_names
            },
        )
        self.stats[4] += done.sum()
        self.stats[5] += extras["time_outs"].sum()
        self.stats[6] += (~torch.isfinite(obs["actor"])).sum()
        if sample:
            row.update(
                next_actor=obs["actor"][:4].cpu().numpy().copy(),
                reward=reward[:4].cpu().numpy().copy(),
                done=done[:4].cpu().numpy().copy(),
                time_out=extras["time_outs"][:4].cpu().numpy().copy(),
            )
            self.samples.append(row)
        self.last = obs
        self.calls += 1
        return obs, reward, done, extras

    def flush(self):
        if self.samples:
            np.savez_compressed(
                self.out / "on-policy-samples.npz",
                **{
                    k: np.asarray([row[k] for row in self.samples])
                    for k in self.samples[0]
                },
            )
        keys = (
            "transitions",
            "zero_command",
            "walking_command",
            "command_observation_mismatch",
            "episode_ends",
            "timeouts",
            "nonfinite_next_actor",
        )
        (self.out / "data-summary.json").write_text(
            json.dumps(dict(zip(keys, self.stats.cpu().tolist())), indent=2)
        )
        (self.out / "command-coverage.json").write_text(
            json.dumps(self.coverage.report(), indent=2)
        )
