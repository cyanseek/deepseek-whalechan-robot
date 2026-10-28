"""Flat-ground whole-sole training targets, matching native acceptance geometry.

Uses mjlab's clearance/landing-height shaping form, but evaluates the minimum
oriented-box sole height instead of the bottom-center site. No actor input or
controller is added. This adapter is intentionally limited to the plane task.
"""

import torch


def heights(env):
    if not hasattr(env, "_whale_sole_geometry"):
        m = env.sim.mj_model
        ids = [m.geom("robot/" + s + "_sole").id for s in ("left", "right")]
        sizes = torch.as_tensor(
            m.geom_size[ids], device=env.device, dtype=torch.float32
        )
        env._whale_sole_geometry = (ids, sizes)
    ids, sizes = env._whale_sole_geometry
    xyz = env.sim.data.geom_xpos[:, ids]
    R = env.sim.data.geom_xmat[:, ids].reshape(env.num_envs, 2, 3, 3)
    return xyz[:, :, 2] - (R[:, :, 2, :].abs() * sizes).sum(-1)


def clearance_cost(
    env, target_height, height_sensor_name, command_name, command_threshold, asset_cfg
):
    v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    cost = (
        (heights(env) - target_height).abs() * torch.linalg.vector_norm(v, dim=-1)
    ).sum(-1)
    active = (
        torch.linalg.vector_norm(env.command_manager.get_command(command_name), dim=-1)
        > command_threshold
    )
    return cost * active


class LandingHeight:
    def __init__(self, cfg, env):
        self.peak = torch.zeros((env.num_envs, 2), device=env.device)

    def __call__(
        self,
        env,
        sensor_name,
        height_sensor_name,
        target_height,
        command_name,
        command_threshold,
    ):
        sensor = env.scene[sensor_name]
        air = sensor.data.found == 0
        # Reset accumulated heights across episode resets; never transfer a previous fall.
        self.peak = torch.where((env.episode_length_buf <= 1)[:, None], 0.0, self.peak)
        self.peak = torch.where(air, torch.maximum(self.peak, heights(env)), self.peak)
        landing = sensor.compute_first_contact(dt=env.step_dt)
        active = (
            torch.linalg.vector_norm(
                env.command_manager.get_command(command_name), dim=-1
            )
            > command_threshold
        )
        cost = (((self.peak / target_height) - 1).square() * landing).sum(-1) * active
        env.extras["log"]["Metrics/whole_sole_peak_height"] = (
            self.peak * landing
        ).sum() / landing.sum().clamp(min=1)
        self.peak = torch.where(landing, 0.0, self.peak)
        return cost
