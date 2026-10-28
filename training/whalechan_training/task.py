"""Own-body adaptation of mjlab velocity template and MicroDuck locomotion recipe.

Reference: pollen-robotics/microduck_rl b02fdea (Apache-2.0).
No Duck body, policy weights, or teacher actions are used.
"""

import json, math
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import mujoco, torch
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.managers import (
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg

if __package__:
    from .actuation import EstimatedMotorCfg
    from .sole_geometry import clearance_cost, LandingHeight
else:
    from actuation import EstimatedMotorCfg
    from sole_geometry import clearance_cost, LandingHeight

BODY = Path(__file__).parent / "body"
C = json.loads((BODY / "contract.json").read_text())
NAMES = C["joints"]
LEGS = [n for n in NAMES if any(s in n for s in ("hip_", "knee", "ankle_"))]


def get_spec():
    return mujoco.MjSpec.from_file(str(BODY / "robot.xml"))


def body_velocity(env):
    v = env.scene["robot"].data.root_link_lin_vel_b
    return torch.stack((-v[:, 1], v[:, 0], v[:, 2]), dim=-1)


def angular_velocity(env):
    return env.scene["robot"].data.root_link_ang_vel_b


def linear_tracking(env):
    error = (
        (body_velocity(env)[:, :2] - env.command_manager.get_command("twist")[:, :2])
        .square()
        .sum(-1)
    )
    return torch.exp(-error / 0.0036)


def angular_tracking(env):
    error = angular_velocity(env)[:, 2] - env.command_manager.get_command("twist")[:, 2]
    return torch.exp(-error.square() / 0.25)


def stop_still(env):
    idle = (
        torch.linalg.vector_norm(env.command_manager.get_command("twist"), dim=-1)
        < 0.005
    )
    return idle.float() * (
        body_velocity(env).square().sum(-1)
        + 0.05 * env.scene["robot"].data.joint_vel.square().mean(-1)
    )


def invalid_state(env):
    return ~torch.isfinite(env.scene["robot"].data.root_link_pos_w).all(-1)


@requires_model_fields("actuator_forcerange")
def expand_motor_fields(env, env_ids=None):
    pass


def body_ground(env):
    # Explicit forbidden-contact sensor, not a root-height proxy.
    return (env.scene["body_ground"].data.found > 0).flatten(1).any(-1)


class OwnVelocityCommand(UniformVelocityCommand):
    def _update_metrics(self):
        steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.linalg.vector_norm(
                self.vel_command_b[:, :2] - body_velocity(self._env)[:, :2], dim=-1
            )
            / steps
        )
        self.metrics["error_vel_yaw"] += (
            self.vel_command_b[:, 2] - angular_velocity(self._env)[:, 2]
        ).abs() / steps


@dataclass(kw_only=True)
class OwnVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return OwnVelocityCommand(self, env)


def make_cfg(n=64, play=False, recipe="baseline"):
    cfg = make_velocity_env_cfg()
    groups = {}
    for i, name in enumerate(NAMES):
        groups.setdefault((C["kp"][i], C["kd"][i], C["caps"][i]), []).append(name)
    motors = tuple(
        EstimatedMotorCfg(
            target_names_expr=tuple(names),
            stiffness=kp,
            damping=kd,
            effort_limit=cap,
            delay_min_lag=2,
            delay_max_lag=2,
        )
        for (kp, kd, cap), names in groups.items()
    )
    robot = EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(C["qpos"][:3]),
            rot=tuple(C["qpos"][3:7]),
            joint_pos=dict(zip(NAMES, C["joint_home"])),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=motors, soft_joint_pos_limit_factor=0.95
        ),
    )
    cfg.scene.entities = {"robot": robot}
    cfg.scene.num_envs = n
    cfg.scene.extent = 1.0
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    feet = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=r"^(left_sole|right_sole)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    bad = ContactSensorCfg(
        name="body_ground",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(?!left_sole$|right_sole$).*collision.*$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    height = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s + "_foot", entity="robot")
            for s in ("left", "right")
        ),
        pattern=RingPatternCfg.single_ring(radius=0.004, num_samples=4),
        ray_alignment="yaw",
        max_distance=0.4,
        include_geom_groups=(0,),
        exclude_parent_body=True,
    )
    cfg.scene.sensors = (feet, bad, height)
    cfg.sim.mujoco.timestep = 0.001
    cfg.decimation = 20
    cfg.sim.mujoco.iterations = 40
    cfg.sim.mujoco.ls_iterations = 20
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.tolerance = 1e-8
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 512
    cfg.episode_length_s = 12.0
    cfg.curriculum = {}
    cfg.viewer.body_name = "base"
    cfg.viewer.distance = 0.65
    # Exact order is exported and reused; body forward is local -Y, not +X.
    for group in ("actor", "critic"):
        terms = deepcopy(cfg.observations[group].terms)
        for name in (
            "height_scan",
            "base_lin_vel",
            "foot_height",
            "foot_air_time",
            "foot_contact",
            "foot_contact_forces",
        ):
            terms.pop(name, None)
        terms["base_ang_vel"].func = angular_velocity
        terms["base_ang_vel"].params = {}
        for name in ("joint_pos", "joint_vel"):
            terms[name].params = {
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=tuple(NAMES), preserve_order=True
                )
            }
        terms["joint_pos"].params["biased"] = group == "actor"
        if group == "critic":
            terms["base_lin_vel"] = ObservationTermCfg(func=body_velocity)
        cfg.observations[group].terms = terms
        cfg.observations[group].enable_corruption = group == "actor" and not play
    cfg.observations["actor"].terms["joint_vel"].noise.n_min = -0.1
    cfg.observations["actor"].terms["joint_vel"].noise.n_max = 0.1
    action = cfg.actions["joint_pos"]
    action.actuator_names = tuple(NAMES)
    action.preserve_order = True
    action.scale = {n: (0.35 if n in LEGS else 0.04) for n in NAMES}
    action.use_default_offset = False
    action.offset = dict(zip(NAMES, C["ctrl"]))
    cmd = cfg.commands["twist"]
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.2
    cmd.rel_forward_envs = 0.0
    cmd.ranges.lin_vel_x = (0.025, 0.08)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.resampling_time_range = (2.0, 4.0)
    # Explicitly forbid velocity injection and world/body axis mixing.
    cmd.init_velocity_prob = 0.0
    cmd.rel_world_envs = 0.0
    cmd.debug_vis = False
    cfg.commands["twist"] = OwnVelocityCommandCfg(**vars(cmd))
    cfg.events.pop("push_robot", None)
    cfg.events["expand_motor_fields"] = EventTermCfg(
        func=expand_motor_fields, mode="startup"
    )
    cfg.events["reset_base"].params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.001, 0.003),
            "yaw": (-0.05, 0.05),
        },
        "velocity_range": {},
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
    cfg.events["foot_friction"].params.update(
        asset_cfg=SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
        ranges=(0.8, 1.1),
    )
    cfg.events["base_com"].params.update(
        asset_cfg=SceneEntityCfg("robot", body_names=("base",)),
        ranges={0: (-0.001, 0.001), 1: (-0.001, 0.001), 2: (-0.001, 0.001)},
    )
    if play:
        for name in ("encoder_bias", "foot_friction", "base_com"):
            cfg.events.pop(name, None)
        cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        cfg.events["reset_base"].params = {"pose_range": {}, "velocity_range": {}}
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=linear_tracking, weight=2.0
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
        func=angular_tracking, weight=0.5
    )
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["pose"].params.update(
        std_standing={".*": 0.15},
        std_walking={".*": 0.4},
        std_running={".*": 0.4},
        walking_threshold=0.005,
    )
    cfg.rewards["pose"].weight = 0.3
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards.pop("angular_momentum")
    cfg.rewards.pop("soft_landing")
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params.update(
        threshold_min=0.10, threshold_max=0.35, command_threshold=0.005
    )
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = ("left_foot", "right_foot")
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params.update(target_height=0.008, command_threshold=0.005)
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.005
    cfg.rewards["stop_still"] = RewardTermCfg(func=stop_still, weight=-2.0)
    if recipe == "fullsole":
        cfg.rewards["foot_clearance"].func = clearance_cost
        cfg.rewards["foot_swing_height"].func = LandingHeight
    cfg.terminations.pop("out_of_terrain_bounds")
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(40)
    cfg.terminations["body_ground"] = TerminationTermCfg(func=body_ground)
    cfg.terminations["invalid"] = TerminationTermCfg(func=invalid_state)
    cfg.seed = 42
    return cfg


def rl_cfg():
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
        logger="tensorboard",
        upload_model=False,
        save_interval=100,
        num_steps_per_env=24,
        seed=42,
        clip_actions=3.0,
    )
