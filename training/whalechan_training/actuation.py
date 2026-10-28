"""Identical estimated motor limits for mjlab and independent NumPy deployment."""

from dataclasses import dataclass
import numpy as np
from mjlab.actuator.pd_actuator import IdealPdActuatorCfg
from mjlab.actuator.actuator import Actuator
from mjlab.utils.spec import create_position_actuator


def numpy_torque(target, q, qd, kp, kd, caps):
    limit = np.minimum(caps, np.maximum(0.0, 0.93 * (1.0 - np.abs(qd) / 8.48230016)))
    return np.clip(kp * (target - q) - kd * qd, -limit, limit)


@dataclass(kw_only=True)
class EstimatedMotorCfg(IdealPdActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return EstimatedMotor(self, entity, target_ids, target_names)


class EstimatedMotor(Actuator):
    @property
    def command_field(self):
        return "position"

    def edit_spec(self, spec, target_names):
        for name in target_names:
            self._mjs_actuators.append(
                create_position_actuator(
                    spec,
                    name,
                    stiffness=self.cfg.stiffness,
                    damping=self.cfg.damping,
                    effort_limit=self.cfg.effort_limit,
                    armature=self.cfg.armature,
                    frictionloss=self.cfg.frictionloss,
                    viscous_damping=self.cfg.viscous_damping,
                    transmission_type=self.cfg.transmission_type,
                )
            )

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.force_range = model.actuator_forcerange

    def compute(self, cmd):
        limit = (0.93 * (1.0 - cmd.vel.abs() / 8.48230016)).clamp(
            min=0, max=self.cfg.effort_limit
        )
        self.force_range[:, self._global_ctrl_ids, 0] = -limit
        self.force_range[:, self._global_ctrl_ids, 1] = limit
        return cmd.position_target
