"""Named sensor/action mapping for the VD39 observation layout, without device I/O."""

from dataclasses import dataclass

import numpy as np


OBSERVATION_ORDER = [
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"
]


def _finite(values, shape, label):
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and finite values")
    return result


@dataclass(frozen=True)
class EncoderCalibration:
    """Measured encoder zero/sign/ratio; angles are radians, velocities rad/s.

    The ratio is encoder radians per joint radian (1 for joint-side feedback).
    Joint angles are absolute model coordinates, before subtracting joint_home.
    """

    zero_rad: float
    direction: int
    encoder_radians_per_joint_radian: float = 1.0

    def __post_init__(self):
        if (self.direction not in (-1, 1) or not np.isfinite(self.zero_rad)
                or not np.isfinite(self.encoder_radians_per_joint_radian)
                or self.encoder_radians_per_joint_radian <= 0):
            raise ValueError("Encoder calibration needs finite zero, sign +/-1 and positive ratio")

    def joint_position(self, encoder_rad):
        value = _finite(encoder_rad, (), "Encoder position")
        return float(self.direction * (value - self.zero_rad)
                     / self.encoder_radians_per_joint_radian)

    def joint_velocity(self, encoder_rad_s):
        value = _finite(encoder_rad_s, (), "Encoder velocity")
        return float(self.direction * value / self.encoder_radians_per_joint_radian)

    def encoder_target(self, joint_rad):
        value = _finite(joint_rad, (), "Joint target")
        return float(self.zero_rad + self.direction
                     * self.encoder_radians_per_joint_radian * value)


class PolicyIO:
    """Build one observation and decode one action by name, retaining action history.

    Supply the verified bundle contract. The caller owns timing, IMU mounting,
    encoder calibration and actuation; this class opens no hardware connection.
    """

    def __init__(self, contract):
        self.joints = tuple(contract["joints"])
        self.actions = tuple(contract["action_joint_names"])
        n = len(self.joints)
        if (not n or any(not isinstance(s, str) or not s for s in self.joints + self.actions)
                or len(set(self.joints)) != n or len(self.actions) != n
                or set(self.actions) != set(self.joints)):
            raise ValueError("Joint and action names must map one-to-one")
        self.observation_size = 9 + 3 * n
        if (contract["observation_order"] != OBSERVATION_ORDER
                or contract.get("actor_dim", self.observation_size) != self.observation_size):
            raise ValueError("Unsupported observation layout; use its matching adapter")
        self.home = _finite(contract["joint_home"], (n,), "Joint home").copy()
        self.offset = _finite(contract["action_offset"], (n,), "Action offset").copy()
        self.scale = _finite(contract["action_scale"], (n,), "Action scale").copy()
        self.reset()

    def reset(self):
        """Clear previous actions at a new physical episode, not at a pause."""
        self.previous = np.zeros(len(self.actions))

    def _named(self, values, label):
        if set(values) != set(self.joints):
            raise ValueError(f"{label} requires exactly the contract joint names")
        return _finite([values[name] for name in self.joints], (len(self.joints),), label)

    def observe(self, *, gyro_body, gravity_body, joint_position, joint_velocity, command):
        """Return float32 actor inputs; gravity is a downward unit vector in body frame.

        Position/velocity dictionaries use model radians/rad/s. Gyro is body
        rad/s. Commands are [forward m/s, lateral m/s, yaw rad/s]; forward is -Y.
        """
        gyro = _finite(gyro_body, (3,), "Body gyro")
        gravity = _finite(gravity_body, (3,), "Body gravity")
        if not np.isclose(np.linalg.norm(gravity), 1.0, atol=0.01, rtol=0):
            raise ValueError("Body gravity must be a unit vector, not raw acceleration")
        cmd = _finite(command, (3,), "Velocity command")
        with np.errstate(over="ignore", invalid="ignore"):
            obs = np.r_[gyro, gravity,
                        self._named(joint_position, "Joint position") - self.home,
                        self._named(joint_velocity, "Joint velocity"),
                        self.previous, cmd].astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Observation exceeds finite float32 range")
        return obs

    def targets(self, action):
        """Return absolute joint targets by action name and remember the clipped action.

        This is the model target transform, not servo register values or a
        hardware safety limiter. Invalid actions leave the history unchanged.
        """
        clipped = np.clip(_finite(action, (len(self.actions),), "Policy action"), -3, 3)
        with np.errstate(over="ignore", invalid="ignore"):
            targets = self.offset + self.scale * clipped
        if not np.isfinite(targets).all():
            raise ValueError("Nonfinite joint targets")
        self.previous = clipped.copy()
        return dict(zip(self.actions, targets.tolist()))
