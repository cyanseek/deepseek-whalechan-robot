"""VD26 experimental bounded-step navigation on the unchanged VD24 body.

This remains prescribed IK/PD with ideal simulated proprioception. It does not
provide robust long-distance walking, running or a learned locomotion policy.
"""
import numpy as np
from .bluefatfish_v24 import LivelyController
from .bluefatfish_v2 import yaw_matrix


class MeasuredStepController(LivelyController):
    """Keep a validated full small step while RGB decides direction and stopping."""
    arrival_distance = .35  # RGB marker range; close off-axis targets can leave the image.
    arrival_bearing = .50  # Stop near a visible person marker; exact facing is separate.
    interaction_jump = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crouch_depth = .004
        self.extension_drop = .002

    def _takeoff_confirmed(self, time):
        # Require upward motion as well as contact separation.
        return (time - self.phase_t >= .015 and not any(self.contacts())
                and self.data.qvel[2] > .02)

    def request_jump(self):
        if not super().request_jump():
            return False
        # Re-anchor XY from measured support but converge to the same crouched
        # support height, rather than freezing an arbitrary idle sway sample.
        self.root_target[:2] = (self.foot_targets.mean(axis=0)
                               + yaw_matrix(self.heading) @ np.array([.006, -.0006, 0.]))[:2]
        self.root_target[2] = self.rest_root[2] - .0091
        self.pre_jump_root = self.root_target.copy()
        return True

    def _jump_preparation_ready(self, time):
        return super()._jump_preparation_ready(time) and time - self.prepare_time >= 2.5

    def set_precision_motion(self, enabled):
        # VD24's continuously scaled navigation stride had different dynamics
        # from its tested manual gait. VD26 uses the existing lively timing;
        # this changes the command-to-step mapping, not the physical body.
        super().set_precision_motion(False)

    def update(self, command, status='idle', walking=True):
        command = np.asarray(command, dtype=float).copy()
        if command[0] > .01:
            command[0] = .12
        old_phase = self.phase
        super().update(command, status, walking)
        if old_phase == 'shift' and self.phase == 'swing' and not self.align_for_jump:
            self.step_yaw = float(np.clip(self.step_yaw, -.012, .012))
            self.heading_end = self.heading + self.step_yaw
            center = self.foot_targets.mean(axis=0)
            offset = self.swing_start - center
            dx = float(np.clip(command[0] * self.step_scale, -.008, self.max_forward_step))
            dy = float(np.clip(command[1] * .08, -.006, .006))
            # Use the same endpoint expression at zero and nonzero yaw. The
            # old stance-relative branch changed the stride at tiny yaw values.
            self.swing_end = (center + yaw_matrix(self.step_yaw) @ offset
                              + yaw_matrix(self.heading) @ np.array([dx, dy, 0.]))
            self.swing_end[2] = self.swing_start[2]
