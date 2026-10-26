"""VD29 bounded, cancellable head feedback on the unchanged VD28 contact gait.

These are joint-space gestures, not visual tracking or a hardware servo driver.
All targets pass through rate/acceleration limits; completion uses measured joints.
"""
import numpy as np

from .bluefatfish_v28 import ContactWalkController


class FeedbackController(ContactWalkController):
    interaction_gestures = ('nod', 'shake', 'look_left', 'look_right')
    head_rate = .45  # rad/s, commanded targets, not a hardware guarantee
    head_acceleration = 1.8  # rad/s^2
    gesture_segment = .8
    poses = {
        'nod': [[0, 0], [0, .10], [0, -.06], [0, .10], [0, 0]],
        'shake': [[0, 0], [.18, 0], [-.18, 0], [.12, 0], [0, 0]],
        'look_left': [[0, 0], [.22, 0], [.22, 0], [0, 0]],
        'look_right': [[0, 0], [-.22, 0], [-.22, 0], [0, 0]],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_ids = np.array([self.model.actuator(n).id for n in ('head_yaw', 'head_pitch')])
        self.head_action = None
        self.head_started = None
        self.head_returning = False
        self.head_done = True
        self.head_settled_since = None

    def head_settled(self):
        ids = self.head_ids
        target = self.channels.get('head_feedback', np.zeros(2))
        velocity = self.channel_velocity.get('head_feedback', np.zeros(2))
        return (np.max(np.abs(target)) < .004 and np.max(np.abs(velocity)) < .025
                # Existing position actuators retain a gravity-loaded static error.
                # About one degree is neutral; velocity must still settle separately.
                and np.max(np.abs(self.data.qpos[self.qaddresses[ids]])) < .018
                and np.max(np.abs(self.data.qvel[self.dofs[ids]])) < .05)

    def feedback_settled(self):
        """A supported stop must also wait for an interrupted head to return."""
        return not self.head_returning and self.head_settled()

    def request_gesture(self, action):
        if (action not in self.poses or not self.active_dynamics or self.phase != 'ready'
                or not all(self.contacts()) or self.head_action or not self.feedback_settled()):
            return False
        self.head_action = action
        self.head_started = float(self.data.time)
        self.head_done = False
        self.head_settled_since = None
        self.events.append(dict(time=self.head_started, event='head_gesture_started', action=action))
        return True

    def cancel(self):
        super().cancel()
        if self.head_action is not None:
            self.events.append(dict(time=float(self.data.time), event='head_gesture_cancelled', action=self.head_action))
            self.head_action = None
            self.head_returning = True
            self.head_started = float(self.data.time)
            self.head_settled_since = None

    def update_upper(self, command, status):
        super().update_upper(command, status)
        t = float(self.data.time)
        if round(t/self.model.opt.timestep) % 10:
            return
        target = np.zeros(2)
        trajectory_complete = self.head_returning
        if self.head_action:
            poses = np.asarray(self.poses[self.head_action])
            age = t-self.head_started
            segment = int(age/self.gesture_segment)
            trajectory_complete = segment >= len(poses)-1
            if not trajectory_complete:
                s, _, _ = self.trajectory(age-segment*self.gesture_segment, self.gesture_segment)
                target = poses[segment]+s*(poses[segment+1]-poses[segment])
        value = self.smooth('head_feedback', target, rate=self.head_rate,
                            acceleration=self.head_acceleration, frequency=12.)
        self.upper_reference[self.head_ids] = value
        if self.head_action:
            self.expression = 'happy' if self.head_action == 'nod' else 'neutral'
        if trajectory_complete:
            if self.head_settled():
                if self.head_settled_since is None:
                    self.head_settled_since = t
                if t-self.head_settled_since >= .15:
                    self.events.append(dict(time=t, event='head_returned' if self.head_returning else 'head_gesture_completed',
                                            action=self.head_action))
                    self.head_action = None
                    self.head_returning = False
                    self.head_done = True
            else:
                self.head_settled_since = None
        if (self.head_action or self.head_returning) and t-self.head_started > 8:
            raise RuntimeError('head_feedback_settle_timeout')
