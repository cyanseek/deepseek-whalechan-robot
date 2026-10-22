"""Observable native-contact metrics, independent of the motion planner."""
import numpy as np
import mujoco
from .bluefatfish import R_FORWARD


class MotionMetrics:
    def __init__(self, model, low):
        self.model, self.low = model, low
        self.soles = [model.geom(s + '_sole').id for s in ('left', 'right')]
        self.floor = model.geom('floor').id
        self.other_floor = set()
        self.max_tilt = self.slip_path = self.loaded_time = self.walk_time = 0.
        self.walk_distance = self.max_clearance = self.max_air_time = self.air_time = 0.
        self.walk_start = None
        self.walk_end = None
        self.last_root = None
        self.foot_peak = np.zeros(2)
        self.max_force = np.zeros(model.nu)
        self.samples = 0

    def sample(self, data, walking=False, dt=.01):
        m, low = self.model, self.low
        rotation = data.xmat[low.base].reshape(3, 3) @ R_FORWARD.T
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        self.max_force = np.maximum(self.max_force, np.abs(data.actuator_force))
        clearance = np.array([data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g]
                              for g in self.soles])
        points = [[], []]; loads = [[], []]
        for ci, contact in enumerate(data.contact):
            if self.floor not in (contact.geom1, contact.geom2):
                continue
            g = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            if g not in self.soles:
                self.other_floor.add(m.geom(g).name)
                continue
            force = np.zeros(6); mujoco.mj_contactForce(m, data, ci, force)
            if force[0] > .1:
                side = self.soles.index(g)
                points[side].append(contact.pos.copy()); loads[side].append(force[0])
        speeds = []
        for side in range(2):
            if not loads[side]:
                continue
            point = np.average(points[side], axis=0, weights=loads[side])
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jac(m, data, jp, jr, point, low.feet[side])
            speed = float(np.linalg.norm((jp @ data.qvel)[:2]))
            speeds.append(speed)
            if walking:
                self.slip_path += speed * dt; self.loaded_time += dt
        if walking:
            if self.walk_start is None:
                self.walk_start = data.qpos[:3].copy()
            if self.last_root is not None:
                self.walk_distance += float(np.linalg.norm(data.qpos[:2] - self.last_root[:2]))
            self.walk_end = data.qpos[:3].copy()
            self.walk_time += dt
            self.foot_peak = np.maximum(self.foot_peak, clearance)
        self.last_root = data.qpos[:3].copy() if walking else None
        if low.phase in ('launch', 'flight') and not any(low.contacts()):
            self.max_clearance = max(self.max_clearance, float(min(clearance)))
            self.air_time += dt; self.max_air_time = max(self.max_air_time, self.air_time)
        else:
            self.air_time = 0.
        self.samples += 1
        return dict(clearance_m=clearance.tolist(), loaded_contact_speed_m_s=speeds, tilt_rad=tilt)

    def report(self):
        delta = self.walk_end - self.walk_start if self.walk_start is not None else np.zeros(3)
        distance = float(np.linalg.norm(delta[:2]))
        return dict(walking_time_s=self.walk_time, walking_net_displacement_m=delta.tolist(),
                    walking_path_m=self.walk_distance, net_speed_m_s=distance / max(self.walk_time, .001),
                    loaded_contact_point_slip_m=self.slip_path,
                    mean_loaded_contact_point_speed_m_s=self.slip_path / max(self.loaded_time, .001),
                    slip_per_net_m=self.slip_path / max(distance, .001),
                    walking_peak_sole_clearance_m=self.foot_peak.tolist(),
                    jump_both_soles_clearance_m=self.max_clearance, jump_max_air_time_s=self.max_air_time,
                    max_tilt_deg=float(np.degrees(self.max_tilt)), nonfoot_floor_contacts=sorted(self.other_floor),
                    max_actuator_force=self.max_force.tolist(), metric_samples=self.samples,
                    definition='100 Hz force-weighted loaded sole contact-point XY velocity, load > 0.1 N; includes touchdown/rolling; ideal MuJoCo proprioception. Jump time resolution 10 ms.')
