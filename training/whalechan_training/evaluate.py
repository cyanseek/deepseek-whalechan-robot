"""Independent native CPU MuJoCo + normalized ONNX. No mjlab, QP, resets or teacher."""

import argparse, json, time, hashlib
from pathlib import Path
from collections import deque
import numpy as np, mujoco, onnxruntime as ort

if __package__:
    from .footsteps import FootstepTracker
    from .bundle import verify
else:
    from footsteps import FootstepTracker
    from bundle import verify


def torque(target, q, qd, kp, kd, caps):
    limit = np.minimum(caps, np.maximum(0.0, 0.93 * (1.0 - np.abs(qd) / 8.48230016)))
    return np.clip(kp * (target - q) - kd * qd, -limit, limit)


class Runtime:
    def __init__(self, run):
        self.run = Path(run)
        self.bundle = verify(self.run)
        self.c = json.loads((self.run / "contract.json").read_text())
        self.m = mujoco.MjModel.from_binary_path(str(self.run / "model.mjb"))
        self.d = mujoco.MjData(self.m)
        c = self.c
        m = self.m
        d = self.d
        d.qpos[:] = c["qpos0"]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.names = c["joints"]
        self.joints = [m.joint("robot/" + s).id for s in self.names]
        self.qids = m.jnt_qposadr[self.joints]
        self.vids = m.jnt_dofadr[self.joints]
        d.qpos[:7] = c["qpos"][:7]
        d.qpos[self.qids] = c["joint_home"]
        mujoco.mj_forward(m, d)
        self.home = np.array(c["joint_home"])
        self.base = m.body("robot/base").id
        self.motornames = [
            m.joint(int(j)).name.split("/")[-1] for j in m.actuator_trnid[:, 0]
        ]
        self.motor_source = [self.names.index(s) for s in self.motornames]
        self.mq = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.mv = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.kp = np.array(c["kp"])[self.motor_source]
        self.kd = np.array(c["kd"])[self.motor_source]
        self.caps = np.array(c["caps"])[self.motor_source]
        self.implicit = bool(np.any(m.actuator_biasprm[:, 1] != 0))
        self.action_to_motor = [
            c["action_joint_names"].index(s) for s in self.motornames
        ]
        self.offset = np.array(c["action_offset"])
        self.scale = np.array(c["action_scale"])
        self.previous = np.zeros(len(self.names))
        self.delay = deque([np.zeros(m.nu), np.zeros(m.nu)], maxlen=2)
        self.feet = [m.geom("robot/" + s + "_sole").id for s in ("left", "right")]
        self.floor = [
            i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ][0]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(
            str(self.run / "policy.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input = self.policy.get_inputs()[0].name

    def obs(self, command):
        d = self.d
        m = self.m
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base, v, 0)
        R = d.xmat[self.base].reshape(3, 3)
        gravity = -R[2, :]
        return np.r_[
            R.T @ v[:3],
            gravity,
            d.qpos[self.qids] - self.home,
            d.qvel[self.vids],
            self.previous,
            command,
        ].astype("float32")

    def targets(self, action):
        action = np.clip(action, -3, 3)
        self.previous = action.copy()
        return (self.offset + self.scale * action)[self.action_to_motor]

    def integrate(self, target):
        delayed = self.delay.popleft()
        self.delay.append(target.copy())
        if self.implicit:
            limit = np.minimum(
                self.caps,
                np.maximum(
                    0.0, 0.93 * (1.0 - np.abs(self.d.qvel[self.mv]) / 8.48230016)
                ),
            )
            self.m.actuator_forcerange[:] = np.stack((-limit, limit), axis=-1)
            self.d.ctrl[:] = delayed
        else:
            self.d.ctrl[:] = torque(
                delayed,
                self.d.qpos[self.mq],
                self.d.qvel[self.mv],
                self.kp,
                self.kd,
                self.caps,
            )
        mujoco.mj_step(self.m, self.d)

    def measurements(self):
        m = self.m
        d = self.d
        normals = np.zeros(2)
        slip = np.zeros(2)
        bad = []
        for contact_id, c in enumerate(d.contact[: d.ncon]):
            if self.floor not in c.geom:
                continue
            other = int(c.geom[1] if c.geom[0] == self.floor else c.geom[0])
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, contact_id, f)
            if other in self.feet:
                side = self.feet.index(other)
                normals[side] += max(0.0, f[0])
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, c.pos, int(m.geom_bodyid[other]))
                if f[0] > 0.3:
                    slip[side] = max(
                        slip[side], float(np.linalg.norm((jac @ d.qvel)[:2]))
                    )
            elif f[0] > 0.3:
                bad.append(m.geom(other).name)
        pos = d.geom_xpos[self.feet].copy()
        clear = []
        for g in self.feet:
            R = d.geom_xmat[g].reshape(3, 3)
            clear.append(float(d.geom_xpos[g, 2] - np.abs(R[2]) @ m.geom_size[g]))
        tilt = float(np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1, 1)))
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base, v, 0)
        return (
            pos,
            np.array(clear),
            normals,
            slip,
            bad,
            tilt,
            -(d.xmat[self.base].reshape(3, 3).T @ v[3:])[1],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--speed", type=float, default=0.05)
    p.add_argument("--start-immediately", action="store_true")
    a = p.parse_args()
    if not np.isfinite(a.seconds) or a.seconds < 0.02 or not np.isfinite(a.speed):
        p.error("--seconds must be at least 0.02 and --speed must be finite")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    r = Runtime(a.run)
    tracker = FootstepTracker()
    rows = []
    slip = np.zeros(2)
    start = r.d.qpos[:3].copy()
    failure = None
    max_tilt = 0.0
    stops = []
    stable_since = None
    before = time.perf_counter()
    initial_forward = -r.d.xmat[r.base].reshape(3, 3)[:, 1].copy()
    phase_records = []
    for k in range(round(a.seconds / 0.02)):
        t = k * 0.02
        walk = ((0 if a.start_immediately else 2) <= t < 8) or (11 <= t < 17)
        command = np.array([a.speed if walk else 0.0, 0.0, 0.0])
        obs = r.obs(command)
        action = r.policy.run(None, {r.input: obs[None]})[0][0]
        target = r.targets(action)
        for _ in range(20):
            r.integrate(target)
            pos, clear, normal, s, bad, tilt, vx = r.measurements()
            slip += s * 0.001 if walk else 0.0
            forward = -r.d.xmat[r.base].reshape(3, 3)[:2, 1]
            tracker.sample(r.d.time, pos, clear, normal > 0.3, forward, walk)
            max_tilt = max(max_tilt, tilt)
            if bad or tilt > np.radians(40) or not np.isfinite(r.d.qpos).all():
                failure = {
                    "time_s": float(r.d.time),
                    "contacts": bad,
                    "tilt_deg": float(np.degrees(tilt)),
                }
                break
        mujoco.mj_forward(r.m, r.d)
        pos, clear, normal, s, bad_at_frame, tilt, vx = r.measurements()
        # Renderer uses original own-body joint/root order, unaffected by mjlab prefix.
        body_v = np.zeros(6)
        mujoco.mj_objectVelocity(r.m, r.d, mujoco.mjtObj.mjOBJ_BODY, r.base, body_v, 0)
        horizontal_speed = float(np.linalg.norm(body_v[3:5]))
        angular_speed = float(np.linalg.norm(body_v[:3]))
        heading = -r.d.xmat[r.base].reshape(3, 3)[:, 1]
        heading_error = float(
            np.arctan2(
                initial_forward[0] * heading[1] - initial_forward[1] * heading[0],
                initial_forward[:2] @ heading[:2],
            )
        )
        rows.append(
            {
                "t": float(r.d.time),
                "qpos": r.d.qpos.tolist(),
                "command": command.tolist(),
                "action": action.tolist(),
                "obs": obs.tolist(),
                "clearance_m": clear.tolist(),
                "tilt_rad": tilt,
                "normal_force": normal.tolist(),
                "speed_m_s": vx,
                "horizontal_speed_m_s": horizontal_speed,
                "angular_speed_rad_s": angular_speed,
                "heading_error_rad": heading_error,
                "steps": tracker.report()["confirmed_steps_by_foot"],
            }
        )
        stop_start = 8.0 if 8 <= t < 11 else (17.0 if t >= 17 else None)
        stable = (
            stop_start is not None
            and horizontal_speed < 0.008
            and angular_speed < 0.2
            and tilt < np.radians(12)
            and bool((normal > 0.3).all())
        )
        if stable:
            if stable_since is None:
                stable_since = float(r.d.time)
            if r.d.time - stable_since >= 0.5 and not any(
                x["command_s"] == stop_start for x in stops
            ):
                stops.append(
                    {
                        "command_s": stop_start,
                        "settled_s": stable_since,
                        "latency_s": stable_since - stop_start,
                    }
                )
        else:
            stable_since = None
        if failure:
            break
    report = {
        **tracker.report(),
        "engine": mujoco.__version__,
        "sim_seconds": float(r.d.time),
        "wall_seconds": time.perf_counter() - before,
        "failure": failure,
        "max_tilt_deg": float(np.degrees(max_tilt)),
        "forward_displacement_m": float((r.d.qpos[:3] - start) @ initial_forward),
        "walking_slip_m": slip.tolist(),
        "stops": stops,
        "neural_policy": True,
        "qp_calls": 0,
        "eval_resets": 0,
        "initialization": "home state; policy from first control tick; no teacher startup",
        "policy_sha256": hashlib.sha256(
            (r.run / "policy.onnx").read_bytes()
        ).hexdigest(),
    }
    report["evaluation_schema"] = (
        "v2: horizontal speed <8 mm/s AND angular speed <0.2 rad/s sustained 0.5 s; independent contact steps"
    )
    report["scenario"] = {
        "speed_m_s": a.speed,
        "initial_stand_s": 0 if a.start_immediately else 2,
        "requested_seconds": a.seconds,
    }
    report["bundle_verification"] = r.bundle
    report["max_heading_error_deg"] = float(
        np.degrees(max(abs(x["heading_error_rad"]) for x in rows))
    )
    steps = report["confirmed_steps_by_foot"]
    report["qualified"] = bool(
        failure is None
        and min(steps) >= 3
        and report["alternating_transitions"] >= 5
        and len(stops) == 2
        and max(slip) < 0.04
        and report["forward_displacement_m"] > 0.12
        and report["max_heading_error_deg"] < 20
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "trajectory.json").write_text(json.dumps(rows))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "footstep_events"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
