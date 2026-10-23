"""VD28 experimental contact-constrained straight walking on the VD24 body.

Uses ideal MuJoCo proprioception, prescribed COM/foot trajectories and bounded
inverse dynamics. This is not a learned policy or hardware controller. Only
actuator targets are written; the free body, contacts and force limits are intact.
"""
import clarabel
import mujoco
import numpy as np
from scipy import sparse

from .bluefatfish_v26 import MeasuredStepController


class ContactWalkController(MeasuredStepController):
    control_stride = 2  # 500 Hz planning, 1 kHz native integration/actuation.
    interaction_jump = False
    navigation_bearing_limit = .25  # Straight corridor only; VD26 retains turns/hops.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_seconds = .5
        self.swing_seconds = .6
        self.center_seconds = .3
        self.stride = .024
        self.step_lift = .008
        self.com_drop = .006
        self.com_fraction = .65
        self.soles = [self.model.geom(s+'_sole').id for s in ('left', 'right')]
        self.dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.qaddresses = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.leg_ids = np.asarray(self.aids).ravel()
        self.upper_ids = np.setdiff1d(np.arange(self.model.nu), self.leg_ids)
        self.active_dynamics = False
        self.torque = None
        self.previous_jcom = None
        self.control_dt = self.control_stride*self.model.opt.timestep
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)
        self.foot_velocity = np.zeros((2, 3))
        self.foot_acceleration = np.zeros((2, 3))
        self.solver_residual = 0.
        self.qp_count = 0
        self.stop_requested = False
        self.coordinate_upper = True
        self.motion_mode = 'contact_straight'

    @staticmethod
    def trajectory(time, duration):
        u = float(np.clip(time/duration, 0, 1))
        return (10*u**3-15*u**4+6*u**5,
                (30*u**2-60*u**3+30*u**4)/duration,
                (60*u-180*u**2+120*u**3)/(duration*duration))

    @staticmethod
    def angle(actual, desired):
        return .5*sum((np.cross(actual[:, k], desired[:, k]) for k in range(3)), start=np.zeros(3))

    def point(self, side):
        g = self.soles[side]
        return self.data.geom_xpos[g]-self.data.geom_xmat[g].reshape(3, 3)[:, 2]*self.model.geom_size[g, 2]

    def loaded(self):
        forces = np.zeros(2)
        floor = self.model.geom('floor').id
        for ci, ct in enumerate(self.data.contact):
            if floor not in (ct.geom1, ct.geom2):
                continue
            g = ct.geom2 if ct.geom1 == floor else ct.geom1
            if g in self.soles:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, ci, f)
                forces[self.soles.index(g)] += max(0, float(f[0]))
        return forces

    def enter(self, phase):
        self.phase = phase
        self.phase_t = float(self.data.time)
        self.events.append(dict(time=self.phase_t, event='contact_walk_'+phase))
        self.com_start = self.com_target.copy()
        self.com_end = self.com_target.copy()
        self.com_velocity[:] = 0
        self.com_acceleration[:] = 0
        if phase == 'shift':
            center = self.anchors.mean(axis=0)
            self.com_end[:2] = (center+self.com_fraction*(self.anchors[1-self.swing]-center))[:2]
            self.com_end[2] = self.com_height-self.com_drop
        elif phase == 'center':
            self.com_end[:2] = self.anchors.mean(axis=0)[:2]
            self.com_end[2] = self.com_height
        elif phase == 'swing':
            self.start_point = self.anchors[self.swing].copy()
            self.end_point = self.start_point+[self.stride, 0, 0]

    def move_com(self, age):
        duration = self.center_seconds if self.phase == 'center' else self.shift_seconds
        s, v, a = self.trajectory(age, duration)
        delta = self.com_end-self.com_start
        self.com_target = self.com_start+s*delta
        self.com_velocity = v*delta
        self.com_acceleration = a*delta

    def cancel(self):
        # Finish an airborne foot placement; never disable supporting torque.
        self.stop_requested = True
        self.wave_start = None
        self.wave_done = True
        if self.tail_started is not None:
            self.events.append(dict(time=float(self.data.time), event='tail_gesture_cancelled'))
        self.tail_started = None
        self.events.append(dict(time=float(self.data.time), event='contact_walk_cancel_requested', phase=self.phase))

    def set_precision_motion(self, enabled):
        self.motion_mode = 'contact_straight'

    def request_jump(self):
        return False

    def update_upper(self, command, status):
        t = float(self.data.time)
        if round(t/self.model.opt.timestep) % 10:
            return
        self.motion_request = np.asarray(command[:3]).copy()
        self.motion_request[2] = 0  # This profile does not implement steering.
        if status == 'arrived' and self.phase == 'ready' and self.wave_start is None and not self.wave_done:
            self.wave_start = t
            self.events.append(dict(time=t, event='wave_started'))
        weight = 0.
        if self.wave_start is not None:
            age = t-self.wave_start
            if status != 'arrived':
                self.wave_start = None
                self.wave_done = True
            elif age < 1:
                weight = self.ease(age)
            elif age < 2.4:
                weight = 1.
            elif age < 3.5:
                weight = 1-self.ease((age-2.4)/1.1)
            else:
                self.wave_start = None
                self.wave_done = True
                self.events.append(dict(time=t, event='wave_completed'))
        upper_command = np.asarray(command).copy()
        upper_command[2] = 0
        for name, value in {**self._upper_targets(t, weight, upper_command), **self._tail_targets(t, weight)}.items():
            self.upper_reference[self.model.actuator(name).id] = value
        self.expression = 'happy' if self.wave_start is not None else 'delight' if self.tail_started is not None else 'neutral'

    def update(self, command, status='idle', walking=True):
        m, d = self.model, self.data
        now = float(d.time)
        if now < 2.5:
            if round(now/m.opt.timestep) % 10:
                return
            return super().update(np.zeros(7), 'idle', False)
        mujoco.mj_forward(m, d)
        if not self.active_dynamics:
            self.active_dynamics = True
            self.com_target = d.subtree_com[self.base].copy()
            self.com_height = float(self.com_target[2])
            self.anchors = np.array([self.point(i) for i in range(2)])
            self.foot_rot = [d.xmat[b].reshape(3, 3).copy() for b in self.feet]
            self.base_rot = d.xmat[self.base].reshape(3, 3).copy()
            self.posture = d.qpos[self.qaddresses].copy()
            self.upper_reference = d.ctrl.copy()
            self.enter('ready')
        active = walking and status == 'approach' and command[0] > .01
        self.stop_requested = not active
        age = now-self.phase_t
        if self.phase == 'ready' and active:
            self.enter('shift')
        elif self.phase == 'shift':
            self.move_com(age)
            if not active:
                self.com_target = d.subtree_com[self.base].copy()
                self.enter('center')
            elif age >= self.shift_seconds:
                loads = self.loaded()
                ready = (loads[1-self.swing] > .70*max(1., sum(loads)) and
                         np.linalg.norm((d.subtree_com[self.base]-self.com_target)[:2]) < .004)
                if age > self.shift_seconds+1.5 and not ready:
                    raise RuntimeError('load_transfer_timeout')
                if ready:
                    self.enter('swing')
                else:
                    self.com_velocity[:] = 0
                    self.com_acceleration[:] = 0
        elif self.phase == 'swing' and age >= self.swing_seconds:
            self.enter('land')
        elif self.phase == 'land':
            # Flatness concerns the sole normal, not its yaw around the floor.
            flat = (abs(self.point(self.swing)[2]-self.end_point[2]) < .001 and
                    np.linalg.norm((self.point(self.swing)-self.end_point)[:2]) < .003 and
                    d.geom_xmat[self.soles[self.swing]].reshape(3, 3)[2, 2] > np.cos(.05))
            if age >= .15 and self.contacts()[self.swing] and flat:
                self.anchors[self.swing] = self.point(self.swing)
                self.anchors[self.swing, 2] = self.end_point[2]
                self.steps_completed += 1
                self.swing = 1-self.swing
                self.enter('center')
            elif age > 2:
                raise RuntimeError('contact_walk_landing_timeout')
        elif self.phase == 'center':
            self.move_com(age)
            if age > self.center_seconds:
                self.enter('ready')
        references = self.anchors.copy()
        supports = [True, True]
        self.foot_velocity[:] = 0
        self.foot_acceleration[:] = 0
        if self.phase in ('swing', 'land'):
            u = float(np.clip((now-self.phase_t)/self.swing_seconds, 0, 1)) if self.phase == 'swing' else 1.
            transfer, tv, ta = self.trajectory((u-.25)*self.swing_seconds, .5*self.swing_seconds)
            if u < .25:
                lift, lv, la = self.trajectory(u*self.swing_seconds, .25*self.swing_seconds)
            elif u > .75:
                lower, lv, la = self.trajectory((u-.75)*self.swing_seconds, .25*self.swing_seconds)
                lift, lv, la = 1-lower, -lv, -la
            else:
                lift, lv, la = 1., 0., 0.
            references[self.swing] = self.start_point+transfer*(self.end_point-self.start_point)
            references[self.swing, 2] += lift*self.step_lift
            self.foot_velocity[self.swing] = tv*(self.end_point-self.start_point)+[0, 0, lv*self.step_lift]
            self.foot_acceleration[self.swing] = ta*(self.end_point-self.start_point)+[0, 0, la*self.step_lift]
            supports[self.swing] = self.phase == 'land' and self.contacts()[self.swing]
        self.references, self.supports = references, supports
        if self.coordinate_upper:
            self.update_upper(command, status)
        self.solve(references, supports)

    def solve(self, references, supports):
        m, d = self.model, self.data
        n, nu = m.nv, m.nu
        N = n+nu+12
        mass = np.zeros((n, n))
        mujoco.mj_fullM(m, mass, d.qM)
        h = d.qfrc_bias-d.qfrc_passive
        jacobians, dots, tasks, goals = [], [], [], []
        jp = np.zeros((3, n)); jr = np.zeros((3, n))
        mujoco.mj_jacSubtreeCom(m, d, jp, self.base)
        jdot = np.zeros(3) if self.previous_jcom is None else ((jp-self.previous_jcom)/self.control_dt)@d.qvel
        self.previous_jcom = jp.copy()
        tasks.append(jp.copy()*30)
        goals.append((100*(self.com_target-d.subtree_com[self.base])-20*jp@d.qvel-jdot+
                      self.com_acceleration+20*self.com_velocity)*30)
        mujoco.mj_jacBody(m, d, jp, jr, self.base)
        tasks.append(jr.copy()*10)
        goals.append((80*self.angle(d.xmat[self.base].reshape(3, 3), self.base_rot)-18*jr@d.qvel)*10)
        for i, body in enumerate(self.feet):
            point = self.point(i)
            mujoco.mj_jac(m, d, jp, jr, point, body)
            J = np.vstack((jp, jr)); jacobians.append(J)
            dp = np.zeros_like(jp); dr = np.zeros_like(jr)
            mujoco.mj_jacDot(m, d, dp, dr, point, body)
            dots.append(np.vstack((dp, dr))@d.qvel)
            goal = np.r_[150*(references[i]-point)-25*jp@d.qvel,
                         80*self.angle(d.xmat[body].reshape(3, 3), self.foot_rot[i])-18*jr@d.qvel]-dots[-1]
            goal[:3] += self.foot_acceleration[i]+25*self.foot_velocity[i]
            if not supports[i]:
                w = np.array([30, 30, 30, 2, 2, 2])
                tasks.append(J*w[:, None]); goals.append(goal*w)
        posture = np.zeros((nu, n)); posture[np.arange(nu), self.dofs] = 1
        tasks.append(posture*.05)
        goals.append((30*(self.posture-d.qpos[self.qaddresses])-8*d.qvel[self.dofs])*.05)
        A = np.zeros((sum(len(t) for t in tasks), N)); A[:, :n] = np.vstack(tasks)
        b = np.concatenate(goals)
        # Regularize internal contact forces. Acceleration tasks alone do not
        # determine a unique double-support wrench in a compliant simulation.
        if sum(supports) == 1:
            fractions = np.asarray(supports, dtype=float)
        else:
            left = np.clip((self.com_target[1]-self.anchors[1, 1])/(self.anchors[0, 1]-self.anchors[1, 1]), .05, .95)
            fractions = np.array([left, 1-left])
        desired = np.zeros(12); desired[[2, 8]] = float(m.body_subtreemass[self.base])*9.81*fractions
        w = np.tile([1, 1, 1, 30, 30, 30], 2)
        extra = np.zeros((12, N)); extra[:, n+nu:] = np.diag(w)
        A = np.vstack((A, extra)); b = np.r_[b, desired*w]
        regularization = np.full(N, .01); regularization[:n] = 1e-8
        P = (A.T@A+np.diag(regularization))/1000
        q = -A.T@b/1000
        G = np.zeros((n, nu+12)); G[self.dofs, np.arange(nu)] = 1; G[:, nu:] = np.vstack(jacobians).T
        C = [np.hstack((mass, -G))]; lo = [-h]; hi = [-h]
        kp = m.actuator_gainprm[:, 0]; kv = -m.actuator_biasprm[:, 2]
        lower = np.maximum(m.actuator_forcerange[:, 0], kp*(m.actuator_ctrlrange[:, 0]-d.qpos[self.qaddresses])-kv*d.qvel[self.dofs])
        upper = np.minimum(m.actuator_forcerange[:, 1], kp*(m.actuator_ctrlrange[:, 1]-d.qpos[self.qaddresses])-kv*d.qvel[self.dofs])
        block = np.zeros((nu, N)); block[:, n:n+nu] = np.eye(nu)
        C.append(block); lo.append(lower); hi.append(upper)
        tau = np.clip(kp*(self.upper_reference-d.qpos[self.qaddresses])-kv*d.qvel[self.dofs], lower, upper)
        C.append(block[self.upper_ids]); lo.append(tau[self.upper_ids]); hi.append(tau[self.upper_ids])
        dofs = self.dofs[self.leg_ids]
        limits = m.jnt_range[m.actuator_trnid[self.leg_ids, 0]]
        position = d.qpos[self.qaddresses[self.leg_ids]]; velocity = d.qvel[dofs]
        block = np.zeros((len(dofs), N)); block[np.arange(len(dofs)), dofs] = 1
        C.append(block); lo.append(80*(limits[:, 0]+.015-position)-18*velocity); hi.append(80*(limits[:, 1]-.015-position)-18*velocity)
        for i in range(2):
            j = n+nu+i*6
            if not supports[i]:
                block = np.zeros((6, N)); block[:, j:j+6] = np.eye(6)
                C.append(block); lo.append(np.zeros(6)); hi.append(np.zeros(6))
                continue
            J = jacobians[i]
            err = np.r_[references[i]-self.point(i), self.angle(d.xmat[self.feet[i]].reshape(3, 3), self.foot_rot[i])]
            required = 100*err-20*J@d.qvel-dots[i]
            block = np.zeros((6, N)); block[:, :n] = J
            C.append(block); lo.append(required); hi.append(required)
            for axis, scale in [(0, .7), (1, .7), (3, .020), (4, .014), (5, .006)]:
                for sign in (-1, 1):
                    row = np.zeros((1, N)); row[0, j+axis] = sign; row[0, j+2] = -scale
                    C.append(row); lo.append([-np.inf]); hi.append([0])
            row = np.zeros((1, N)); row[0, j+2] = 1
            C.append(row); lo.append([0]); hi.append([25])
        C = np.vstack(C); lo = np.concatenate(lo); hi = np.concatenate(hi)
        self.solver_debug = dict(P=P, q=q, C=C, lo=lo, hi=hi)
        eq = lo == hi; upper = np.isfinite(hi)&~eq; lower = np.isfinite(lo)&~eq
        con = np.vstack((C[eq], C[upper], -C[lower])); rhs = np.r_[hi[eq], hi[upper], -lo[lower]]
        cones = [clarabel.ZeroConeT(int(eq.sum())), clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))]
        settings = clarabel.DefaultSettings(); settings.verbose = False
        result = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(P)), q, sparse.csc_matrix(con), rhs, cones, settings).solve()
        if str(result.status) != 'Solved':
            raise RuntimeError('contact_walk_solver_'+str(result.status))
        solution = np.asarray(result.x)
        residual = float(max(0, np.max(lo-C@solution), np.max(C@solution-hi)))
        self.solver_residual = max(self.solver_residual, residual)
        if not np.isfinite(solution).all() or residual > 1e-4:
            raise RuntimeError('contact_walk_constraint_residual')
        self.torque = solution[n:n+nu].copy()
        self.qp_count += 1
        self.apply_actuation()

    def apply_actuation(self):
        if self.torque is None:
            return
        m, d = self.model, self.data
        kp = m.actuator_gainprm[:, 0]; kv = -m.actuator_biasprm[:, 2]
        control = d.qpos[self.qaddresses]+(self.torque+kv*d.qvel[self.dofs])/kp
        d.ctrl[:] = np.clip(control, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        d.ctrl[self.upper_ids] = self.upper_reference[self.upper_ids]
