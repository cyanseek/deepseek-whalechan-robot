"""VD2 contact-aware movement, balance and actuator-driven jump experiments.

Base character/art: 木子恶魔. Character adaptation and this controller:
OpenDafeiyu. VD1.1 dynamics and controller remain available separately.
"""
import numpy as np
import mujoco
from .bluefatfish import NativeController, R_FORWARD, build_scene as build_v1, DELIVERY


def build_scene(delivery=DELIVERY):
    from .bluefatfish_expressions import adapt_alpha_decals
    model,xml,assets,binding=build_v1(delivery)
    xml,assets,cutout=adapt_alpha_decals(model,xml,assets)
    model=mujoco.MjModel.from_xml_string(xml,assets=assets)
    binding.update(revision='VD2',base_model_art_author='木子恶魔',
        project_contribution='OpenDafeiyu: character posing/display adaptation, embodied dynamics/control/interaction, runtime morphs and alpha decal adaptation',
        expression_cutout=cutout)
    binding['runtime_visual_adaptation']='8 original skins unchanged; expression decal skin derived by alpha cutout; original input files retained byte-for-byte'
    binding['limitations']=[x for x in binding['limitations'] if 'Expression morphs' not in x]
    return model,xml,assets,binding


def yaw_matrix(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


class EmbodiedController(NativeController):
    def __init__(self, model, data, binding, balance=True):
        super().__init__(model,data,binding)
        self.balance_enabled=balance
        self.motion=np.zeros(3)
        self.heading=0.; self.foot_yaw=np.zeros(2)
        self.jump_requested=False; self.jump_done=False; self.jump_airborne=False
        self.last_time=0.; self.feedback=np.zeros(2)
        self.jump_peak=0.; self.emotion='neutral'
        # Selected from a retained, deterministic 24-case local exploration.
        self.jump_slew=12.;self.flight_crouch=.007;self.landing_offset=.002
        self.align_for_jump=False
        self.shift_seconds=.8;self.swing_seconds=.6;self.land_seconds=.15;self.center_seconds=.8
        self.step_lift=.005;self.step_scale=.10;self.max_forward_step=.012;self.motion_slew=2.
        self.crouch_depth=.007;self.extension_drop=.0015
        self.jump_forward_bias=.006
        self.recovery_seconds=.8

    def request_jump(self):
        if self.phase=='ready' and all(self.contacts()):
            self.jump_requested=True; self.jump_done=False
            return True
        return False

    def cancel(self):
        self.motion[:]=0; self.jump_requested=False;self.align_for_jump=False
        if self.phase=='crouch':
            self.phase='recover'; self.phase_t=float(self.data.time)
            self.recovery_start=self.root_target.copy()
        # Once launched, continue to land; cancellation never freezes a foot in air.

    def _tail_targets(self, time, greeting_weight):
        return {'tail_yaw':.05*greeting_weight}

    def _landing_confirmed(self, time):
        return any(self.contacts()) and time-self.phase_t>.02

    def _takeoff_confirmed(self, time):
        return not any(self.contacts())

    def _upper_targets(self,time,weight,command):
        flutter=.08*np.sin(2*np.pi*time/.7) if weight>.99 else 0.
        return dict(right_shoulder_roll=(-.6+flutter)*weight,right_shoulder_pitch=-.18*weight,
                    right_elbow=-.55*weight,head_yaw=np.clip(command[2]*.3,-.3,.3),head_pitch=.04*weight)

    def _solve(self, root_position, foot_positions, heading=0.):
        original_base=self.rest_base_rot; original_rot=self.rest_rot
        yaw=yaw_matrix(heading)
        self.rest_rot=[yaw.T@yaw_matrix(self.foot_yaw[i])@self.flat_rot[i] for i in range(2)]
        self.feedback[:]=0
        if self.balance_enabled and self.phase not in ('crouch','launch','flight','recover'):
            # Ideal simulated orientation + actual gyro, stated explicitly as an IMU
            # attitude input. No privileged target/environment location is consumed.
            actual=self.data.xmat[self.base].reshape(3,3)@R_FORWARD.T
            gravity=actual.T@np.array([0.,0.,1.])
            tilt=np.array([np.arctan2(gravity[1],gravity[2]), -np.arctan2(gravity[0],gravity[2])])
            gyro=R_FORWARD@self.data.sensor('gyro').data.copy()
            self.feedback=np.clip(1.5*tilt+.05*gyro[:2],-.4,.4)
        try: super()._solve(root_position,foot_positions,heading)
        finally: self.rest_base_rot=original_base;self.rest_rot=original_rot
        if self.balance_enabled and self.phase not in ('crouch','launch','flight','recover'):
            for ids in self.aids:
                self.target[ids[4]]+=self.feedback[1]
                self.target[ids[5]]+=self.feedback[0]

    def update(self, command, status='idle', walking=True):
        t=float(self.data.time);dt=max(.001,min(.03,t-self.last_time));self.last_time=t
        if not self.started:
            if t>=1.:
                self.flat_rot=[self.data.xmat[b].reshape(3,3).copy() for b in self.feet]
            super().update(command,status,walking=False)
            return
        active=walking and status=='approach' and (abs(command[0])>.01 or abs(command[1])>.01 or abs(command[2])>.03)
        age=t-self.phase_t
        if self.phase=='lower':
            self.root_target=self.rest_root+self.ease(age/1.2)*np.array([.006,0,-.009])+[0,0,-.0001]
            if age>=1.2:self.phase='ready';self.phase_t=t
        elif self.phase=='ready':
            if self.jump_requested:
                local=(yaw_matrix(self.heading).T@self.foot_targets.T).T
                if abs(local[0,0]-local[1,0])>.001:
                    self.align_for_jump=True;self.swing=int(np.argmin(local[:,0]))
                    self.phase='shift';self.phase_t=t;self.phase_root=self.root_target.copy()
                    self.events.append(dict(time=t,event='jump_square_feet'))
                    return
                self.align_for_jump=False
                self.jump_requested=False;self.phase='crouch';self.phase_t=t
                self.jump_origin=self.root_target.copy();self.phase_root=self.root_target.copy()
                offset=self.rest_root-self.rest_feet.mean(axis=0)+np.array([self.jump_forward_bias,0.,0.])
                self.jump_origin[:2]=self.foot_targets.mean(axis=0)[:2]+(yaw_matrix(self.heading)@offset)[:2]
                self.jump_airborne=False;self.jump_peak=0.
                self.events.append(dict(time=t,event='jump_crouch'))
            elif active:
                self.phase='shift';self.phase_t=t;self.phase_root=self.root_target.copy()
                self.events.append(dict(time=t,event='walking_started'))
        elif self.phase=='shift':
            stance=1-self.swing; center=self.foot_targets.mean(axis=0)
            desired=self.root_target.copy()
            desired[:2]=center[:2]+(yaw_matrix(self.heading)@np.array([.006,(.023 if stance==0 else -.023)-.0006,0]))[:2]
            self.root_target=self.phase_root+self.ease(age/self.shift_seconds)*(desired-self.phase_root)
            if age>=self.shift_seconds:
                if active or self.align_for_jump:
                    self.phase='swing';self.phase_t=t
                    self.swing_start=self.foot_targets[self.swing].copy()
                    # Bounded short steps; high-level command is a direction/speed request.
                    dx=float(np.clip(command[0]*self.step_scale,-.008,self.max_forward_step));dy=float(np.clip(command[1]*.08,-.006,.006))
                    self.step_yaw=float(np.clip(command[2]*.10,-.035,.035))
                    self.heading_start=self.heading;self.heading_end=self.heading+self.step_yaw
                    center=self.foot_targets.mean(axis=0)
                    offset=self.swing_start-center
                    self.swing_end=center+yaw_matrix(self.step_yaw)@offset+yaw_matrix(self.heading)@np.array([dx,dy,0])
                    if abs(self.step_yaw)>1e-5:
                        width=self.rest_feet[0,1]-self.rest_feet[1,1]
                        self.swing_end=self.foot_targets[stance]+yaw_matrix(self.heading_end)@np.array([dx,(width if self.swing==0 else -width)+dy,0.])
                    if self.align_for_jump:
                        self.step_yaw=0.;self.heading_end=self.heading
                        delta=self.foot_targets[stance]-self.swing_start
                        forward=yaw_matrix(self.heading)[:,0]
                        self.swing_end=self.swing_start+forward*float(delta@forward)
                    self.swing_end[2]=self.swing_start[2]
                    self.yaw_start=self.foot_yaw[self.swing]
                    self.events.append(dict(time=t,event='foot_lift',side=self.swing))
                else:self.phase='center';self.phase_t=t;self.phase_root=self.root_target.copy()
        elif self.phase=='swing':
            u=float(np.clip(age/self.swing_seconds,0,1));w=self.ease(u)
            self.foot_targets[self.swing]=self.swing_start+w*(self.swing_end-self.swing_start)
            self.foot_targets[self.swing,2]+=self.step_lift*np.sin(np.pi*u)**2
            self.heading=self.heading_start+w*self.step_yaw
            self.foot_yaw[self.swing]=self.yaw_start+w*(self.heading_end-self.yaw_start)
            if u>=1:self.phase='land';self.phase_t=t
        elif self.phase=='land':
            if age>=self.land_seconds and self.contacts()[self.swing]:
                self.events.append(dict(time=t,event='foot_landed',side=self.swing))
                self.steps_completed+=1;self.swing=1-self.swing
                self.phase='shift' if active else 'center';self.phase_t=t;self.phase_root=self.root_target.copy()
            elif age>.65:self.phase='failed';self.events.append(dict(time=t,event='landing_timeout'))
        elif self.phase=='center':
            desired=self.root_target.copy();desired[:2]=self.foot_targets.mean(axis=0)[:2]+(yaw_matrix(self.heading)@np.array([.006,-.0006,0]))[:2]
            self.root_target=self.phase_root+self.ease(age/self.center_seconds)*(desired-self.phase_root)
            if age>=self.center_seconds:self.phase='ready';self.phase_t=t
        elif self.phase=='crouch':
            crouch=self.jump_origin+np.array([0.,0.,-self.crouch_depth])
            self.root_target=self.phase_root+self.ease(age/.6)*(crouch-self.phase_root)
            self.emotion='relaxed'
            if age>=.8:
                self.phase='launch';self.phase_t=t
                self.launch_ctrl=self.data.ctrl.copy();self.launch_root=self.root_target.copy()
                extension=self.jump_origin.copy();extension[2]=self.rest_root[2]-self.extension_drop
                self._solve(extension,self.foot_targets,self.heading)
                self.launch_target=self.target.copy()
                tuck=self.jump_origin.copy();tuck[2]=self.rest_root[2]-self.flight_crouch
                self._solve(tuck,self.foot_targets,self.heading)
                self.flight_target=self.target.copy()
                self.target[:]=self.launch_ctrl
                self.events.append(dict(time=t,event='jump_launch'))
        elif self.phase=='launch':
            # Extend the legs against the ground. Dynamics, not root transforms,
            # determine whether this produces a jump.
            for ids in self.aids:self.target[ids]=self.launch_target[ids]
            if self._takeoff_confirmed(t):
                self.jump_airborne=True;self.phase='flight';self.phase_t=t
                self.flight_start=self.data.ctrl.copy()
                self.events.append(dict(time=t,event='jump_takeoff'))
            elif age>.35:
                self.phase='recover';self.phase_t=t;self.recovery_start=self.root_target.copy()
                self.events.append(dict(time=t,event='jump_no_takeoff'))
        elif self.phase=='flight':
            self.emotion='happy'
            for ids in self.aids:self.target[ids]=self.flight_target[ids]
            if self._landing_confirmed(t):
                self.phase='recover';self.phase_t=t
                self.foot_targets=np.array([self.data.xpos[b].copy() for b in self.feet])
                self.foot_targets[:,2]=self.rest_feet[:,2]
                self.root_target=self.data.qpos[:3].copy();self.recovery_start=self.root_target.copy()
                self.jump_origin=self.root_target.copy();self.jump_origin[2]=self.rest_root[2]-.0091
                self.jump_origin[:2]=self.foot_targets.mean(axis=0)[:2]+(yaw_matrix(self.heading)@np.array([self.landing_offset,-.0006,0.]))[:2]
                self.events.append(dict(time=t,event='jump_landing'))
            elif age>.7:self.phase='failed';self.events.append(dict(time=t,event='jump_landing_timeout'))
        elif self.phase=='recover':
            self.root_target=self.recovery_start+self.ease(age/self.recovery_seconds)*(self.jump_origin-self.recovery_start)
            level=float(self.data.xmat[self.base].reshape(3,3)[2,2])
            if age>1.2 and all(self.contacts()) and level>np.cos(.10) and np.linalg.norm(self.data.qvel[3:6])<.15:
                self.phase='ready';self.phase_t=t;self.jump_done=True;self.emotion='happy'
                self.events.append(dict(time=t,event='jump_recovered'))
            elif age>4.:
                self.phase='failed';self.events.append(dict(time=t,event='recovery_timeout'))
        if self.phase not in ('launch','flight'):self._solve(self.root_target,self.foot_targets,self.heading)
        if status=='arrived' and self.phase=='ready' and self.wave_start is None and not self.wave_done:
            self.wave_start=t;self.events.append(dict(time=t,event='wave_started'))
        weight=0.
        if self.wave_start is not None:
            u=t-self.wave_start
            if status!='arrived':self.wave_start=None;self.wave_done=True
            elif u<1:weight=self.ease(u)
            elif u<2.4:weight=1.
            elif u<3.5:weight=1-self.ease((u-2.4)/1.1)
            else:self.wave_start=None;self.wave_done=True;self.events.append(dict(time=t,event='wave_completed'))
        for name,value in {**self._upper_targets(t,weight,command),**self._tail_targets(t,weight)}.items():
            self.target[self.model.actuator(name).id]=value
        if weight>0:self.emotion='happy'
        slew=(self.jump_slew if self.phase in ('launch','flight') else self.motion_slew)*dt
        self.data.ctrl[:]+=np.clip(self.target-self.data.ctrl,-slew,slew)
