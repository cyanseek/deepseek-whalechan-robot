"""Phase-coordinated gait, tail, gestures and expressions for OpenDafeiyu.

Only actuator targets are written. The demonstration supplies explicit high-level
commands; physical contacts and completion events determine phase transitions.
"""
import numpy as np
from .bluefatfish_v21 import build_scene as build_v21, ExpressiveController
from .bluefatfish_v2 import EmbodiedController


def build_scene(delivery=None):
    m,xml,assets,b=build_v21() if delivery is None else build_v21(delivery)
    b.update(revision='VD2.2',coordination='Contact-phase gait arms/tail, jump preparation, stable landing recovery, smooth expression arbitration')
    return m,xml,assets,b


class CoordinatedController(ExpressiveController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.channels={};self.channel_velocity={};self.channel_time={}
        self.jump_preparing=False;self.prepare_time=0.;self.jump_finished_at=None
        self.gait_phase=0.;self.activity='settle';self.expression='neutral'
        self.motion_request=np.zeros(3);self.pre_jump_root=None

    def _landing_confirmed(self,time):
        # A brief toe contact during upward extension is not the landing phase.
        # Root vertical velocity is an ideal simulated proprioceptive input.
        return super()._landing_confirmed(time) and self.data.qvel[2]<=0.

    def smooth(self,name,target,rate,acceleration,frequency=10.):
        target=np.asarray(target,dtype=float);t=float(self.data.time)
        if name not in self.channels:
            self.channels[name]=np.zeros_like(target);self.channel_velocity[name]=np.zeros_like(target)
            self.channel_time[name]=t-.01
        dt=float(np.clip(t-self.channel_time[name],0,.03));self.channel_time[name]=t
        pos=self.channels[name];vel=self.channel_velocity[name]
        acc=np.clip(frequency**2*(target-pos)-2*frequency*vel,-acceleration,acceleration)
        vel[:]=np.clip(vel+acc*dt,-rate,rate);pos[:]+=vel*dt
        return pos.copy()

    def request_jump(self):
        if self.jump_preparing or self.phase!='ready' or not all(self.contacts()):return False
        self.jump_preparing=True;self.prepare_time=float(self.data.time);self.jump_done=False
        self.tail_started=None
        self.pre_jump_root=self.root_target.copy()
        self.events.append(dict(time=self.prepare_time,event='coordinated_jump_prepare'))
        return True

    def request_tail(self):
        if self.jump_preparing or self.phase in ('crouch','launch','flight','recover'):return False
        return super().request_tail()

    def cancel(self):
        aborted_crouch=self.phase=='crouch'
        self.jump_preparing=False;super().cancel()
        if aborted_crouch and self.pre_jump_root is not None:
            self.jump_origin=self.pre_jump_root.copy()
        self.events.append(dict(time=float(self.data.time),event='coordination_cancelled'))

    def _jump_preparation_ready(self,time):
        tail=self.channels.get('tail',np.zeros(3));velocity=self.channel_velocity.get('tail',np.zeros(3))
        return time-self.prepare_time>.8 and np.max(np.abs(tail-[0,.04,0]))<.008 and np.max(np.abs(velocity))<.06

    def update(self,command,status='idle',walking=True):
        t=float(self.data.time)
        self.motion_request=np.asarray(command[:3]).copy()
        if self.jump_preparing:
            command=np.zeros_like(command);status='idle';walking=False
            if self._jump_preparation_ready(t):
                if EmbodiedController.request_jump(self):
                    self.jump_preparing=False
                    self.events.append(dict(time=t,event='coordinated_jump_ready'))
        old_done=self.jump_done
        super().update(command,status,walking)
        if self.jump_done and not old_done:self.jump_finished_at=t
        if self.jump_preparing:self.activity='prepare';self.expression='focus'
        elif self.phase=='crouch':self.activity='crouch';self.expression='focus'
        elif self.phase in ('launch','flight') or (self.phase=='recover' and t-self.phase_t<.3):
            self.activity='jump';self.expression='effort'
        elif self.phase=='recover':self.activity='recover';self.expression='delight'
        elif self.wave_start is not None:self.activity='greeting';self.expression='happy'
        elif self.phase in ('shift','swing','land','center'):
            self.activity='turning' if abs(command[2])>.02 else 'walking';self.expression='neutral'
        elif self.tail_started is not None:self.activity='tail';self.expression='delight'
        elif self.jump_finished_at is not None and t-self.jump_finished_at<1.4:self.activity='relief';self.expression='delight'
        else:self.activity='idle' if self.started else 'settle';self.expression='neutral'

    def gait(self):
        age=float(self.data.time)-self.phase_t
        if self.phase in ('shift','swing','land') and not self.align_for_jump:
            if self.phase=='shift':u=.35*np.clip(age/self.shift_seconds,0,1)
            elif self.phase=='swing':u=.35+.55*np.clip(age/self.swing_seconds,0,1)
            else:u=.90+.10*np.clip(age/self.land_seconds,0,1)
            self.gait_phase=np.pi*(self.steps_completed+u)
            return self.gait_phase,1.
        return self.gait_phase,0.

    def _tail_targets(self,time,greeting_weight):
        phase,moving=self.gait();critical=self.jump_preparing or self.phase in ('crouch','launch','flight','recover')
        desired=np.array([.012*np.sin(2*np.pi*.24*time),.055,.010*np.sin(2*np.pi*.24*time-.7)])
        if moving:
            desired=np.array([-.065*np.sin(phase)-.08*self.motion_request[2],
                              .055+.025*np.cos(2*phase),.025*np.sin(phase-.7)])
        if greeting_weight:
            desired+=greeting_weight*np.array([.10*np.sin(2*np.pi*.6*time),.07,.045*np.sin(2*np.pi*.6*time-.8)])
        if self.tail_started is not None:
            age=time-self.tail_started
            if age>=self.tail_duration:
                self.tail_started=None;self.events.append(dict(time=time,event='tail_gesture_completed'))
            else:
                w=self.ease(age/.8)*self.ease((self.tail_duration-age)/.8);p=2*np.pi*.65*age
                desired=(1-w)*desired+w*np.array([.17*np.sin(p),.13+.07*np.sin(p),.08*np.sin(p-.8)])
        if critical:desired=np.array([0.,.04,0.])
        value=self.smooth('tail',desired,rate=.85,acceleration=4.5)
        return dict(zip(['tail_yaw','tail_pitch','tail_bend'],value))

    def _upper_targets(self,time,weight,command):
        phase,moving=self.gait();critical=self.jump_preparing or self.phase in ('crouch','launch','flight','recover')
        left=.08*moving*np.sin(phase);right=-left
        targets=np.array([left,right,.015,-.015,-.035*moving,-.035*moving,
                          np.clip(command[2]*.3,-.2,.2),.006*np.sin(2*np.pi*.25*time)])
        if critical:
            targets[:6]=[0.,0.,.015,-.015,-.045,-.045]
            targets[6:]=[0.,-.018 if self.phase in ('crouch','launch','flight') else .01]
        targets[1]=(1-weight)*targets[1]-.18*weight
        targets[3]=(1-weight)*targets[3]+weight*(-.6+.05*np.sin(2*np.pi*time/.7))
        targets[5]=(1-weight)*targets[5]-.55*weight
        targets[7]+=.025*weight
        values=self.smooth('upper',targets,rate=1.5,acceleration=6.)
        names=['left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll',
               'left_elbow','right_elbow','head_yaw','head_pitch']
        return dict(zip(names,values))


class CoordinatedDemo:
    """Explicit scripted task input with contact/completion-gated transitions."""
    def __init__(self,walk_steps=6,turn_steps=4):
        self.stage='settle';self.events=[];self.start=0.;self.step_start=0;self.cancelled=False
        self.walk_steps=walk_steps;self.turn_steps=turn_steps
    def enter(self,stage,low):
        self.stage=stage;self.start=float(low.data.time);self.step_start=low.steps_completed
        self.events.append(dict(time=self.start,event='demo_stage',stage=stage))
        if stage in ('hello','celebrate'):
            low.wave_start=None;low.wave_done=False
        if stage=='celebrate':low.request_tail()
    def cancel(self,low):
        low.cancel();self.cancelled=True;self.enter('cancelled',low)
    def update(self,low):
        t=float(low.data.time);command=np.zeros(7);status='idle'
        if self.stage=='settle' and low.phase=='ready':self.enter('hello',low)
        if self.stage=='hello':
            status='arrived'
            if low.wave_done:self.enter('walk',low)
        if self.stage=='walk':
            status='approach';command[0]=.12
            if low.steps_completed-self.step_start>=self.walk_steps:self.enter('stop',low);command[:]=0;status='idle'
        if self.stage=='stop' and low.phase=='ready':
            if low.request_jump():self.enter('jump',low)
        if self.stage=='jump' and low.jump_done and low.phase=='ready':self.enter('landed',low)
        if self.stage=='landed' and t-self.start>1.:self.enter('turn',low)
        if self.stage=='turn':
            status='approach';command[:3]=[.015,0,.35]
            if low.steps_completed-self.step_start>=self.turn_steps:self.enter('stop_turn',low);command[:]=0;status='idle'
        if self.stage=='stop_turn' and low.phase=='ready':self.enter('celebrate',low)
        if self.stage=='celebrate':
            status='arrived'
            if low.wave_done and low.tail_started is None:self.enter('rest',low);status='idle'
        if self.stage=='rest' and t-self.start>1.5:self.enter('complete',low)
        if self.stage in ('cancelled','complete'):command[:]=0;status='stopped' if self.cancelled else 'idle'
        return command,status
