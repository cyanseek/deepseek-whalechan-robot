"""VD2.4 fitted clothing, an open-eyed smile, and revised movement control."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .bluefatfish import DELIVERY,values,R_FORWARD
from .bluefatfish_v22 import build_scene as build_v22
from .bluefatfish_v23 import SmallStepController
from .bluefatfish_v2 import yaw_matrix
from .bluefatfish_expressions import Expressions

DESIGN=Path(__file__).resolve().parents[2]/'assets/characters/bluefatfish-v24/clothing.json'

def build_scene(delivery=DELIVERY):
    previous,xml,assets,b=build_v22(delivery)
    design=json.loads(DESIGN.read_text('utf8'))
    manifest=json.loads((Path(delivery)/'mujoco_preview/manifest.json').read_text('utf8'))
    bones={x['source_bone']:x for x in manifest['bones']}
    tree=ET.fromstring(xml);asset=tree.find('asset');deform=tree.find('deformable')
    ET.SubElement(asset,'material',name='cotton_white_v24',rgba='.97 .98 1 1',emission='.65',specular='.05',shininess='.1')
    verts=[];faces=[];weights=[]
    def surface(center,size,side=None,hem=False):
        start=len(verts);n=48;rings=18
        for j in range(rings+1):
            phi=np.pi*j/rings
            for i in range(n):
                theta=2*np.pi*i/n
                p=np.array(center)+np.array(size)*[np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)]
                if hem:p+=np.array([.0006*np.cos(12*theta)*np.cos(theta),.0006*np.cos(12*theta)*np.sin(theta),.0004*np.cos(12*theta)])
                verts.append(p)
                # Smooth hips-to-thigh blend, shared by the leg and its fabric hem.
                leg=0. if side is None else float(np.clip((.083-p[2])/.045,0,.38))
                weights.append([1-leg,leg if side==0 else 0,leg if side==1 else 0])
        for j in range(rings):
            for i in range(n):
                a=start+j*n+i;b0=start+j*n+(i+1)%n;c=b0+n;d=a+n
                faces.extend([[a,d,c],[a,c,b0]])
    surface(design['waist_center'],design['waist_size'])
    for i,sign in enumerate([1,-1]):
        center=np.array(design['leg_center']);center[0]*=sign
        surface(center,design['leg_size'],i)
        surface(center+[0,0,-.016],design['hem_size'],i,True)
    # Small opaque bow, enclosed inside the skirt envelope.
    for sign in [1,-1]:surface([sign*.0034,-.025,.084],[.0038,.0018,.0022])
    surface([0,-.026,.084],[.0015,.0015,.0017])
    verts=np.asarray(verts);weights=np.asarray(weights)
    skin=ET.SubElement(deform,'skin',name='fitted_bloomers',vertex=values(verts),face=' '.join(map(str,np.asarray(faces).ravel())),
                       material='cotton_white_v24',group='2',rgba='1 1 1 1')
    for i,name in enumerate(['hips','upper_leg.L','upper_leg.R']):
        ids=np.flatnonzero(weights[:,i]>0);bone=bones[name]
        ET.SubElement(skin,'bone',body=bone['body'],bindpos=values(bone['bindpos']),bindquat=values(bone['bindquat_wxyz']),
                      vertid=' '.join(map(str,ids)),vertweight=values(weights[ids,i]))
    # The source thighs are wider than their coarse collision capsules. Derive
    # a close-fitting opaque fabric surface from the actual skin, retaining its
    # exact bone weights, so a rigid primitive cannot sink inside a bent thigh.
    si=previous.skin('skin_0_0').id
    source=previous.skin_vert[previous.skin_vertadr[si]:previous.skin_vertadr[si]+previous.skin_vertnum[si]].copy()
    original_faces=previous.skin_face[previous.skin_faceadr[si]:previous.skin_faceadr[si]+previous.skin_facenum[si]//2].copy()
    centers=source[original_faces].mean(axis=1)
    selected=original_faces[(centers[:,2]>.045)&(centers[:,2]<.096)&(abs(centers[:,0])<.070)&(abs(centers[:,1])<.044)]
    normals=np.zeros_like(source)
    tri=source[original_faces];face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for k in range(3):np.add.at(normals,original_faces[:,k],face_normals)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    used,inverse=np.unique(selected,return_inverse=True)
    fitted=source[used]+.0006*normals[used]
    skin=ET.SubElement(deform,'skin',name='fitted_bloomer_surface',vertex=values(fitted),
        face=' '.join(map(str,inverse.ravel())),material='cotton_white_v24',group='2',rgba='1 1 1 1')
    for bi in range(previous.skin_boneadr[si],previous.skin_boneadr[si]+previous.skin_bonenum[si]):
        w=np.zeros(len(source));adr=previous.skin_bonevertadr[bi];n=previous.skin_bonevertnum[bi]
        w[previous.skin_bonevertid[adr:adr+n]]=previous.skin_bonevertweight[adr:adr+n]
        selected_weights=w[used];ids=np.flatnonzero(selected_weights>0)
        if len(ids):
            ET.SubElement(skin,'bone',body=previous.body(previous.skin_bonebodyid[bi]).name,
                bindpos=values(previous.skin_bonebindpos[bi]),bindquat=values(previous.skin_bonebindquat[bi]),
                vertid=' '.join(map(str,ids)),vertweight=values(selected_weights[ids]))
    xml=ET.tostring(tree,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets=assets)
    b.update(revision='VD2.4',clothing=dict(design_source=DESIGN.relative_to(DESIGN.parents[3]).as_posix(),
        description='Compact white bloomers, plus an actual-thigh-derived fabric shell at 0.6 mm offset with identical source skin weights; small bow and hems',
        dynamics='Visual skin only; existing mass/inertia/collision/force limits retained; no cloth solver'),
        joint_range_revision='Original joint ranges and actuator force limits retained')
    return m,xml,assets,b

class CheerfulExpressions(Expressions):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        # Derive only the lower-face part of the original happy morph. This keeps
        # the original eye shape open instead of forcing a permanent laughing face.
        z=self.neutral[:,2];mask=np.clip((.214-z)/.008,0,1)
        mask*=np.clip((.070-np.abs(self.neutral[:,0]))/.025,0,1)
        self.deltas['smile']=self.deltas['happy']*mask[:,None]
        self.semantic_targets={'neutral':{'smile':.70,'happy':.10},'focus':{'smile':.45,'happy':.06,'blink':.12}}
        self.report['default_expression']='Open-eyed smile: derived original happy mouth + 10% original happy face, with independent natural blink'
    def update(self,time,emotion='neutral',talking=False):
        super().update(time,emotion,talking)
        if emotion=='neutral' and self.weights.get('blink',0)>0:
            w=self.weights.copy();w['happy']=w.get('happy',0.)*(1-w['blink']);self.apply(w)

class LivelyController(SmallStepController):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.shift_seconds=.28;self.swing_seconds=.23;self.land_seconds=.06
        self.step_scale=.026/.12;self.max_forward_step=.026;self.step_lift=.008;self.motion_slew=4.
        self.lateral_scale=.5
        self.minimum_launch=.04
        self.crouch_depth=.007;self.jump_slew=24.;self.flight_crouch=.009
        self.landing_offset=.006;self.jump_forward_bias=.008;self.recovery_seconds=.25
        self.launch_hip_bias=-.30;self.tuck_height=.024;self.release_seconds=.075
        self.precision_requested=False;self.motion_mode='lively'
        self.walk_ranges=self.planning_ranges.copy()
        for side in ['left','right']:
            for name,limits in [('hip_pitch',[-1.,1.]),('knee',[0.,1.8]),('ankle_pitch',[-1.3,1.])]:
                self.walk_ranges[self.model.joint(side+'_'+name).id]=limits
    def _takeoff_confirmed(self,time):
        return time-self.phase_t>=self.minimum_launch and not any(self.contacts())

    def set_precision_motion(self,enabled):
        """RGB alignment keeps the validated slower stance and turn settings.

        Change only at supported/initial phases; never change an airborne step.
        """
        self.precision_requested=bool(enabled)
        desired='precision' if enabled else 'lively'
        if desired==self.motion_mode or self.phase not in ('settle','lower','ready'):return
        self.motion_mode=desired
        self.shift_seconds,self.swing_seconds,self.land_seconds,self.max_forward_step,self.step_lift,self.motion_slew,self.lateral_scale=(
            (.45,.30,.08,.020,.009,3.,1.) if enabled else (.28,.23,.06,.026,.008,4.,.5))
        self.step_scale=self.max_forward_step/.12
        self.events.append(dict(time=float(self.data.time),event='motion_mode',mode=desired))
    def _solve(self,root,feet,heading=0):
        self.planning_ranges[:]=self.model.jnt_range if self.phase in ('crouch','launch','flight','recover') else self.walk_ranges
        root=root.copy()
        if self.phase in ('shift','swing','land','center') and not self.align_for_jump:
            center=feet.mean(axis=0);lateral=yaw_matrix(heading)[:,1]
            root+=(self.lateral_scale-1)*float((root-center)@lateral)*lateral
        old=self.phase
        if old=='recover':self.phase='ready'
        try:super()._solve(root,feet,heading)
        finally:self.phase=old

    def update(self,*args,**kwargs):
        self.set_precision_motion(self.precision_requested)
        before=self.data.ctrl.copy();old=self.phase
        super().update(*args,**kwargs)
        if self.phase=='launch' and old!='launch':
            for ids in self.aids:self.launch_target[ids[2]]+=self.launch_hip_bias
        if self.phase=='flight':
            # Model-based proprioception places the landing feet under the
            # measured COM. A brief tuck transitions to an extended landing.
            # Only the scratch IK workspace receives a base pose assignment.
            yaw=yaw_matrix(self.heading);root=self.data.qpos[:3].copy()
            center=self.data.subtree_com[self.base].copy()
            feet=self.rest_feet.copy()
            feet[:,:2]=(yaw@(self.rest_feet-self.rest_feet.mean(axis=0)).T).T[:,:2]+center[:2]
            feet[:,2]=self.rest_feet[:,2]+self.tuck_height*max(0,1-(self.data.time-self.phase_t)/self.release_seconds)
            original=self.rest_base_rot
            self.rest_base_rot=yaw.T@self.data.xmat[self.base].reshape(3,3)
            try:self._solve(root,feet,self.heading)
            finally:self.rest_base_rot=original
            for ids in self.aids:self.data.ctrl[ids]=before[ids]+np.clip(self.target[ids]-before[ids],-.4,.4)
