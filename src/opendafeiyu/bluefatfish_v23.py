"""White bloomer shorts and a bounded, quicker small-step character profile."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from .bluefatfish import DELIVERY,R_FORWARD,rot,quat,values
from .bluefatfish_v22 import build_scene as build_v22,CoordinatedController

DESIGN=Path(__file__).resolve().parents[2]/'assets/characters/bluefatfish-v23/clothing.json'


def build_scene(delivery=DELIVERY):
    _,xml,assets,b=build_v22(delivery)
    design=json.loads(DESIGN.read_text('utf8'))
    manifest=json.loads((Path(delivery)/'mujoco_preview/manifest.json').read_text('utf8'))
    bones={x['source_bone']:x for x in manifest['bones']}
    tree=ET.fromstring(xml);asset=tree.find('asset')
    ET.SubElement(asset,'material',name='cotton_white',rgba='.97 .98 1 1',emission='.65',specular='.05',shininess='.1')
    def cloth(bone,name,center,size,kind='ellipsoid'):
        bone_info=bones[bone];r=rot(bone_info['bindquat_wxyz']);p=np.array(bone_info['bindpos'])
        body=tree.find(f".//body[@name='{bone_info['body']}']")
        return ET.SubElement(body,'geom',name=name,type=kind,pos=values(r.T@(np.asarray(center)-p)),
            quat=values(quat(r.T)),size=values(size),material='cotton_white',group='2',
            contype='0',conaffinity='0',mass='0',rgba='1 1 1 1')
    cloth('hips','bloomer_waist',design['waist_center'],design['waist_size'])
    for sign,suffix in [(1,'.L'),(-1,'.R')]:
        center=np.array(design['leg_center']);center[0]*=sign
        cloth('upper_leg'+suffix,'bloomer_leg'+suffix,center,design['leg_size'])
        # A continuous folded fabric hem, rather than separate bead-like bumps.
        bone_info=bones['upper_leg'+suffix];r=rot(bone_info['bindquat_wxyz']);p=np.array(bone_info['bindpos'])
        vertices=[];faces=[];segments=64
        for rx,ry,z in [(.021,.023,-.014),(.027,.029,-.018),(.029,.031,-.021),(.024,.026,-.020)]:
            for theta in np.linspace(0,2*np.pi,segments,endpoint=False):
                ripple=.0012*np.cos(12*theta)
                point=center+[(rx+ripple)*np.cos(theta),(ry+ripple)*np.sin(theta),z+.6*ripple]
                vertices.append(r.T@(point-p))
        for j in range(4):
            for i in range(segments):
                a=j*segments+i;b0=j*segments+(i+1)%segments;c=((j+1)%4)*segments+(i+1)%segments;d=((j+1)%4)*segments+i
                faces.extend([[a,b0,c],[a,c,d]])
        mesh_name='bloomer_hem'+suffix
        ET.SubElement(asset,'mesh',name=mesh_name,vertex=values(vertices),face=' '.join(str(x) for f in faces for x in f))
        body=tree.find(f".//body[@name='{bone_info['body']}']")
        ET.SubElement(body,'geom',name=mesh_name,type='mesh',mesh=mesh_name,material='cotton_white',group='2',
                      contype='0',conaffinity='0',mass='0',rgba='1 1 1 1')
    for sign in [-1,1]:
        cloth('hips','bloomer_bow_'+str(sign),[sign*.004,-.032,.086],[.0045,.0022,.0025])
    cloth('hips','bloomer_bow_center',[0,-.033,.086],[.0018,.002,.002])
    xml=ET.tostring(tree,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets=assets)
    b.update(revision='VD2.3',clothing=dict(design_source=DESIGN.relative_to(DESIGN.parents[3]).as_posix(),
        description='Opaque white bloomer shorts, rounded scalloped cuffs and a small bow; pelvis/upper-leg attachments',
        dynamics='Visual cloth surface only, existing body mass/collision budget unchanged; no cloth solver'))
    return m,xml,assets,b


class SmallStepController(CoordinatedController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shift_seconds=.45;self.swing_seconds=.30;self.land_seconds=.08;self.center_seconds=.6
        self.step_scale=.167;self.max_forward_step=.020;self.step_lift=.009;self.motion_slew=3.
        self.prepare_stable_since=None
        self.jump_forward_bias=.012

    def request_jump(self):
        if self.jump_preparing or self.phase!='ready' or not all(self.contacts()):return False
        # Re-anchor planning targets to measured support after walking. Only the
        # scratch targets change; the actual free base remains physically integrated.
        self.foot_targets=np.array([self.data.xpos[f].copy() for f in self.feet])
        self.foot_targets[:,2]=self.rest_feet[:,2]
        self.root_target=self.data.qpos[:3].copy()
        orientation=self.data.xmat[self.base].reshape(3,3)@R_FORWARD.T
        self.heading=float(np.arctan2(orientation[1,0],orientation[0,0]))
        for i,f in enumerate(self.feet):
            relative=self.data.xmat[f].reshape(3,3)@self.flat_rot[i].T
            self.foot_yaw[i]=np.arctan2(relative[1,0],relative[0,0])
        self.prepare_stable_since=None
        self.events.append(dict(time=float(self.data.time),event='jump_support_targets_reanchored'))
        return super().request_jump()

    def _jump_preparation_ready(self,time):
        stable=all(self.contacts()) and self.data.xmat[self.base].reshape(3,3)[2,2]>np.cos(.06) and np.linalg.norm(self.data.qvel[3:6])<.12 and abs(self.data.qvel[2])<.015
        if not stable:self.prepare_stable_since=None
        elif self.prepare_stable_since is None:self.prepare_stable_since=time
        return super()._jump_preparation_ready(time) and self.prepare_stable_since is not None and time-self.prepare_stable_since>.2
