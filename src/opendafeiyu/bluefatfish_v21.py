"""VD2.1 articulated whale tail; original shape and total estimated mass retained.

Base character/art: 木子恶魔. Articulation/control: OpenDafeiyu.
The tail has root yaw/pitch and a mid-tail pitch. Flukes follow the distal tail;
there is no claim of soft-body water propulsion or independent fin actuation.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .bluefatfish import DELIVERY, R_FORWARD, rot, values, quat
from .bluefatfish_v2 import build_scene as build_v2, EmbodiedController


def build_scene(delivery=DELIVERY):
    previous,xml,assets,binding=build_v2(delivery)
    tree=ET.fromstring(xml)
    manifest=json.loads((delivery/'mujoco_preview/manifest.json').read_text('utf8'))
    bones={b['source_bone']:b for b in manifest['bones']}
    def element(name):return tree.find(f".//body[@name='{bones[name]['body']}']")
    root=element('WeiBa');mid=element('WeiBa.002')
    rr=rot(bones['WeiBa']['bindquat_wxyz']);rm=rot(bones['WeiBa.002']['bindquat_wxyz'])
    pr=np.array(bones['WeiBa']['bindpos']);pm=np.array(bones['WeiBa.002']['bindpos'])
    # Move the same distal mass/collision shapes into their articulated frame.
    # Transform exact capsule endpoints; do not remove contacts or add hidden mass.
    for name in ('tail_distal_collision','tail_flukes_collision'):
        geom=root.find(f"geom[@name='{name}']")
        endpoints=np.fromstring(geom.get('fromto'),sep=' ').reshape(2,3)
        endpoints=((endpoints@rr.T+pr-pm)@rm)
        geom.set('fromto',values(endpoints));root.remove(geom);mid.append(geom)
    for bone,name,limits in [('WeiBa','tail_pitch',[-.05,.32]),('WeiBa.002','tail_bend',[-.25,.30])]:
        axis=rot(bones[bone]['bindquat_wxyz']).T@R_FORWARD.T@np.array([0.,1.,0.])
        ET.SubElement(element(bone),'joint',name=name,axis=values(axis),range=values(limits))
        ET.SubElement(tree.find('actuator'),'position',name=name,joint=name,kp='.6',kv='.025',
                      ctrllimited='true',ctrlrange=values(limits),forcerange='-.08 .08')
    xml=ET.tostring(tree,encoding='unicode');model=mujoco.MjModel.from_xml_string(xml,assets=assets)
    binding.update(revision='VD2.1',nq=model.nq,nv=model.nv,nu=model.nu,
        joints=binding['joints']+['tail_pitch','tail_bend'],
        robot_mass_kg=float(model.body_subtreemass[model.body('base').id]),
        tail=dict(joints=['tail_yaw','tail_pitch','tail_bend'],estimated_mass_kg=.028,
            flukes='Follow distal tail; no independent fluke joints or soft-body dynamics',
            control='Smooth bounded prescribed gesture through actual position actuators'))
    binding['mass_estimates']=[dict(body=model.body(i).name,mass_kg=float(model.body_mass[i]),
        inertia_kg_m2=model.body_inertia[i].tolist()) for i in range(model.nbody)
        if model.body_mass[i]>0 and model.body_mocapid[i]<0]
    return model,xml,assets,binding


class ExpressiveController(EmbodiedController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tail_started=None
        self.tail_duration=6.

    def request_tail(self):
        if self.phase!='ready' or self.tail_started is not None:return False
        self.tail_started=float(self.data.time)
        self.events.append(dict(time=self.tail_started,event='tail_gesture_started'))
        return True

    def cancel(self):
        super().cancel()
        if self.tail_started is not None:
            self.events.append(dict(time=float(self.data.time),event='tail_gesture_cancelled'))
            self.tail_started=None

    def _tail_targets(self,time,greeting_weight):
        envelope=0.;phase=0.
        if self.tail_started is not None:
            age=time-self.tail_started
            if age>=self.tail_duration:
                self.events.append(dict(time=time,event='tail_gesture_completed'));self.tail_started=None
            else:
                envelope=self.ease(age/.8)*self.ease((self.tail_duration-age)/.8)
                phase=2*np.pi*.65*age
        return dict(tail_yaw=.05*greeting_weight+.19*envelope*np.sin(phase),
                    tail_pitch=envelope*(.14+.09*np.sin(phase)),
                    tail_bend=.10*envelope*np.sin(phase-.8))
