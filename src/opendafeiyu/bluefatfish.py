"""Native, free-base dynamics for the locally supplied BlueFatFish visual rig.

Original model/art: 木子恶魔. Assets retain their restrictive VRM metadata;
this module does not confer redistribution permission. The supplied skins are
bound directly to articulated MuJoCo bodies, never to mocap replay bodies.
Masses, inertias and actuator limits below are simulation estimates, not CAD.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DELIVERY = ROOT / 'assets/local/BlueFatFish_Visual_Delivery'
DELIVERY = LOCAL_DELIVERY
R_FORWARD = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def values(x):
    return ' '.join(f'{v:.12g}' for v in np.asarray(x).ravel())


def rot(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def quat(matrix):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix).ravel())
    return result


def build_scene(delivery=DELIVERY):
    """Build new dynamics without overwriting any delivered asset.

    146 supplied bone bind frames and all 9 binary skins remain byte-identical.
    Their parent hierarchy now belongs to a free dynamic base. Only the two
    external visual targets are mocap bodies. Positive masses come from coarse
    coarse collision shapes; neither a floor weld nor an external support exists.
    """
    delivery = Path(delivery)
    folder = delivery / 'mujoco_preview'
    manifest = json.loads((folder / 'manifest.json').read_text('utf8'))
    # Fail on silently changed dependencies, while preserving the author's files.
    receipt=json.loads((delivery/'DELIVERY_MANIFEST.json').read_text('utf8'))
    entries=receipt.get('files',receipt) if isinstance(receipt,dict) else receipt
    if isinstance(entries,dict):entries=[dict(path=k,**v) for k,v in entries.items()]
    for entry in entries:
        name=entry.get('path',entry.get('file',''))
        if name.startswith('mujoco_preview/skins/') or name.startswith('mujoco_preview/textures/') or name in ('mujoco_preview/manifest.json','mujoco_preview/visual_mocap.xml'):
            if hashlib.sha256((delivery/name).read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Delivered dependency hash mismatch: '+name)
    if manifest['source_glb_sha256'] != hashlib.sha256((delivery / 'BlueFatFish_Visual.glb').read_bytes()).hexdigest():
        raise ValueError('Delivered GLB hash differs from skin manifest')
    original = ET.fromstring((folder / 'visual_mocap.xml').read_text('utf8'))
    scene = ET.Element('mujoco', model='BlueFatFish_VD1_native_dynamics_estimated')
    ET.SubElement(scene, 'compiler', angle='radian', fusestatic='false', inertiafromgeom='true')
    ET.SubElement(scene, 'option', timestep='.001', gravity='0 0 -9.81', integrator='implicitfast',
                  cone='elliptic', iterations='80', tolerance='1e-9')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    ET.SubElement(visual, 'quality', offsamples='4')
    ET.SubElement(visual, 'headlight', ambient='.10 .10 .10', diffuse='.15 .15 .15', specular='0 0 0')
    ET.SubElement(visual, 'map', znear='.003', zfar='15')
    ET.SubElement(scene, 'statistic', center='0 0 .2', extent='1')
    default = ET.SubElement(scene, 'default')
    ET.SubElement(default, 'joint', damping='.025', armature='.000015', limited='true')
    ET.SubElement(default, 'geom', contype='1', conaffinity='1', condim='4', friction='.85 .01 .001',
                  solref='.008 1', solimp='.95 .99 .001', group='3', rgba='.2 .7 .95 .4')
    asset = copy.deepcopy(original.find('asset'))
    for material in asset.findall('material'):
        material.set('emission','.65')
    scene.append(asset)
    assets = {}
    for node in asset.iter():
        if 'file' in node.attrib:
            filename = node.get('file')
            path = (folder / filename).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Asset path outside received preview')
            assets[filename] = path.read_bytes()
    # Skin elements may be under deformable rather than asset in MuJoCo 3.3.
    if original.find('deformable') is not None:
        deformable = copy.deepcopy(original.find('deformable'))
        scene.append(deformable)
        for node in deformable:
            node.set('group', '2')
            if node.get('file'):
                assets[node.get('file')] = (folder / node.get('file')).read_bytes()
    for skin in asset.findall('skin'):
        skin.set('group', '2')
    ET.SubElement(asset, 'texture', name='floor_texture', type='2d', builtin='checker',
                  rgb1='.80 .84 .89', rgb2='.87 .90 .94', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor_material', texture='floor_texture',
                  texrepeat='12 12', texuniform='true', reflectance='0',emission='.45')
    world = ET.SubElement(scene, 'worldbody')
    floor = ET.SubElement(world, 'body', name='floor')
    ET.SubElement(floor, 'geom', name='floor', type='plane', size='3 3 .05', group='0',
                  material='floor_material', rgba='1 1 1 1')
    ET.SubElement(world, 'light', pos='1 -1 2', dir='-.3 .3 -1', diffuse='.10 .10 .10')
    root = ET.SubElement(world, 'body', name='base', pos='0 0 .002', quat=values(quat(R_FORWARD)))
    ET.SubElement(root, 'freejoint', name='floating_base')
    bones = {b['source_bone']: b for b in manifest['bones']}
    positions = {n: np.array(b['bindpos']) for n, b in bones.items()}
    rotations = {n: rot(b['bindquat_wxyz']) for n, b in bones.items()}
    elements = {}
    for name, b in bones.items():
        parent = b['parent_source_bone']
        pr, pp = (rotations[parent], positions[parent]) if parent else (np.eye(3), np.zeros(3))
        elements[name] = ET.SubElement(elements[parent] if parent else root, 'body', name=b['body'],
            pos=values(pr.T @ (positions[name] - pp)), quat=values(quat(pr.T @ rotations[name])))
    actuator = ET.SubElement(scene, 'actuator')
    joints = []

    def joint(bone, name, world_axis, limits, kp=8., kv=.12, force=.6):
        # Axes are expressed in the new forward-X frame, then in source bone coordinates.
        axis = rotations[bone].T @ R_FORWARD.T @ np.asarray(world_axis)
        ET.SubElement(elements[bone], 'joint', name=name, axis=values(axis), range=values(limits))
        ET.SubElement(actuator, 'position', name=name, joint=name, kp=str(kp), kv=str(kv),
                      ctrllimited='true', ctrlrange=values(limits), forcerange=values([-force, force]))
        joints.append(name)

    def collision(bone, name, center, size, mass, kind='ellipsoid', end=None):
        r, p = rotations[bone], positions[bone]
        attr = dict(name=name, type=kind, mass=str(mass), size=values(size))
        if end is None:
            attr.update(pos=values(r.T @ (np.asarray(center) - p)), quat=values(quat(r.T @ R_FORWARD.T)))
        else:
            attr['fromto'] = values([r.T @ (np.asarray(center) - p), r.T @ (np.asarray(end) - p)])
        return ET.SubElement(elements[bone], 'geom', **attr)

    collision('hips', 'pelvis_collision', [0, 0, .085], [.021, .025, .017], .20)
    collision('chest', 'torso_collision', [0, 0, .125], [.028, .037, .041], .20)
    joint('head', 'head_yaw', [0, 0, 1], [-.45, .45], 1.5, .05, .15)
    joint('head', 'head_pitch', [0, 1, 0], [-.2, .2], 1.5, .05, .15)
    collision('head', 'head_collision', [0, .003, .278], [.077, .091, .087], .16)
    # Hair and skirt are coarse rigid envelopes, not soft-body cloth simulation.
    for s, side in [(1, 'left'), (-1, 'right')]:
        suffix = '.L' if s == 1 else '.R'
        upper, lower, foot = ('upper_leg'+suffix, 'lower_leg'+suffix, 'foot'+suffix)
        joint(upper, side+'_hip_yaw', [0, 0, 1], [-.5, .5])
        joint(upper, side+'_hip_roll', [1, 0, 0], [-.8, .8])
        joint(upper, side+'_hip_pitch', [0, 1, 0], [-1.0, 1.0])
        joint(lower, side+'_knee', [0, 1, 0], [0., 1.8])
        joint(foot, side+'_ankle_pitch', [0, 1, 0], [-1.3, 1.0])
        joint(foot, side+'_ankle_roll', [1, 0, 0], [-.8, .8])
        collision(upper, side+'_thigh_collision', positions[upper], [.010], .045, 'capsule', positions[lower])
        collision(lower, side+'_shin_collision', positions[lower], [.010], .035, 'capsule', positions[foot])
        collision(foot, side+'_sole', [s*.034, -.003, .009], [.022, .0165, .009], .065, 'box')
        arm, fore = 'upper_arm'+suffix, 'lower_arm'+suffix
        joint(arm, side+'_shoulder_pitch', [0, 1, 0], [-.8, .8], 1., .04, .12)
        joint(arm, side+'_shoulder_roll', [1, 0, 0], [-1.3, 1.3], 1., .04, .12)
        joint(fore, side+'_elbow', [0, 1, 0], [-1.4, .2], 1., .04, .12)
        collision(arm, side+'_upper_arm_collision', positions[arm], [.010], .018, 'capsule', positions[fore])
        collision(fore, side+'_forearm_collision', positions[fore], [.009], .014, 'capsule', positions['hand'+suffix])
        collision('head', side+'_hair_collision', [s*.095, .062, .18], [.006], .009, 'capsule', [s*.10, .055, .075])
    for i,angle in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):
        top=[.050*np.cos(angle),.035*np.sin(angle),.116]
        bottom=[.102*np.cos(angle),.075*np.sin(angle),.043]
        collision('spine',f'skirt_collision_{i}',top,[.0015],.001,'capsule',bottom)
    joint('WeiBa', 'tail_yaw', [0, 0, 1], [-.25, .25], .6, .025, .08)
    collision('WeiBa', 'tail_proximal_collision', positions['WeiBa'], [.008], .010, 'capsule', positions['WeiBa.002'])
    collision('WeiBa', 'tail_distal_collision', positions['WeiBa.002'], [.008], .010, 'capsule', positions['WeiBa.004'])
    collision('WeiBa', 'tail_flukes_collision', positions['WeiBa.L.002'], [.007], .008, 'capsule', positions['WeiBa.R.002'])
    # Sensor is at the face, looking forward; its RGB is the task's only target input.
    head_r, head_p = rotations['head'], positions['head']
    cam_source = np.array([0., -.096, .248])
    axes_forward = np.array([[0, -1, 0], [.10, 0, .994987437]])
    ET.SubElement(elements['head'], 'camera', name='robot_camera', fovy='60',
                  pos=values(head_r.T @ (cam_source-head_p)),
                  xyaxes=values((head_r.T @ R_FORWARD.T @ axes_forward.T).T))
    ET.SubElement(root, 'site', name='imu', size='.001', rgba='0 0 0 0')
    sensor = ET.SubElement(scene, 'sensor')
    ET.SubElement(sensor, 'gyro', name='gyro', site='imu')
    ET.SubElement(sensor, 'accelerometer', name='accelerometer', site='imu')
    for name, y, color in [('red', .06, '1 .025 .025 1'), ('green', -.12, '.02 1 .1 1')]:
        ET.SubElement(asset, 'material', name='marker_'+name, rgba=color, emission='.5')
        target = ET.SubElement(world, 'body', name='target_'+name, mocap='true', pos=f'.48 {y} .24')
        ET.SubElement(target, 'geom', name='target_'+name, type='sphere', size='.045', group='0',
                      material='marker_'+name, rgba=color, mass='.05')
    xml = ET.tostring(scene, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    report = dict(schema='opendafeiyu-bluefatfish-dynamics/1', revision='VD1.1', joints=joints,
        source_bones={name: b['body'] for name, b in bones.items()},
        source_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        source_glb_sha256=manifest['source_glb_sha256'], author='木子恶魔',
        display_height_m=manifest['display_height_m'], nskin=model.nskin, nq=model.nq,
        nv=model.nv, nu=model.nu, nmocap=model.nmocap, robot_mass_kg=float(model.body_subtreemass[model.body('base').id]),
        physics='new estimated rigid-body dynamics fitted to delivered bone positions; no Open Duck policy reuse',
        limitations=['Estimated masses/inertias/actuator limits, not mechanical validation.',
                     'Coarse collision envelopes; no articulated hair/cloth/springbone dynamics.',
                     'Expression morphs are unchanged at their delivered values.'],
        source_skin_assets_unchanged=True, character_mocap_bodies=0,
        mass_estimates=[dict(body=model.body(i).name,mass_kg=float(model.body_mass[i]),
                        inertia_kg_m2=model.body_inertia[i].tolist())
                       for i in range(model.nbody) if model.body_mass[i]>0 and model.body_mocapid[i]<0],
        collision_geoms=[model.geom(i).name for i in range(model.ngeom) if model.geom_contype[i]])
    return model, xml, assets, report


class NativeController:
    """Slow model-based IK/PD controller; only writes actuator targets.

    The scratch MjData is a kinematic planning workspace, not the simulated
    body. Base coordinates in the live MjData are never assigned. This is an
    explicitly prescribed gait with measured foot-contact feedback, not a
    learned locomotion policy. High-level navigation still consumes RGB only.
    """
    def __init__(self, model, data, binding):
        self.model, self.data, self.binding = model, data, binding
        self.ik = mujoco.MjData(model)
        self.ik.qpos[:] = data.qpos
        self.base = model.body('base').id
        self.feet = [model.body(binding['source_bones']['foot'+s]).id for s in ('.L', '.R')]
        self.hips = [model.body(binding['source_bones']['upper_leg'+s]).id for s in ('.L', '.R')]
        self.leg_names = [[side+'_'+name for name in ('hip_yaw','hip_roll','hip_pitch','knee','ankle_pitch','ankle_roll')]
                          for side in ('left','right')]
        self.qids = [[int(model.jnt_qposadr[model.joint(n).id]) for n in names] for names in self.leg_names]
        self.vids = [[int(model.jnt_dofadr[model.joint(n).id]) for n in names] for names in self.leg_names]
        self.aids = [[model.actuator(n).id for n in names] for names in self.leg_names]
        self.started = False
        self.phase = 'settle'
        self.phase_t = 0.
        self.swing = 1
        self.steps_completed = 0
        self.events = []
        self.wave_start = None
        self.wave_done = False
        self.max_ik_error = 0.
        self.target = data.ctrl.copy()
        self.phase_root = np.zeros(3)
        self.planning_ranges = model.jnt_range.copy()

    @staticmethod
    def ease(t):
        x = float(np.clip(t, 0, 1))
        return x*x*x*(10+x*(-15+6*x))

    def contacts(self):
        foot_geoms = [self.model.geom(side+'_sole').id for side in ('left','right')]
        floor = self.model.geom('floor').id
        return [any({int(c.geom1),int(c.geom2)}=={floor,g} for c in self.data.contact) for g in foot_geoms]

    def _solve(self, root_position, foot_positions, heading=0.):
        m, d = self.model, self.ik
        d.qpos[:3] = root_position
        c,s=np.cos(heading),np.sin(heading)
        yaw=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        d.qpos[3:7] = quat(yaw@self.rest_base_rot)
        for _ in range(14):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            maximum = 0.
            for side in range(2):
                body = self.feet[side]
                current = d.xmat[body].reshape(3,3)
                desired = yaw@self.rest_rot[side]
                # World-space small-angle residual, weighted to metres.
                angle = .5*sum((np.cross(current[:,k],desired[:,k]) for k in range(3)),start=np.zeros(3))
                err = np.r_[foot_positions[side]-d.xpos[body], .04*angle]
                maximum = max(maximum,float(np.linalg.norm(err[:3])))
                jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jacBody(m,d,jp,jr,body)
                jac=np.vstack([jp[:,self.vids[side]],.04*jr[:,self.vids[side]]])
                delta=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-7,err)
                d.qpos[self.qids[side]]+=np.clip(delta,-.12,.12)
                for n,qid in zip(self.leg_names[side],self.qids[side]):
                    d.qpos[qid]=np.clip(d.qpos[qid],*self.planning_ranges[m.joint(n).id])
            if maximum<2e-5:break
        self.max_ik_error=max(self.max_ik_error,maximum)
        for side in range(2):self.target[self.aids[side]]=d.qpos[self.qids[side]]

    def update(self, command, status='idle', walking=True):
        t=float(self.data.time)
        if t<1.0:
            self.data.ctrl[:]=0
            return
        if not self.started:
            self.started=True
            self.rest_root=self.data.qpos[:3].copy()
            self.rest_feet=np.array([self.data.xpos[b].copy() for b in self.feet])
            self.rest_rot=[self.data.xmat[b].reshape(3,3).copy() for b in self.feet]
            self.rest_base_rot=self.data.xmat[self.base].reshape(3,3).copy()
            self.foot_targets=self.rest_feet.copy()
            self.root_target=self.rest_root.copy()
            self.phase_root=self.root_target.copy()
            self.ik.qpos[:]=self.data.qpos
            for ids in self.qids:
                self.ik.qpos[ids[2]]=-.10;self.ik.qpos[ids[3]]=.25;self.ik.qpos[ids[4]]=-.15
            self.phase='lower';self.phase_t=t
            self.heading=0.
        active=walking and status=='approach' and command[0]>.01
        age=t-self.phase_t
        if self.phase=='lower':
            u=self.ease(age/1.2)
            self.root_target=self.rest_root+u*np.array([.006,0.,-.009])+[0.,0.,-.0001]
            if age>=1.2:
                self.phase='ready';self.phase_t=t
        elif self.phase=='ready':
            if active:
                self.phase='shift';self.phase_t=t;self.phase_root=self.root_target.copy()
                self.events.append({'time':t,'event':'walking_started'})
        elif self.phase=='shift':
            stance=1-self.swing
            desired=self.root_target.copy()
            center=np.mean(self.foot_targets[:,:2],axis=0)
            desired[:2]=center+[.006,(.023 if stance==0 else -.023)-.0006]
            self.root_target=self.phase_root+self.ease(age/1.0)*(desired-self.phase_root)
            if age>=1.0:
                if active:
                    self.phase='swing';self.phase_t=t
                    self.swing_start=self.foot_targets[self.swing].copy()
                    self.swing_end=self.swing_start+np.array([.008,0.,0.])
                    self.events.append({'time':t,'event':'foot_lift','side':self.swing})
                else:
                    self.phase='center';self.phase_t=t;self.phase_root=self.root_target.copy()
        elif self.phase=='swing':
            u=float(np.clip(age/.65,0,1))
            self.foot_targets[self.swing]=self.swing_start+self.ease(u)*(self.swing_end-self.swing_start)
            self.foot_targets[self.swing,2]+= .005*np.sin(np.pi*u)**2
            if u>=1:
                self.phase='land';self.phase_t=t
        elif self.phase=='land':
            # Finish the in-flight foot before cancelling; do not freeze a foot in air.
            if age>=.15 and self.contacts()[self.swing]:
                self.events.append({'time':t,'event':'foot_landed','side':self.swing})
                self.steps_completed+=1;self.swing=1-self.swing
                self.phase='shift' if active else 'center';self.phase_t=t;self.phase_root=self.root_target.copy()
            elif age>.6:
                self.events.append({'time':t,'event':'landing_timeout','side':self.swing})
                self.phase='failed'
        elif self.phase=='center':
            desired=self.root_target.copy();desired[:2]=np.mean(self.foot_targets[:,:2],axis=0)+[.006,-.0006]
            self.root_target=self.phase_root+self.ease(age/1.0)*(desired-self.phase_root)
            if age>=1.0:self.phase='ready';self.phase_t=t
        self._solve(self.root_target,self.foot_targets,self.heading)
        # Finite greeting, driven through real shoulder/elbow/head/tail motors.
        if status=='arrived' and self.phase=='ready' and self.wave_start is None and not self.wave_done:
            self.wave_start=t;self.events.append({'time':t,'event':'wave_started'})
        weight=0.
        if self.wave_start is not None:
            u=t-self.wave_start
            if status!='arrived':
                self.wave_start=None;self.wave_done=True;self.events.append({'time':t,'event':'wave_cancelled'})
            elif u<1.:weight=self.ease(u)
            elif u<2.4:weight=1.
            elif u<3.5:weight=1-self.ease((u-2.4)/1.1)
            else:
                self.wave_done=True;self.wave_start=None;self.events.append({'time':t,'event':'wave_completed'})
        flutter=.08*np.sin(2*np.pi*(t-self.wave_start)/.7) if self.wave_start is not None and 1.<t-self.wave_start<2.4 else 0.
        self.target[self.model.actuator('right_shoulder_roll').id]=(-.60+flutter)*weight
        self.target[self.model.actuator('right_shoulder_pitch').id]=-.18*weight
        self.target[self.model.actuator('right_elbow').id]=-.55*weight
        self.target[self.model.actuator('tail_yaw').id]=.05*weight
        self.target[self.model.actuator('head_yaw').id]=np.clip(command[2]*.5,-.3,.3)
        self.target[self.model.actuator('head_pitch').id]=.04*weight
        # Joint targets remain slew-limited. No qpos/qvel or external wrench writes.
        self.data.ctrl[:]+=np.clip(self.target-self.data.ctrl,-.025,.025)
