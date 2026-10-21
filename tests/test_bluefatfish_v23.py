"""Opaque garment coverage and real-engine small-step/regression checks."""
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish_v22 import build_scene as build_v22,CoordinatedController,CoordinatedDemo
from opendafeiyu.bluefatfish_v23 import build_scene,SmallStepController


@pytest.fixture(scope='module')
def built():return build_scene()


def simulate(built,kind,seconds=18):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=(CoordinatedController if kind=='previous' else SmallStepController)(m,d,b)
    demo=CoordinatedDemo(walk_steps=12,turn_steps=6) if kind=='sequence' else None
    start=end=None;final=[];max_tilt=0.;peak_clearance=0.
    for i in range(round(seconds/m.opt.timestep)):
        if i%10==0:
            state=(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)
            before=[x.copy() for x in state]
            if demo:cmd,status=demo.update(c)
            else:
                walking=3<=d.time<14;cmd=np.array([.12 if walking else 0,0,0,0,0,0,0]);status='approach' if walking else 'idle'
            c.update(cmd,status)
            for a,b0 in zip(before,state):np.testing.assert_array_equal(a,b0)
        mujoco.mj_step(m,d)
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)
        for contact in d.contact:
            names={m.geom(contact.geom1).name,m.geom(contact.geom2).name}
            if 'floor' in names:assert names<=set(('floor','left_sole','right_sole'))
        max_tilt=max(max_tilt,np.arccos(np.clip(d.xmat[c.base].reshape(3,3)[2,2],-1,1)))
        if i==3000:start=d.qpos[:3].copy()
        if i==13999:end=d.qpos[:3].copy()
        if c.phase in ('launch','flight') and not any(c.contacts()):
            clear=[]
            for side in ('left','right'):
                g=m.geom(side+'_sole').id;clear.append(d.geom_xpos[g,2]-np.abs(d.geom_xmat[g].reshape(3,3)[2])@m.geom_size[g])
            peak_clearance=max(peak_clearance,min(clear))
        if d.time>seconds-1:final.append(c.contacts())
    assert c.phase=='ready' and np.min(np.mean(final,axis=0))>.99
    return c,d,dict(displacement=end-start,tilt=max_tilt,clearance=peak_clearance,demo=demo)


def test_opaque_bloomers_cover_pelvis_and_preserve_existing_physics(built):
    m,_,_,b=built;old,_,_,_=build_v22()
    for name in ('body_mass','body_inertia','jnt_range','actuator_forcerange','dof_damping'):
        np.testing.assert_array_equal(getattr(m,name),getattr(old,name))
    for i in range(old.ngeom):
        original=old.geom(i);g=m.geom(original.name).id
        assert m.geom_contype[g]==old.geom_contype[i] and m.geom_conaffinity[g]==old.geom_conaffinity[i]
        np.testing.assert_array_equal(m.geom_size[g],old.geom_size[i])
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    # Rays from inside the covered pelvis must reach opaque garment geometry.
    center=d.geom_xpos[m.geom('bloomer_waist').id].copy()
    groups=np.array([0,0,1,0,0,0],dtype=np.uint8)
    for direction in [np.array([1.,0,0]),np.array([-1.,0,0]),np.array([0.,1,0]),np.array([0.,-1,0]),np.array([0.,0,-1.])]:
        gid=np.zeros(1,dtype=np.int32);distance=mujoco.mj_ray(m,d,center,direction,groups,1,-1,gid)
        assert 0<distance<.08 and m.geom(int(gid[0])).name.startswith('bloomer_')
        assert m.mat_rgba[m.geom_matid[int(gid[0])],3]==1.


def test_small_steps_are_physically_faster_than_previous_profile(built):
    old,_,previous=simulate(built,'previous');new,_,current=simulate(built,'current')
    assert new.steps_completed>=12 and new.steps_completed>old.steps_completed*1.5
    assert current['displacement'][0]>previous['displacement'][0]*2.
    assert current['tilt']<.12


def test_fast_walk_to_jump_to_turn_completes_without_body_or_tail_ground_contact(built):
    c,d,result=simulate(built,'sequence',39)
    assert result['demo'].stage=='complete' and c.steps_completed==18
    assert c.jump_done and c.jump_airborne and result['clearance']>.003
    assert result['tilt']<.20
    assert any(e['event']=='jump_support_targets_reanchored' for e in c.events)


@pytest.mark.parametrize('phase',['prepare','crouch','flight'])
def test_interrupting_jump_preparation_or_flight_returns_to_supported_stand(built,phase):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=SmallStepController(m,d,b)
    cancelled=False;support=[];tilt=[]
    for i in range(12000):
        if i%10==0:
            if i==3000:assert c.request_jump()
            if not cancelled and ((phase=='prepare' and c.jump_preparing and d.time>3.3) or c.phase==phase):
                before=d.qpos.copy();c.cancel();np.testing.assert_array_equal(before,d.qpos);cancelled=True
            c.update(np.zeros(7),'stopped' if cancelled else 'idle')
        mujoco.mj_step(m,d)
        if i>=11000:
            support.append(c.contacts());tilt.append(np.arccos(np.clip(d.xmat[c.base].reshape(3,3)[2,2],-1,1)))
    assert cancelled and c.phase=='ready' and not c.jump_preparing
    assert np.min(np.mean(support,axis=0))>.99 and np.any(support,axis=1).all() and max(tilt)<.05
    assert c.jump_airborne==(phase=='flight')
