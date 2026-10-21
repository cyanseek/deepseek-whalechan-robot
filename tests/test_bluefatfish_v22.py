"""Physical coordination and interruption checks for the current character."""
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish_v21 import build_scene as build_v21
from opendafeiyu.bluefatfish_v22 import build_scene, CoordinatedController, CoordinatedDemo


@pytest.fixture(scope='module')
def built():
    return build_scene()


def controlled_step(m,d,c,command,status):
    state=(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)
    before=[x.copy() for x in state]
    c.update(command,status)
    for old,new in zip(before,state):
        np.testing.assert_array_equal(old,new)


def physical_step(m,d):
    mujoco.mj_step(m,d)
    assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)
    for contact in d.contact:
        names={m.geom(contact.geom1).name,m.geom(contact.geom2).name}
        if 'floor' in names:
            assert names<=set(('floor','left_sole','right_sole'))


def test_coordinated_sequence_finishes_with_physical_gait_takeoff_and_landing(built):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=CoordinatedController(m,d,b);demo=CoordinatedDemo()
    walked=[];critical=[];expressions=set();final=[];airborne=0
    for i in range(39000):
        if i%10==0:
            cmd,status=demo.update(c);controlled_step(m,d,c,cmd,status)
            expressions.add(c.expression)
            if demo.stage=='walk' and c.phase=='swing':
                walked.append([d.joint(n).qpos[0] for n in ('tail_yaw','tail_bend','left_shoulder_pitch','right_shoulder_pitch')])
            if c.phase in ('crouch','launch','flight'):
                critical.append([d.joint(n).qpos[0] for n in ('tail_yaw','tail_bend')])
        physical_step(m,d)
        if c.phase in ('launch','flight') and not any(c.contacts()):airborne+=1
        if i>=38000:final.append((all(c.contacts()),d.xmat[c.base].reshape(3,3)[2,2]))
    assert demo.stage=='complete' and c.phase=='ready' and c.steps_completed==10
    assert c.jump_airborne and c.jump_done and airborne>=12
    assert all(contact and level>np.cos(.10) for contact,level in final)
    assert np.all(np.ptp(walked,axis=0)>[.07,.02,.08,.08])
    # Gravity deflects the distal segment slightly; assert a restrained physical
    # tail, rather than equating a zero motor target with a zero joint angle.
    critical=np.asarray(critical)
    assert np.max(np.abs(critical[:,0]))<.01
    assert np.max(np.abs(critical[:,1]))<.05
    assert {'neutral','happy','focus','effort','delight'}<=expressions
    assert any(e['event']=='tail_gesture_completed' for e in c.events)
    stages=[e['stage'] for e in demo.events]
    assert stages.index('landed')<stages.index('turn')<stages.index('celebrate')


@pytest.mark.parametrize('when',['prepare','crouch','flight'])
def test_cancel_during_tail_to_jump_transition_keeps_contacts_and_recovers(built,when):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=CoordinatedController(m,d,b);cancelled=False;final=[]
    for i in range(11000):
        if i%10==0:
            if i==3000:assert c.request_tail()
            if i==4500:assert c.request_jump()
            cancel_now=(when=='prepare' and c.jump_preparing and d.time>4.7) or c.phase==when
            if cancel_now and not cancelled:
                c.cancel();cancelled=True
            controlled_step(m,d,c,np.zeros(7),'stopped' if cancelled else 'idle')
        physical_step(m,d)
        if i>=10000:final.append((c.contacts(),d.xmat[c.base].reshape(3,3)[2,2]))
    assert cancelled and c.phase=='ready' and not c.jump_preparing and c.tail_started is None
    assert c.jump_airborne==(when=='flight')
    # The 1 kHz solver can lose one sole-edge contact for a single millisecond.
    # Require both feet supported >99% of the last second, no whole-body flight,
    # and a tighter final tilt bound, as in the existing balance regression.
    feet=np.array([contact for contact,_ in final])
    assert np.all(feet.mean(axis=0)>.99) and feet.any(axis=1).all()
    assert all(level>np.cos(.05) for _,level in final)
    if when=='flight':
        assert c.jump_done and any(e['event']=='jump_landing' for e in c.events)


def test_coordination_preserves_existing_body_and_actuator_limits(built):
    old,_,_,_=build_v21();m,_,_,_=built
    for name in ('body_mass','body_inertia','geom_contype','geom_conaffinity',
                 'actuator_forcerange','jnt_range','geom_size','dof_damping'):
        np.testing.assert_array_equal(getattr(m,name),getattr(old,name))
