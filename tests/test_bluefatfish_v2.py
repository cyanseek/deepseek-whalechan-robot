"""Actual engine checks for movement, jumping, balance and original expressions."""
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish import build_scene as build_v1
from opendafeiyu.bluefatfish_v2 import build_scene, EmbodiedController
from opendafeiyu.bluefatfish_expressions import Expressions


@pytest.fixture(scope='module')
def built():return build_scene()


def simulate(built, seconds=9, mode='stand', balance=True, axis=1):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d);c=EmbodiedController(m,d,b,balance=balance)
    tilt=[];contacts=[];tail_contacts=0
    for i in range(round(seconds/m.opt.timestep)):
        if i%10==0:
            if mode=='jump' and i==3000:assert c.request_jump()
            saved=[x.copy() for x in (d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)]
            c.update(np.array([.015,0,.35,0,0,0,0]) if mode=='turn' else np.zeros(7),'approach' if mode=='turn' else 'idle')
            for old,new in zip(saved,(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)):
                np.testing.assert_array_equal(old,new)
        if mode=='push':d.xfrc_applied[c.base,axis]=5. if 4000<=i<4150 else 0.
        mujoco.mj_step(m,d)
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)
        tilt.append(np.arccos(np.clip(d.xmat[c.base].reshape(3,3)[2,2],-1,1)))
        contacts.append(c.contacts())
        floor=m.geom('floor').id
        tail_contacts+=any('tail' in m.geom(x.geom1 if x.geom2==floor else x.geom2).name for x in d.contact if x.geom1==floor or x.geom2==floor)
    return c,d,np.array(tilt),np.array(contacts),tail_contacts


def test_alpha_adaptation_preserves_dynamics_and_morphs_change_render(built):
    m,_,assets,_=built;original,_,_,_=build_v1()
    for name in ('body_mass','body_inertia','geom_contype','geom_conaffinity','actuator_forcerange','jnt_range'):
        np.testing.assert_array_equal(getattr(m,name),getattr(original,name))
    face=Expressions(m,mapping=assets['expression-cutout-map.npz']);neutral=m.skin_vert.copy()
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);qpos=d.qpos.copy()
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.27];camera.distance=.49;camera.azimuth=175;camera.elevation=-2
    with mujoco.Renderer(m,240,320) as renderer:
        renderer.update_scene(d,camera=camera);image=renderer.render().copy()
        face.apply({'happy':1});renderer.update_scene(d,camera=camera);happy=renderer.render().copy()
        assert np.mean(np.abs(image.astype(float)-happy))>.5
        assert np.linalg.norm(m.skin_vert-neutral,axis=1).max()>.005
        np.testing.assert_array_equal(qpos,d.qpos)
    face.apply({});np.testing.assert_array_equal(neutral,m.skin_vert)
    assert face.report['max_mapping_error_m']<1e-6 and len(face.deltas)==17


def test_turn_uses_physical_steps_and_changes_body_heading(built):
    c,d,tilt,contact,tail=simulate(built,seconds=12,mode='turn')
    from opendafeiyu.bluefatfish import R_FORWARD
    orientation=d.xmat[c.base].reshape(3,3)@R_FORWARD.T
    assert np.arctan2(orientation[1,0],orientation[0,0])>.12
    assert c.steps_completed>=5 and max(tilt)<.12 and tail==0
    assert np.any(contact.sum(axis=1)==1)


def test_small_jump_leaves_ground_and_recovers_on_feet(built):
    c,d,tilt,contact,tail=simulate(built,seconds=9,mode='jump')
    assert c.jump_airborne and c.jump_done and c.phase=='ready'
    assert np.sum(~contact[3000:].any(axis=1))>=20
    assert contact[-1000:].all() and max(tilt[-1000:])<.05
    assert max(tilt)<.18 and tail==0


@pytest.mark.parametrize('axis',[0,1])
def test_imu_feedback_reduces_tilt_in_bounded_push(built,axis):
    enabled=simulate(built,mode='push',balance=True,axis=axis)
    disabled=simulate(built,mode='push',balance=False,axis=axis)
    assert max(enabled[2][4000:6000])<max(disabled[2][4000:6000])
    # Contact solver can report isolated 1 ms edge separations; require each
    # foot's support over 99% of the final second and no airborne body.
    assert np.all(enabled[3][-1000:].mean(axis=0)>.99)
    assert enabled[3][-1000:].any(axis=1).all() and max(enabled[2][-1000:])<.05
