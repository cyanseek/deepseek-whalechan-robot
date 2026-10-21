"""Real physics regression and derived clothing/expression checks for VD2.4."""
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish_v22 import build_scene as old_scene,CoordinatedDemo
from opendafeiyu.bluefatfish_v23 import SmallStepController
from opendafeiyu.bluefatfish_v24 import build_scene,LivelyController,CheerfulExpressions

@pytest.fixture(scope='module')
def built():return build_scene()

def test_new_clothing_preserves_original_physics_and_has_normalized_skin_weights(built):
    m,_,_,_=built;old,*_=old_scene()
    for name in ('body_mass','body_inertia','jnt_range','dof_damping','actuator_forcerange','actuator_gainprm','actuator_biasprm','geom_size','geom_contype','geom_conaffinity'):
        np.testing.assert_array_equal(getattr(m,name),getattr(old,name))
    si=m.skin('fitted_bloomers').id;weights=np.zeros(m.skin_vertnum[si])
    for bi in range(m.skin_boneadr[si],m.skin_boneadr[si]+m.skin_bonenum[si]):
        adr=m.skin_bonevertadr[bi];n=m.skin_bonevertnum[bi]
        weights[m.skin_bonevertid[adr:adr+n]]+=m.skin_bonevertweight[adr:adr+n]
    np.testing.assert_allclose(weights,1.,atol=1e-6)
    assert m.mat_rgba[m.skin_matid[si],3]==1
    verts=m.skin_vert[m.skin_vertadr[si]:m.skin_vertadr[si]+m.skin_vertnum[si]]
    assert abs(verts[:,0]).max()<.054 and abs(verts[:,1]).max()<.029 and verts[:,2].min()>.044
    shell=m.skin('fitted_bloomer_surface').id;weights=np.zeros(m.skin_vertnum[shell])
    for bi in range(m.skin_boneadr[shell],m.skin_boneadr[shell]+m.skin_bonenum[shell]):
        adr=m.skin_bonevertadr[bi];n=m.skin_bonevertnum[bi]
        weights[m.skin_bonevertid[adr:adr+n]]+=m.skin_bonevertweight[adr:adr+n]
    np.testing.assert_allclose(weights,1.,atol=1e-6)
    assert m.mat_rgba[m.skin_matid[shell],3]==1 and m.skin_vertnum[shell]>100

def test_default_smile_preserves_open_eye_shape_and_blends_smoothly(built):
    m,_,a,_=built;face=CheerfulExpressions(m,mapping=a['expression-cutout-map.npz'])
    face.update(2,'neutral');assert face.weights['smile']==pytest.approx(.7)
    eye=face.neutral[:,2]>.222
    np.testing.assert_array_equal(face.deltas['smile'][eye],0.)
    before=m.skin_vert.copy();face.update(2.01,'happy');np.testing.assert_allclose(m.skin_vert,before,atol=1e-8)
    face.update(2.30,'happy');assert face.weights['happy']==pytest.approx(.85)
    assert face.weights.get('smile',0)<1e-5
    m.skin_vert[:]=face.neutral

def test_closed_expression_returns_to_neutral_without_missing_zero_weight(built):
    m,_,a,_=built;face=CheerfulExpressions(m,mapping=a['expression-cutout-map.npz'])
    try:
        face.update(10,'relaxed');face.update(10.4,'relaxed')
        before=m.skin_vert.copy()
        face.update(10.5,'neutral')
        np.testing.assert_allclose(m.skin_vert,before,atol=1e-8)
        face.update(10.79,'neutral')
        assert face.weights['smile']==pytest.approx(.7)
        assert np.isfinite(m.skin_vert).all()
    finally:
        m.skin_vert[:]=face.neutral

def simulate(built,kind,seconds=18,cancel_phase=None):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=(SmallStepController if kind=='old' else LivelyController)(m,d,b)
    demo=CoordinatedDemo(walk_steps=16,turn_steps=6) if kind=='sequence' else None
    peak=0.;max_tilt=0.;start=end=None;cancelled=False;support=[]
    for i in range(round(seconds/m.opt.timestep)):
        if i%10==0:
            arrays=(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)
            before=[x.copy() for x in arrays]
            if demo:cmd,status=demo.update(c)
            elif kind=='jump':
                cmd=np.zeros(7);status='idle'
                if i==3000:assert c.request_jump()
                if cancel_phase and not cancelled and ((cancel_phase=='prepare' and c.jump_preparing and d.time>3.3) or c.phase==cancel_phase):
                    c.cancel();cancelled=True
            else:
                cmd=np.zeros(7);moving=3<=d.time<12;cmd[0]=.12 if moving else 0;status='approach' if moving else 'idle'
            c.update(cmd,status)
            for old,current in zip(before,arrays):np.testing.assert_array_equal(old,current)
        mujoco.mj_step(m,d)
        assert c.phase!='failed'
        max_tilt=max(max_tilt,float(np.arccos(np.clip(d.xmat[c.base].reshape(3,3)[2,2],-1,1))))
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)
        for contact in d.contact:
            names={m.geom(contact.geom1).name,m.geom(contact.geom2).name}
            if 'floor' in names:assert names<=set(('floor','left_sole','right_sole'))
        if i==3000:start=d.qpos[:3].copy()
        if i==11999:end=d.qpos[:3].copy()
        if c.phase in ('launch','flight') and not any(c.contacts()):
            clear=[]
            for side in ('left','right'):
                g=m.geom(side+'_sole').id;clear.append(d.geom_xpos[g,2]-np.abs(d.geom_xmat[g].reshape(3,3)[2])@m.geom_size[g])
            peak=max(peak,min(clear))
        if d.time>seconds-1:support.append(c.contacts())
    assert c.phase=='ready' and np.min(np.mean(support,axis=0))>.99
    return c,dict(peak=peak,tilt=max_tilt,delta=end-start if end is not None else None,demo=demo,cancelled=cancelled)

def test_short_step_speed_increases_in_same_nine_second_command_window(built):
    old,prev=simulate(built,'old');new,now=simulate(built,'new')
    assert new.steps_completed>old.steps_completed
    assert now['delta'][0]>prev['delta'][0]*1.6 and now['tilt']<.12

def test_complete_walk_jump_turn_sequence(built):
    c,r=simulate(built,'sequence',39)
    assert r['demo'].stage=='complete' and c.steps_completed==22 and c.jump_done
    assert r['peak']>.006 and r['tilt']<.20

def test_standalone_hop_has_larger_actual_clearance_and_recovers(built):
    c,r=simulate(built,'jump',12)
    assert c.jump_airborne and c.jump_done and r['peak']>.009 and r['tilt']<.10

def test_navigation_precision_mode_latches_only_between_steps(built):
    m,_,_,b=built;d=mujoco.MjData(m);c=LivelyController(m,d,b)
    c.set_precision_motion(True);assert c.motion_mode=='precision' and c.shift_seconds==.45
    c.phase='swing';c.set_precision_motion(False);assert c.motion_mode=='precision'
    c.phase='ready';c.set_precision_motion(c.precision_requested)
    assert c.motion_mode=='lively' and c.shift_seconds==.28

@pytest.mark.parametrize('phase',['prepare','crouch','flight'])
def test_cancellation_completes_supported_recovery(built,phase):
    c,r=simulate(built,'jump',12,cancel_phase=phase)
    assert r['cancelled'] and not c.jump_preparing
    assert c.jump_airborne==(phase=='flight')
