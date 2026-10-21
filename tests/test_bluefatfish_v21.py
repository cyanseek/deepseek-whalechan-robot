"""Regression checks for continuous eyelids and physical tail articulation."""
import numpy as np
import mujoco
from opendafeiyu.bluefatfish_v2 import build_scene as build_v2
from opendafeiyu.bluefatfish_v21 import build_scene, ExpressiveController
from opendafeiyu.bluefatfish_expressions import Expressions


def test_relaxed_transition_keeps_original_eyelid_path_and_replay_morph():
    m,_,assets,_=build_v2();f=Expressions(m,mapping=assets['expression-cutout-map.npz'])
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);original=d.qpos.copy()
    f.update(1.,'relaxed');assert f.weights=={}
    previous=0.
    for t in np.linspace(1.01,1.28,28):
        f.update(t,'relaxed')
        assert f.weights.get('relaxed',0)==0
        w=f.weights.get('blink',0)
        assert previous<=w<=1 and w-previous<.06
        previous=w
    assert np.isclose(f.weights['blink'],1.)
    np.testing.assert_array_equal(d.qpos,original)
    before=m.skin_vert.copy();f.update(1.30,'happy')
    np.testing.assert_array_equal(m.skin_vert,before)
    f.update(1.58,'happy');assert np.isclose(f.weights['happy'],.85)
    assert f.weights.get('blink',0)<1e-10
    f.apply({'relaxed':.85})
    assert f.weights=={'relaxed':.85}  # Exact old explicit weights/replay remain possible.


def test_tail_keeps_rest_collision_geometry_and_mass_and_moves_with_torque():
    old,_,_,_=build_v2();m,_,_,b=build_scene()
    od=mujoco.MjData(old);d=mujoco.MjData(m)
    mujoco.mj_forward(old,od);mujoco.mj_forward(m,d)
    assert m.nu==23 and m.nmocap==2
    assert abs(m.body_subtreemass[m.body('base').id]-.972)<1e-8
    for name in ['tail_proximal_collision','tail_distal_collision','tail_flukes_collision']:
        oi=old.geom(name).id;ni=m.geom(name).id
        # Source bind quaternions are float32; compare capsule endpoints within
        # 0.1 micrometre. Rotation around a capsule's own axis is immaterial.
        np.testing.assert_allclose(od.geom_xpos[oi],d.geom_xpos[ni],atol=1e-7)
        np.testing.assert_allclose(od.geom_xmat[oi].reshape(3,3)[:,2],d.geom_xmat[ni].reshape(3,3)[:,2],atol=1e-6)
        assert m.geom_contype[ni]==old.geom_contype[oi]
    c=ExpressiveController(m,d,b);angles=[];tilts=[];tail_ground=0
    for i in range(12000):
        if i%10==0:
            if i==3000:assert c.request_tail()
            state=[x.copy() for x in (d.qpos,d.qvel,d.xfrc_applied,d.qfrc_applied)]
            c.update(np.zeros(7),'idle',walking=False)
            for before,after in zip(state,(d.qpos,d.qvel,d.xfrc_applied,d.qfrc_applied)):
                np.testing.assert_array_equal(before,after)
        mujoco.mj_step(m,d)
        angles.append([d.joint(n).qpos[0] for n in ['tail_yaw','tail_pitch','tail_bend']])
        tilts.append(np.arccos(np.clip(d.xmat[c.base].reshape(3,3)[2,2],-1,1)))
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-8)
        for contact in d.contact:
            names=[m.geom(contact.geom1).name,m.geom(contact.geom2).name]
            tail_ground+=('floor' in names and any(n.startswith('tail_') for n in names))
    assert np.all(np.ptp(np.array(angles)[3000:9000],axis=0)>[.25,.12,.12])
    assert max(tilts)<.05 and all(c.contacts()) and c.phase=='ready' and tail_ground==0
    assert any(e['event']=='tail_gesture_completed' for e in c.events)
