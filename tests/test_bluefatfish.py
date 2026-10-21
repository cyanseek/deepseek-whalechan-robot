"""Boundaries of the delivered character's new native dynamics (no art scoring)."""
from pathlib import Path
import json
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish import build_scene, NativeController, DELIVERY, R_FORWARD
from opendafeiyu.tasks import detect_targets


@pytest.fixture(scope='module')
def built():
    # Intentionally fail if local delivery dependencies disappear; no engine skip.
    return build_scene()


def test_dynamic_body_is_free_and_skin_bound_without_mocap_or_weld(built):
    m,xml,assets,binding=built
    assert (m.nq,m.nv,m.nu,m.nskin,m.nmocap)==(28,27,21,9,2)
    assert m.neq==0 and m.jnt_type[0]==mujoco.mjtJoint.mjJNT_FREE
    assert all(m.body_mocapid[m.body(b).id]==-1 for b in binding['source_bones'].values())
    assert abs(binding['robot_mass_kg']-.972)<1e-9
    for name in ('left_sole','right_sole','head_collision','pelvis_collision','tail_distal_collision','skirt_collision_0'):
        geom=m.geom(name);assert geom.contype[0] and geom.conaffinity[0]
    for name,raw in assets.items():
        if name.endswith('.skn') or name.startswith('textures/'):
            assert raw==(DELIVERY/'mujoco_preview'/name).read_bytes()


def test_rest_skin_and_robot_rgb_are_real_engine_outputs(built):
    m,_,_,_=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    with mujoco.Renderer(m,240,320) as renderer:
        renderer.update_scene(d,camera='robot_camera',scene_option=option)
        rgb=renderer.render()
        expected=m.skin_vert@R_FORWARD.T+[0,0,.002]
        assert np.max(np.linalg.norm(renderer.scene.skinvert.reshape(-1,3)-expected,axis=1))<2e-6
        assert set(detect_targets(rgb))=={'red','green'}
        d.mocap_pos[m.body_mocapid[m.body('target_red').id]]=[-2,0,.24]
        mujoco.mj_forward(m,d);renderer.update_scene(d,camera='robot_camera',scene_option=option)
        assert 'red' not in detect_targets(renderer.render())


def test_controller_only_writes_motors_and_physics_produces_steps(built):
    m,_,_,binding=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    controller=NativeController(m,d,binding)
    singles=0
    for step in range(7000):
        if step%20==0:
            before=[v.copy() for v in (d.qpos,d.qvel,d.mocap_pos,d.xfrc_applied,d.qfrc_applied)]
            controller.update(np.array([.1,0,0,0,0,0,0]),'approach')
            for old,new in zip(before,(d.qpos,d.qvel,d.mocap_pos,d.xfrc_applied,d.qfrc_applied)):
                np.testing.assert_array_equal(old,new)
        mujoco.mj_step(m,d)
        singles+=sum(controller.contacts())==1
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-8)
    assert abs(d.time-7.)<1e-9 and controller.steps_completed>=2
    assert d.qpos[0]>.012 and singles>100
    assert controller.phase!='failed' and controller.max_ik_error<.0001


def test_original_author_and_restrictive_license_not_replaced():
    data=json.loads((DELIVERY/'source/model_document.json').read_text('utf8'))
    meta=data['extensions']['VRMC_vrm']['meta']
    assert meta['authors']==['木子恶魔']
    assert meta['creditNotation']=='required'
    assert meta['allowRedistribution'] is False and meta['modification']=='prohibited'
