import json
import threading
import time
import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish_v24 import build_scene, LivelyController
from opendafeiyu.dialogue import InteractionController, command_task, validate_task, DeepSeekClient

@pytest.fixture(scope='module')
def built():return build_scene()

def runtime(built,translator=None):
    m,_,_,b=built;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return m,d,InteractionController(LivelyController(m,d,b),translator)

def advance(m,d,c,seconds):
    for _ in range(round(seconds/m.opt.timestep)):
        if round(d.time/m.opt.timestep)%10==0:
            arrays=(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)
            before=[a.copy() for a in arrays];c.tick({},float(d.time))
            for old,new in zip(before,arrays):np.testing.assert_array_equal(old,new)
        mujoco.mj_step(m,d)
        assert c.low.phase!='failed'
        assert np.all(np.abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)

def test_wave_tail_cancel_and_repeat_complete_in_real_physics(built):
    m,d,c=runtime(built);advance(m,d,c,3)
    for text,seconds in [('挥手',5),('摆尾',8),('挥手',5)]:
        c.submit(text,float(d.time));advance(m,d,c,seconds)
        assert c.status=='complete' and c.low.phase=='ready'
    c.submit('摆尾',float(d.time));advance(m,d,c,2)
    assert c.low.tail_started is not None
    c.submit('停一下',float(d.time));advance(m,d,c,2)
    assert c.status=='stopped' and c.low.tail_started is None
    assert all(c.low.contacts())

def test_stop_does_not_wait_for_delayed_model_and_rejects_its_result(built):
    release=threading.Event()
    def delayed(text):
        release.wait(4)
        return {'task':command_task('摆尾'),'mode':'test_delayed_translator'}
    m,d,c=runtime(built,delayed)
    try:
        advance(m,d,c,3);c.submit('摆尾',float(d.time));advance(m,d,c,.1)
        c.submit('停一下',float(d.time));advance(m,d,c,1)
        assert c.status=='stopped' and c.low.tail_started is None
        release.set()
        deadline=time.perf_counter()+2
        while c.pending and time.perf_counter()<deadline:c.poll(float(d.time));time.sleep(.001)
        assert any(x['event']=='stale_response_rejected' for x in c.events)
        assert c.status=='stopped' and c.low.tail_started is None
    finally:release.set();c.close()

def test_invalid_model_response_keeps_body_stopped(built):
    m,d,c=runtime(built,lambda text:{'task':{'action':'raw_joints','target':'none','reply':'x'},'mode':'test_invalid'})
    try:
        advance(m,d,c,3);c.submit('test',float(d.time));advance(m,d,c,2)
        c.poll(float(d.time))
        assert c.status=='model_error' and c.action is None and c.low.phase=='ready'
    finally:c.close()

def test_stale_action_is_rejected_after_new_request(built):
    m,d,c=runtime(built);advance(m,d,c,3)
    old=c.submit('过来',float(d.time));c.submit('挥手',float(d.time))
    assert not c.accept(old,command_task('摆尾'),float(d.time),'stale_test')
    advance(m,d,c,5);assert c.status=='complete'
    assert [e['action'] for e in c.events if e['event']=='action_completed']==['wave']

def test_model_stop_after_initial_settle_reaches_stopped(built):
    m,d,c=runtime(built);advance(m,d,c,3)
    generation=c.submit('稍等，先别动了',float(d.time));advance(m,d,c,1)
    assert c.stop_requested is None
    c.accept(generation,command_task('停一下'),float(d.time),'delayed_model_test')
    advance(m,d,c,1)
    assert c.status=='stopped' and all(c.low.contacts())

def test_completion_face_returns_to_open_eyed_idle(built):
    m,d,c=runtime(built);advance(m,d,c,3)
    c.submit('挥手',float(d.time));advance(m,d,c,4)
    assert c.status=='complete' and c.expression(float(d.time))=='happy'
    advance(m,d,c,2)
    assert c.expression(float(d.time))=='neutral'

@pytest.mark.parametrize('task',[
    {'action':'wave','target':'red','reply':'hi'},
    {'action':'approach','target':'none','reply':'hi'},
    {'action':'tail','target':'none','reply':'hi','joints':[1,2]},
    {'action':'raw_jump_torque','target':'none','reply':'hi'},
])
def test_language_intents_cannot_escape_supported_actions(task):
    with pytest.raises(ValueError):validate_task(task)

def test_negative_command_is_not_treated_as_stop_or_motion():
    assert command_task('不要停一下')['action']=='talk'
    assert command_task('不要过来')['action']=='talk'

def test_missing_key_and_insecure_url_are_rejected_without_exposing_config(tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    p=tmp_path/'private.json';p.write_text(json.dumps({'api_key':''}))
    with pytest.raises(ValueError,match='Fill api_key'):DeepSeekClient(p)
    p.write_text(json.dumps({'base_url':'http://example.com','api_key':'do-not-print'}))
    with pytest.raises(ValueError,match='HTTPS'):DeepSeekClient(p)

def test_vd25_rejects_interactive_hop_and_keeps_support(built):
    m,d,c=runtime(built);advance(m,d,c,3)
    c.submit('跳一下',float(d.time));advance(m,d,c,3)
    assert c.action is None and c.status=='reply'
    assert c.low.phase=='ready' and all(c.low.contacts())
    assert any(e['event']=='unsupported_profile_action' for e in c.events)

@pytest.mark.parametrize('text', ['点头', '摇头', '向左看', '向右看'])
def test_legacy_profile_declines_new_head_actions(built, text):
    m,d,c=runtime(built)
    try:
        c.submit(text,float(d.time))
        assert c.action is None and c.status=='reply'
        assert c.events[-1]['event']=='unsupported_profile_action'
    finally:c.close()
