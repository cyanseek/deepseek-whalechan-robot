import threading,time
import mujoco,numpy as np,pytest
from opendafeiyu.bluefatfish_v24 import build_scene
from opendafeiyu.bluefatfish_v29 import FeedbackController
from opendafeiyu.behaviors import BehaviorController
from opendafeiyu.dialogue import command_task

@pytest.fixture(scope='module')
def scene():return build_scene()

def runtime(scene,translator=None):
    m,_,_,b=scene;d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return m,d,BehaviorController(FeedbackController(m,d,b),translator)

def advance(m,d,c,seconds):
    for _ in range(round(seconds/m.opt.timestep)):
        if round(d.time/m.opt.timestep)%2==0:
            arrays=(d.qpos,d.qvel,d.xfrc_applied,d.qfrc_applied)
            before=[x.copy() for x in arrays];c.tick({},float(d.time))
            for x,y in zip(before,arrays):np.testing.assert_array_equal(x,y)
        c.low.apply_actuation();mujoco.mj_step(m,d)
        assert c.low.phase!='failed'
        assert np.all(abs(d.actuator_force)<=m.actuator_forcerange[:,1]+1e-7)

def test_sequence_pause_resume_finishes_only_on_feedback(scene):
    m,d,c=runtime(scene);advance(m,d,c,3)
    c.submit('打个招呼',float(d.time));advance(m,d,c,1)
    c.submit('停一下',float(d.time));advance(m,d,c,2)
    assert c.status=='stopped' and c.behavior_state()['can_resume']
    assert c.saved==['wave','tail','nod']
    c.submit('继续',float(d.time));advance(m,d,c,18)
    assert c.behavior_status=='complete' and not c.saved and all(c.low.contacts())
    done=[e['token'] for e in c.events if e['event']=='sequence_step_completed']
    assert done==['wave','tail','nod']
    assert sum(e['event']=='sequence_completed' for e in c.events)==1

def test_cancel_clears_future_steps_and_resume_does_not_restart(scene):
    m,d,c=runtime(scene);advance(m,d,c,3)
    c.submit_sequence(['tail','nod'],float(d.time));advance(m,d,c,1)
    c.submit('取消任务',float(d.time));advance(m,d,c,2)
    c.submit('继续',float(d.time));advance(m,d,c,3)
    assert c.status=='stopped' and not c.plan and not c.saved
    assert not any(e['event']=='sequence_step_completed' for e in c.events)

def test_invalid_sequence_is_atomic_and_new_intent_replaces_saved(scene):
    m,d,c=runtime(scene);advance(m,d,c,3);c.submit_sequence(['wave','tail'],float(d.time))
    generation=c.generation
    with pytest.raises(ValueError):c.submit_sequence(['wave','raw_joints'],float(d.time))
    with pytest.raises(ValueError):c.submit_sequence(['jump'],float(d.time))
    assert c.generation==generation and c.plan==['wave','tail']
    c.submit('停一下',float(d.time));advance(m,d,c,1);c.submit('点头',float(d.time));advance(m,d,c,5)
    assert not c.saved and not c.plan and c.status=='complete'

def test_delayed_model_cannot_restart_after_pause(scene):
    release=threading.Event()
    def delayed(text):
        release.wait(4);return {'task':command_task('摆尾'),'mode':'delayed_test_fixture'}
    m,d,c=runtime(scene,delayed)
    try:
        advance(m,d,c,3);c.submit('请摆尾',float(d.time));advance(m,d,c,.1)
        c.submit('停一下',float(d.time));advance(m,d,c,1);release.set()
        end=time.perf_counter()+2
        while c.pending and time.perf_counter()<end:c.poll(float(d.time));time.sleep(.001)
        assert not c.saved and c.status=='stopped'
        assert any(e['event']=='stale_response_rejected' for e in c.events)
    finally:release.set();c.close()

def test_deadline_stops_and_discards_unfinished_routine(scene):
    m,d,c=runtime(scene);advance(m,d,c,3);c.submit_sequence(['tail','wave'],float(d.time))
    c.deadline=float(d.time)+.1;advance(m,d,c,2)
    assert c.behavior_status=='timed_out' and c.status=='stopped'
    assert not c.plan and not c.saved
