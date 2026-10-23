"""Native checks for the optional VD28 profile; no model/physics substitute."""
import mujoco
import numpy as np
import pytest

pytest.importorskip('clarabel', reason='Install requirements-contact.txt for VD28 native checks')
from opendafeiyu.bluefatfish_v24 import build_scene
from opendafeiyu.bluefatfish_v28 import ContactWalkController
from opendafeiyu.dialogue import InteractionController
from opendafeiyu.motion_metrics import MotionMetrics
from opendafeiyu.tasks import Detection


@pytest.fixture(scope='module')
def built():
    return build_scene()


def runtime(built):
    m, _, _, b = built
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = ContactWalkController(m, d, b)
    return m, d, low, MotionMetrics(m, low)


def advance(m, d, low, metrics, seconds, command=None, interaction=None):
    command = np.zeros(7) if command is None else command
    for _ in range(round(seconds/m.opt.timestep)):
        before = [a.copy() for a in (d.qpos, d.qvel, d.mocap_pos, d.mocap_quat, d.qfrc_applied, d.xfrc_applied)]
        if round(d.time/m.opt.timestep) % 2 == 0:
            if interaction:
                interaction.tick({}, float(d.time))
            else:
                low.update(command, 'approach' if command[0] else 'idle')
        low.apply_actuation()
        for old, current in zip(before, (d.qpos, d.qvel, d.mocap_pos, d.mocap_quat, d.qfrc_applied, d.xfrc_applied)):
            np.testing.assert_array_equal(old, current)
        mujoco.mj_step(m, d)
        assert np.all(np.abs(d.actuator_force) <= m.actuator_forcerange[:, 1]+1e-7)
        if round(d.time/m.opt.timestep) % 10 == 0:
            sample = metrics.sample(d, low.phase in ('shift', 'swing', 'land'))
            assert sample['tilt_rad'] < .15
            assert not metrics.other_floor


def test_real_foot_lift_and_stop_resume_without_state_writes(built):
    m, d, low, metrics = runtime(built)
    advance(m, d, low, metrics, 3)
    moving = np.array([.12, 0, 0, 0, 0, 0, 0])
    advance(m, d, low, metrics, 4, moving)
    assert low.steps_completed >= 2
    before = low.steps_completed
    low.cancel()
    advance(m, d, low, metrics, 2)
    assert low.phase == 'ready' and all(low.contacts())
    assert low.steps_completed <= before+1
    before = low.steps_completed
    advance(m, d, low, metrics, 4, moving)
    advance(m, d, low, metrics, 2)
    assert low.steps_completed > before
    assert low.phase == 'ready' and all(low.contacts())
    assert min(metrics.report()['walking_peak_sole_clearance_m']) > .003
    assert low.solver_residual < 1e-4


def test_supported_gestures_and_tail_interrupt_in_native_physics(built):
    m, d, low, metrics = runtime(built)
    c = InteractionController(low)
    try:
        advance(m, d, low, metrics, 3, interaction=c)
        for word, duration in [('挥手', 5), ('摆尾', 8)]:
            c.submit(word, float(d.time))
            advance(m, d, low, metrics, duration, interaction=c)
            assert c.status == 'complete'
        c.submit('摆尾', float(d.time))
        advance(m, d, low, metrics, 2, interaction=c)
        assert low.tail_started is not None
        c.submit('停一下', float(d.time))
        advance(m, d, low, metrics, 1, interaction=c)
        assert c.status == 'stopped' and low.tail_started is None
        c.submit('跳一下', float(d.time))
        advance(m, d, low, metrics, 1, interaction=c)
        assert c.status == 'reply' and low.phase == 'ready' and all(low.contacts())
    finally:
        c.close()


def test_side_target_is_declined_in_straight_profile(built):
    m, d, low, metrics = runtime(built)
    c = InteractionController(low)
    try:
        advance(m, d, low, metrics, 3, interaction=c)
        c.submit('过来', float(d.time))
        advance(m, d, low, metrics, .2, interaction=c)
        # This is an explicit synthetic perception input, not a camera test.
        from types import SimpleNamespace
        c.tick({'red': SimpleNamespace(bearing=.4, distance=.7)}, float(d.time))
        assert c.status == 'stopping' and c.action is None
        assert any(e['event'] == 'unsupported_profile_bearing' for e in c.events)
    finally:
        c.close()
