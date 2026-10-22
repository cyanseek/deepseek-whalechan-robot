import mujoco
import numpy as np
import pytest
from opendafeiyu.bluefatfish_v24 import build_scene
from opendafeiyu.bluefatfish_v26 import MeasuredStepController
from opendafeiyu.dialogue import InteractionController
from opendafeiyu.motion_metrics import MotionMetrics


@pytest.fixture(scope='module')
def built():
    return build_scene()


@pytest.mark.parametrize('input_time,stop_phase', [(3., None), (3.01, None), (3.01, 'crouch'), (3.01, 'flight')])
def test_hop_timing_and_cancel_keep_physical_support(built, input_time, stop_phase):
    m, _, _, binding = built
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = MeasuredStepController(m, d, binding)
    c = InteractionController(low); metrics = MotionMetrics(m, low)
    sent = stopped = False
    for step in range(18000):
        now = float(d.time)
        if not sent and now >= input_time:
            c.submit('跳一下', now); sent = True
        if stop_phase and low.phase == stop_phase and not stopped:
            c.submit('停一下', now); stopped = True
        if step % 10 == 0:
            arrays = (d.qpos, d.qvel, d.mocap_pos, d.mocap_quat, d.xfrc_applied, d.qfrc_applied)
            before = [a.copy() for a in arrays]
            c.tick({}, now)
            for old, current in zip(before, arrays): np.testing.assert_array_equal(old, current)
        mujoco.mj_step(m, d)
        assert np.all(np.abs(d.actuator_force) <= m.actuator_forcerange[:, 1] + 1e-7)
        if step % 10 == 0: metrics.sample(d)
    assert low.phase == 'ready' and all(low.contacts())
    assert not metrics.other_floor and metrics.max_tilt < .15
    if stop_phase:
        assert stopped and c.status == 'stopped'
        assert any(e['event'] == 'supported_stop' for e in c.events)
    else:
        assert c.status == 'complete' and low.jump_done
        assert metrics.max_clearance > .002 and metrics.max_air_time >= .03


def test_vd26_arrival_contract_is_bounded_and_does_not_change_vd25(built):
    from opendafeiyu.bluefatfish_v24 import LivelyController
    m, _, _, binding = built
    for cls, distance, bearing in [(LivelyController, .32, .15), (MeasuredStepController, .35, .50)]:
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        c = InteractionController(cls(m, d, binding))
        assert c.nav.arrival_distance == distance and c.nav.arrival_bearing == bearing
