"""Native contact/balance checks for feedback motions and interrupted recentering."""
import mujoco
import numpy as np
import pytest

pytest.importorskip('clarabel', reason='Install requirements-contact.txt for native VD29 checks')
from opendafeiyu.bluefatfish_v24 import build_scene
from opendafeiyu.bluefatfish_v29 import FeedbackController
from opendafeiyu.dialogue import InteractionController, command_task
from opendafeiyu.motion_metrics import MotionMetrics
from test_contact_walk import advance


def test_head_feedback_and_cancellation_on_native_body():
    m, _, _, binding = build_scene()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = FeedbackController(m, d, binding)
    c = InteractionController(low); metrics = MotionMetrics(m, low)
    try:
        advance(m, d, low, metrics, 3, interaction=c)
        for word, axis, minimum in [('点头', 1, .06), ('摇头', 0, .10),
                                    ('向左看', 0, .15), ('向右看', 0, .15)]:
            c.submit(word, float(d.time))
            peak = 0.
            for _ in range(60):
                advance(m, d, low, metrics, .1, interaction=c)
                peak = max(peak, abs(d.qpos[low.qaddresses[low.head_ids[axis]]]))
                assert np.max(np.abs(low.channel_velocity['head_feedback'])) <= low.head_rate+1e-9
            assert peak > minimum
            assert c.status == 'complete' and low.feedback_settled()
            assert all(low.contacts())
        c.submit('摇头', float(d.time))
        advance(m, d, low, metrics, 1.3, interaction=c)
        assert abs(d.qpos[low.qaddresses[low.head_ids[0]]]) > .05
        c.submit('停一下', float(d.time))
        advance(m, d, low, metrics, .1, interaction=c)
        assert c.status == 'stopping'  # Do not announce a stationary body as fully stopped.
        advance(m, d, low, metrics, 3, interaction=c)
        assert c.status == 'stopped' and low.feedback_settled()
        assert any(e['event'] == 'head_gesture_cancelled' for e in low.events)
        assert low.solver_residual < 1e-4
    finally:
        c.close()


def test_walk_to_feedback_waits_for_support_and_can_resume():
    m, _, _, binding = build_scene()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = FeedbackController(m, d, binding); metrics = MotionMetrics(m, low)
    c = InteractionController(low)
    try:
        advance(m, d, low, metrics, 3)
        moving = np.array([.12, 0, 0, 0, 0, 0, 0])
        advance(m, d, low, metrics, 3, moving)
        # Seek a real swing before requesting feedback, not a synthetic body state.
        for _ in range(200):
            if low.phase == 'swing' and max(low.point(i)[2] for i in range(2)) > .002:
                break
            advance(m, d, low, metrics, .01, moving)
        assert low.phase == 'swing'
        before = low.steps_completed
        c.submit('点头', float(d.time))
        advance(m, d, low, metrics, .1, interaction=c)
        assert low.head_action is None
        advance(m, d, low, metrics, 6, interaction=c)
        assert c.status == 'complete' and all(low.contacts())
        advance(m, d, low, metrics, 4, moving)
        advance(m, d, low, metrics, 2)
        assert low.steps_completed > before+1
        assert low.phase == 'ready' and low.feedback_settled()
    finally:
        c.close()


def test_explicit_gestures_do_not_parse_negation_as_motion():
    assert command_task('不要点头')['action'] == 'talk'
    assert command_task('点头然后挥手')['action'] == 'talk'
