import numpy as np
import pytest

pytest.importorskip('cv2')
from opendafeiyu.tasks import Detection, TaskController, detect_targets, manual_task


def active(target='red'):
    controller = TaskController()
    token = controller.request(0.)
    controller.accept(token, {'action': 'approach', 'target': target}, 0., 'test')
    return controller, token


def test_unknown_manual_input_cannot_start_walking():
    assert manual_task(' GREEN ') == {'action': 'approach', 'target': 'green'}
    assert manual_task('停下') == {'action': 'stop', 'target': 'none'}
    with pytest.raises(ValueError):
        manual_task('unrecognized input')


def test_cancel_invalidates_pending_language_and_clears_motion():
    c, token = active()
    visible = {'red': Detection('red', .1, 1., 100)}
    assert c.command(visible, .1)[0] > 0
    c.cancel(.2)
    assert not c.accept(token, {'action': 'approach', 'target': 'red'}, .3, 'late')
    assert np.array_equal(c.command(visible, .3), np.zeros(7))


def test_target_switch_uses_new_visible_target():
    c, _ = active()
    token = c.request(1.)
    c.accept(token, {'action': 'approach', 'target': 'green'}, 1., 'test')
    command = c.command({'red': Detection('red', .2, .8, 100),
                         'green': Detection('green', -.2, .8, 100)}, 1.1)
    assert command[2] < 0 and c.target == 'green'


def test_missing_target_stops_forward_immediately_then_reports_lost():
    c, _ = active()
    c.command({'red': Detection('red', 0, .8, 100)}, 0.)
    assert np.array_equal(c.command({}, .1), np.zeros(7))
    c.command({}, .7)
    assert c.status == 'lost'


def test_new_goal_search_is_bounded_and_uses_previous_visual_bearing():
    c, _ = active()
    c.command({'green': Detection('green', -.3, .8, 100)}, .1)
    token = c.request(.2)
    c.accept(token, {'action': 'approach', 'target': 'green'}, .2, 'test')
    command = c.command({}, .3)
    assert command[0] == 0 and command[2] < 0
    assert not c.command({}, 4.3).any()
    assert c.status == 'lost'


def test_arrival_requires_observed_proximity_and_tolerates_pixel_jitter():
    c, _ = active()
    c.command({'red': Detection('red', 0, .5, 100)}, 10.)
    assert c.status == 'approach'
    c.command({'red': Detection('red', 0, .31, 100)}, 10.1)
    c.command({'red': Detection('red', 0, .33, 100)}, 10.2)
    c.command({'red': Detection('red', 0, .33, 100)}, 10.5)
    assert c.status == 'arrived'


@pytest.mark.parametrize('task', [{'action': 'raw_pwm', 'target': 'red'},
                                 {'action': 'approach', 'target': 'person'},
                                 {'action': 'approach', 'target': 'red', 'joints': []}])
def test_language_cannot_bypass_task_interface(task):
    c = TaskController()
    token = c.request(0.)
    with pytest.raises(ValueError):
        c.accept(token, task, 0., 'test')


def test_detection_uses_pixels_and_rejects_truncated_marker():
    import cv2
    frame = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(frame, (100, 120), 15, (255, 0, 0), -1)
    result = detect_targets(frame)['red']
    assert result.bearing > 0 and .55 < result.distance < .7
    assert not detect_targets(np.zeros_like(frame))
    frame[:] = 0
    cv2.circle(frame, (100, 238), 15, (255, 0, 0), -1)
    assert not detect_targets(frame)
