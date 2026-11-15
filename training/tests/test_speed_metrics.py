from copy import deepcopy
import json

import numpy as np
import pytest

from whalechan_training.speed_metrics import walking_speed_windows


def row(t, speed, command=.05):
    return {"t": t, "speed_m_s": speed, "command": [command, 0., 0.]}


def test_oscillation_is_not_hidden_by_matching_average_speed():
    rows = [row(2.02, 0.), row(2.04, .1)]
    original = deepcopy(rows)
    first, second = walking_speed_windows(rows)
    assert first["mean_command_m_s"] == pytest.approx(.05)
    assert first["mean_forward_speed_m_s"] == pytest.approx(.05)
    assert first["mean_error_m_s"] == pytest.approx(0.)
    assert first["mae_m_s"] == pytest.approx(.05)
    assert first["rmse_m_s"] == pytest.approx(.05)
    assert first["samples"] == 2 and not first["complete"]
    assert second["samples"] == 0 and second["mae_m_s"] is None
    assert rows == original


@pytest.mark.parametrize("speed,error", [(0., -.05), (-.05, -.1), (.05, 0.)])
def test_standing_backward_and_exact_tracking(speed, error):
    result = walking_speed_windows([row(2.02, speed)])[0]
    assert result["mean_error_m_s"] == pytest.approx(error)
    assert result["mae_m_s"] == pytest.approx(abs(error))
    assert result["rmse_m_s"] == pytest.approx(abs(error))


def test_errors_use_each_samples_command():
    result = walking_speed_windows([row(2.02, .1, .1), row(2.04, .1, .2)])[0]
    assert result["mean_error_m_s"] == pytest.approx(-.05)
    assert result["mae_m_s"] == pytest.approx(.05)
    assert result["rmse_m_s"] == pytest.approx(np.sqrt(.005))


def test_post_step_boundaries_and_initial_stand():
    rows = [row(.02, .8), row(2., .9, 0.), row(2.02, .03), row(8., .05),
            row(8.02, 99., 0.), row(11., 99., 0.), row(11.02, .07), row(17., .09),
            row(17.02, 99., 0.)]
    first, second = walking_speed_windows(rows)
    assert first["samples"] == second["samples"] == 2
    assert first["mean_forward_speed_m_s"] == pytest.approx(.04)
    assert second["mean_forward_speed_m_s"] == pytest.approx(.08)
    assert first["observed_from_s"] == 2.02 and first["observed_until_s"] == 8.
    immediate = walking_speed_windows(rows, start_immediately=True)[0]
    assert immediate["command_s"] == 0. and immediate["samples"] == 3


@pytest.mark.parametrize("immediate", [False, True])
def test_complete_requires_all_control_samples_and_tolerates_clock_roundoff(immediate):
    start = 0. if immediate else 2.
    rows = [row(float(t + 1e-11), .05) for t in np.arange(start + .02, 8.01, .02)]
    assert walking_speed_windows(rows, start_immediately=immediate)[0]["complete"]
    del rows[10]
    assert not walking_speed_windows(rows, start_immediately=immediate)[0]["complete"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_sample_is_visible_and_does_not_become_zero_error(value):
    result = walking_speed_windows([row(2.02, value), row(2.04, .05)])[0]
    assert result["invalid_samples"] == 1 and result["samples"] == 2
    assert result["mean_forward_speed_m_s"] is None and result["rmse_m_s"] is None
    json.dumps(result, allow_nan=False)


def test_empty_zero_commands_and_partial_failure_keep_missing_evidence():
    empty = walking_speed_windows([])
    assert empty == walking_speed_windows([row(2.02, .1, 0.)])
    assert all(r["samples"] == 0 and not r["complete"] and r["observed_until_s"] is None for r in empty)
    first = walking_speed_windows([row(2.003, .01)])[0]
    assert first["samples"] == 1 and not first["complete"]
    assert first["mae_m_s"] == pytest.approx(.04)
