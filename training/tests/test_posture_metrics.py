from copy import deepcopy
import json
import math

import pytest

from whalechan_training.posture_metrics import posture_windows


def row(t, tilt=6., heading=-4., command=.05):
    return dict(t=t, tilt_rad=math.radians(tilt), heading_error_rad=math.radians(heading),
                command=[command, 0., 0.])


def test_tilt_and_wrapped_heading_keep_initial_reference():
    rows = [row(2.02, 6., 179.), row(2.04, 18., -179.)]
    original = deepcopy(rows)
    result = posture_windows(rows)[0]
    assert result["mean_tilt_deg"] == pytest.approx(12.)
    assert result["sampled_max_tilt_deg"] == pytest.approx(18.)
    assert result["max_abs_heading_error_deg"] == pytest.approx(179.)
    assert result["final_heading_error_deg"] == pytest.approx(-179.)
    assert not result["complete"] and rows == original


def test_boundaries_and_commands_select_walk_and_stop_samples():
    rows = [row(.02), row(2., command=0.), row(2.02), row(8.),
            row(8.02, command=0.), row(11., command=0.), row(11.02), row(17.),
            row(17.02, command=0.), row(20., command=0.)]
    wrong = row(9., command=0.); wrong["command"][2] = .1
    rows.extend([wrong, row(3., command=0.), row(10.)])
    result = posture_windows(rows)
    assert [w["phase"] for w in result] == ["walk", "stop", "walk", "stop"]
    assert [w["samples"] for w in result] == [2, 2, 2, 2]
    assert posture_windows(rows, start_immediately=True)[0]["samples"] == 3


@pytest.mark.parametrize("immediate", [False, True])
def test_complete_recording_and_missing_duplicate_or_reordered_samples(immediate):
    rows = []
    for k in range(1000):
        walk = ((0 if immediate else 2) <= k * .02 < 8) or 11 <= k * .02 < 17
        rows.append(row((k + 1) * .02 + 1e-11, command=.05 if walk else 0.))
    result = posture_windows(rows, start_immediately=immediate)
    assert [w["samples"] for w in result] == [400 if immediate else 300, 150, 300, 150]
    assert all(w["complete"] and w["invalid_samples"] == 0 for w in result)
    for damaged in (rows[:450] + rows[451:], rows[:450] + rows[451:452] + rows[451:],
                    rows[:450] + [rows[451], rows[450]] + rows[452:]):
        stop = posture_windows(damaged, start_immediately=immediate)[1]
        assert not stop["complete"] and stop["mean_tilt_deg"] == pytest.approx(6.)


@pytest.mark.parametrize("field,value", [
    ("tilt_rad", None), ("tilt_rad", -.1), ("tilt_rad", math.pi + .1),
    ("tilt_rad", [0.]), ("tilt_rad", "bad"), ("tilt_rad", 10 ** 400),
    ("heading_error_rad", float("nan")), ("heading_error_rad", float("inf")),
    ("heading_error_rad", -math.pi - .1), ("heading_error_rad", math.pi + .1),
])
def test_invalid_sample_is_counted_without_silently_dropping_it(field, value):
    bad = row(2.02); bad[field] = value
    result = posture_windows([bad, row(2.04)])[0]
    assert result["samples"] == 2 and result["invalid_samples"] == 1
    for key in ("mean_tilt_deg", "sampled_max_tilt_deg", "max_abs_heading_error_deg",
                "final_heading_error_deg"):
        assert result[key] is None
    json.dumps(result, allow_nan=False)


def test_empty_missing_angles_and_partial_failed_step():
    assert all(w["samples"] == 0 and w["sampled_max_tilt_deg"] is None
               and not w["complete"] for w in posture_windows([]))
    missing = row(2.02); del missing["heading_error_rad"]
    assert posture_windows([missing])[0]["invalid_samples"] == 1
    partial = posture_windows([row(8.003, 180., -180., command=0.)])[1]
    assert partial["observed_until_s"] == 8.003 and not partial["complete"]
    assert partial["sampled_max_tilt_deg"] == 180.
    assert partial["final_heading_error_deg"] == -180.
