from copy import deepcopy

import pytest

from whalechan_training.stop_metrics import stop_windows


def row(t, x=0., y=0., command=(0., 0., 0.), speed=.01):
    return {"t": t, "qpos": [x, y, .3], "command": list(command),
            "horizontal_speed_m_s": speed, "angular_speed_rad_s": .1}


def test_drift_uses_last_walk_state_and_world_horizontal_distance():
    rows = [row(8, 1, 2, (.05, 0, 0)), row(8.02, 1.03, 2.04),
            row(10, 1.06, 2.08), row(11, 1.03, 2.04, speed=.003),
            row(11.02, 50, 50, (.05, 0, 0))]
    original = deepcopy(rows)
    first, missing = stop_windows(rows)
    assert not first["complete"] and first["samples"] == 3
    assert first["net_drift_m"] == pytest.approx(.05)
    assert first["peak_drift_m"] == pytest.approx(.1)
    assert first["end_horizontal_speed_m_s"] == .003
    assert first["end_angular_speed_rad_s"] == .1
    assert missing["samples"] == 0 and not missing["complete"]
    assert missing["net_drift_m"] is None
    assert rows == original


@pytest.mark.parametrize("end", [17.3, 20.0])
def test_sparse_second_stop_is_incomplete_even_when_it_reaches_the_end(end):
    result = stop_windows([row(17, 1, command=(.05, 0, 0)), row(17.02, 1.01), row(end, 1.02)])[1]
    assert not result["complete"]
    assert result["observed_until_s"] == end
    assert result["net_drift_m"] == pytest.approx(.02)


def test_missing_boundary_does_not_invent_drift():
    result = stop_windows([row(7.98, 2), row(8.02, 3), row(11, 4)])[0]
    assert not result["complete"]
    assert result["net_drift_m"] is None and result["peak_drift_m"] is None
    assert result["end_horizontal_speed_m_s"] == .01


def test_initial_standing_and_empty_trajectory_are_not_stops():
    assert stop_windows([row(1), row(2)]) == stop_windows([])
    assert all(r["net_drift_m"] is None and r["observed_until_s"] is None for r in stop_windows([]))


def test_roundoff_at_boundary_and_window_end():
    result = stop_windows([row(8 + 1e-11, command=(.05, 0, 0)),
                           row(8.02, .01), row(11 - 1e-11, .02)])[0]
    assert not result["complete"] and result["samples"] == 2
    assert result["net_drift_m"] == pytest.approx(.02)


def complete_window(start):
    return [row(start, command=(.05, 0., 0.))] + [
        row(start + tick * .02, x=tick * .001) for tick in range(1, 151)]


def test_both_complete_windows_tolerate_physics_clock_roundoff():
    rows = complete_window(8.) + complete_window(17.)
    for sample in rows:
        sample["t"] += 1e-11
    for result in stop_windows(rows):
        assert result["complete"] and result["samples"] == 150
        assert result["net_drift_m"] == pytest.approx(.15)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "off_schedule", "walking"])
def test_reaching_stop_end_does_not_hide_missing_or_invalid_sampling(change):
    rows = complete_window(8.)
    if change == "missing":
        del rows[40]
    elif change == "duplicate":
        rows[40] = deepcopy(rows[39])
    elif change == "reordered":
        rows[40], rows[41] = rows[41], rows[40]
    elif change == "off_schedule":
        rows[40]["t"] += .005
    else:
        rows[40]["command"] = [.05, 0., 0.]
    result = stop_windows(rows)[0]
    assert not result["complete"]
    assert result["observed_until_s"] == 11.
    # Partial measurements stay useful; no interpolation or invented samples.
    assert result["net_drift_m"] == pytest.approx(.15)
    assert result["peak_drift_m"] == pytest.approx(.15)
