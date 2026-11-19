from copy import deepcopy
import json

import pytest

from whalechan_training.support_metrics import support_windows


def row(t, forces, command=.05):
    return {"t": t, "normal_force": forces, "command": [command, 0., 0.]}


def test_asymmetric_loads_and_contact_threshold_are_not_step_counts():
    rows = [row(2.02, [.31, 0.]), row(2.04, [.3, .31]),
            row(2.06, [2., 6.]), row(2.08, [0., 0.])]
    original = deepcopy(rows)
    result = support_windows(rows)[0]
    assert result["support_sample_counts"] == dict(left_only=1, right_only=1, both=1, neither=1)
    assert result["support_sample_fractions"] == dict(left_only=.25, right_only=.25, both=.25, neither=.25)
    assert result["mean_normal_force_n"] == pytest.approx([2.61 / 4, 6.31 / 4])
    assert result["left_normal_force_share"] == pytest.approx(2.61 / 8.92)
    assert result["loaded_threshold_n"] == .3 and not result["complete"]
    assert rows == original


def test_window_boundaries_follow_post_step_commands():
    rows = [row(.02, [10., 0.]), row(2., [99., 99.], 0.), row(2.02, [1., 0.]),
            row(8., [1., 0.]), row(8.02, [2., 2.], 0.), row(11., [2., 2.], 0.),
            row(11.02, [0., 3.]), row(17., [0., 3.]), row(17.02, [4., 4.], 0.),
            row(20., [4., 4.], 0.)]
    result = support_windows(rows)
    assert [w["phase"] for w in result] == ["walk", "stop", "walk", "stop"]
    assert [w["samples"] for w in result] == [2, 2, 2, 2]
    assert [w["mean_normal_force_n"] for w in result] == [[1., 0.], [2., 2.], [0., 3.], [4., 4.]]
    immediate = support_windows(rows, start_immediately=True)[0]
    assert immediate["command_s"] == 0. and immediate["samples"] == 3


@pytest.mark.parametrize("immediate", [False, True])
def test_full_recording_roundoff_and_missing_sample(immediate):
    rows = []
    for k in range(1000):
        walk = ((0 if immediate else 2) <= k * .02 < 8) or 11 <= k * .02 < 17
        rows.append(row((k + 1) * .02 + 1e-11, [3., 3.], .05 if walk else 0.))
    result = support_windows(rows, start_immediately=immediate)
    assert all(w["complete"] and w["invalid_samples"] == 0 for w in result)
    assert [w["samples"] for w in result] == [400 if immediate else 300, 150, 300, 150]
    del rows[450]
    stop = support_windows(rows, start_immediately=immediate)[1]
    assert not stop["complete"] and stop["samples"] == 149
    assert stop["support_sample_counts"]["both"] == 149


@pytest.mark.parametrize("forces", [[float("nan"), 1.], [1., float("inf")],
                                    [-.1, 1.], [1.], [[1., 2.]], None])
def test_bad_force_sample_does_not_disappear_from_statistics(forces):
    result = support_windows([row(2.02, forces), row(2.04, [1., 1.])])[0]
    assert result["samples"] == 2 and result["invalid_samples"] == 1
    for key in ("support_sample_counts", "support_sample_fractions",
                "mean_normal_force_n", "left_normal_force_share"):
        assert result[key] is None
    json.dumps(result, allow_nan=False)


def test_empty_and_zero_load_are_distinct_and_no_contact_is_not_flight():
    empty = support_windows([])
    assert all(w["samples"] == 0 and not w["complete"]
               and w["support_sample_counts"] is None for w in empty)
    assert support_windows([row(2.02, [1., 1.], 0.), row(8.02, [1., 1.])]) == empty
    zero = support_windows([row(2.02, [0., 0.])])[0]
    assert zero["support_sample_fractions"]["neither"] == 1.
    assert zero["mean_normal_force_n"] == [0., 0.]
    assert zero["left_normal_force_share"] is None


def test_partial_failure_keeps_observed_support_without_claiming_full_window():
    result = support_windows([row(8.003, [1., 3.], 0.)])[1]
    assert result["samples"] == 1 and not result["complete"]
    assert result["observed_from_s"] == result["observed_until_s"] == 8.003
    assert result["mean_normal_force_n"] == [1., 3.]
    assert result["left_normal_force_share"] == .25


def test_force_share_uses_total_load_not_mean_of_per_sample_shares():
    result = support_windows([row(2.02, [1., 0.]), row(2.04, [0., 9.])])[0]
    assert result["support_sample_fractions"]["left_only"] == .5
    assert result["left_normal_force_share"] == pytest.approx(.1)


def test_large_finite_forces_stay_json_finite():
    result = support_windows([row(2.02, [1e308, 1e308]), row(2.04, [1e308, 1e308])])[0]
    assert result["mean_normal_force_n"] == [1e308, 1e308]
    assert result["left_normal_force_share"] == .5
    json.dumps(result, allow_nan=False)
