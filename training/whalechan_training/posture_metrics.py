"""Recorded body tilt and heading by walk/stop period, not gait qualification."""

import numpy as np


def posture_windows(rows, *, start_immediately=False):
    """Use the evaluator's 50 Hz tilt and signed initial-heading error.

    Keep partial records without interpolation. Heading remains wrapped to
    [-pi, pi]; it is not accumulated turning or a per-window heading reset.
    These sampled peaks cannot replace the evaluator's physics-step tilt peak.
    """
    result = []
    periods = (("walk", 0. if start_immediately else 2., 8.),
               ("stop", 8., 11.), ("walk", 11., 17.), ("stop", 17., 20.))
    for phase, start, end in periods:
        window = [r for r in rows if start + 1e-7 < r["t"] <= end + 1e-7
                  and (r["command"][0] != 0. if phase == "walk"
                       else all(value == 0. for value in r["command"]))]
        times = np.asarray([r["t"] for r in window])
        expected = start + np.arange(1, round((end - start) / .02) + 1) * .02
        entry = {
            "phase": phase, "command_s": start, "window_end_s": end,
            "samples": len(window),
            "observed_from_s": window[0]["t"] if window else None,
            "observed_until_s": window[-1]["t"] if window else None,
            "complete": bool(len(times) == len(expected)
                             and np.allclose(times, expected, atol=1e-7, rtol=0)),
            "invalid_samples": 0, "mean_tilt_deg": None,
            "sampled_max_tilt_deg": None, "max_abs_heading_error_deg": None,
            "final_heading_error_deg": None,
        }
        angles = []
        for row in window:
            try:
                pair = np.asarray([row.get("tilt_rad"), row.get("heading_error_rad")], dtype=float)
                valid = (pair.shape == (2,) and np.isfinite(pair).all()
                         and 0 <= pair[0] <= np.pi and abs(pair[1]) <= np.pi)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if valid:
                angles.append(pair)
            else:
                entry["invalid_samples"] += 1
        if window and not entry["invalid_samples"]:
            tilt, heading = np.degrees(angles).T
            entry.update(mean_tilt_deg=float(tilt.mean()),
                         sampled_max_tilt_deg=float(tilt.max()),
                         max_abs_heading_error_deg=float(np.abs(heading).max()),
                         final_heading_error_deg=float(heading[-1]))
        result.append(entry)
    return result
