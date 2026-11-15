"""Walking-speed diagnostics from recorded post-control-step body velocity."""

import numpy as np


def walking_speed_windows(rows, *, start_immediately=False):
    """Compare signed body-forward speed with each row's forward command.

    Means weight the recorded 50 Hz samples equally, including a partial final
    control step after failure. They are not integrated world displacement.
    Keep missing/invalid evidence explicit; do not change motion qualification.
    """
    result = []
    for start, end in ((0. if start_immediately else 2., 8.), (11., 17.)):
        window = [r for r in rows if start + 1e-7 < r["t"] <= end + 1e-7
                  and r["command"][0] != 0.]
        expected_times = start + np.arange(1, round((end - start) / .02) + 1) * .02
        times = np.asarray([r["t"] for r in window])
        entry = {
            "command_s": start, "window_end_s": end, "samples": len(window),
            "observed_from_s": window[0]["t"] if window else None,
            "observed_until_s": window[-1]["t"] if window else None,
            "complete": bool(len(times) == len(expected_times)
                             and np.allclose(times, expected_times, atol=1e-7, rtol=0)),
            "invalid_samples": 0, "mean_command_m_s": None,
            "mean_forward_speed_m_s": None, "mean_error_m_s": None,
            "mae_m_s": None, "rmse_m_s": None,
        }
        if window:
            command = np.asarray([r["command"][0] for r in window])
            speed = np.asarray([r["speed_m_s"] for r in window])
            valid = np.isfinite(command) & np.isfinite(speed)
            entry["invalid_samples"] = int(np.count_nonzero(~valid))
            if valid.all():
                error = speed - command
                entry.update(mean_command_m_s=float(command.mean()),
                             mean_forward_speed_m_s=float(speed.mean()),
                             mean_error_m_s=float(error.mean()),
                             mae_m_s=float(np.abs(error).mean()),
                             rmse_m_s=float(np.sqrt(np.square(error).mean())))
        result.append(entry)
    return result
