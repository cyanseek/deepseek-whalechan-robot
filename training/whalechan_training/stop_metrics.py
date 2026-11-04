"""Stop-window diagnostics from the evaluator's recorded post-step states."""
import numpy as np


def stop_windows(rows):
    """Measure the fixed 8–11 s and 17–20 s stops; never infer missing motion.

    The row at the command boundary is the last walking state. Displacements
    use world XY root positions, not signed forward travel or foot-contact slip.
    These sampled diagnostics do not replace the evaluator's stop criteria.
    """
    result = []
    for start, end in ((8.0, 11.0), (17.0, 20.0)):
        # Allow accumulated physics-time roundoff at exact schedule boundaries.
        boundary = next((r for r in rows if abs(r["t"] - start) < 1e-7), None)
        window = [r for r in rows if start + 1e-7 < r["t"] <= end + 1e-7
                  and all(value == 0 for value in r["command"])]
        entry = {"command_s": start, "window_end_s": end,
                 "samples": len(window), "observed_until_s": window[-1]["t"] if window else None,
                 "complete": bool(boundary is not None and window and abs(window[-1]["t"] - end) < 1e-7),
                 "net_drift_m": None, "peak_drift_m": None,
                 "end_horizontal_speed_m_s": None, "end_angular_speed_rad_s": None}
        if window:
            entry["end_horizontal_speed_m_s"] = window[-1]["horizontal_speed_m_s"]
            entry["end_angular_speed_rad_s"] = window[-1]["angular_speed_rad_s"]
        if boundary is not None and window:
            origin = np.asarray(boundary["qpos"][:2])
            distance = np.linalg.norm(np.asarray([r["qpos"][:2] for r in window]) - origin, axis=1)
            entry["net_drift_m"] = float(distance[-1])
            entry["peak_drift_m"] = float(distance.max())
        result.append(entry)
    return result
