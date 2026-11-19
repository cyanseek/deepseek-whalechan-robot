"""Sampled sole-contact diagnostics, independent of gait qualification."""

import numpy as np


def support_windows(rows, *, start_immediately=False):
    """Summarize left/right normal forces in each scheduled walk/stop period.

    Use recorded 50 Hz post-step samples, including a partial final step. Counts
    and fractions describe sampled sole loading, not contact duration, impulse,
    confirmed steps or balance. Never interpolate missing samples.
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
            "loaded_threshold_n": .3, "invalid_samples": 0,
            "support_sample_counts": None, "support_sample_fractions": None,
            "mean_normal_force_n": None, "left_normal_force_share": None,
        }
        forces = []
        for row in window:
            try:
                force = np.asarray(row.get("normal_force"), dtype=float)
                valid = force.shape == (2,) and np.isfinite(force).all() and (force >= 0).all()
            except (TypeError, ValueError):
                valid = False
            if valid:
                forces.append(force)
            else:
                entry["invalid_samples"] += 1
        if window and not entry["invalid_samples"]:
            forces = np.asarray(forces)
            left, right = (forces > entry["loaded_threshold_n"]).T
            counts = {name: int(np.count_nonzero(mask)) for name, mask in (
                ("left_only", left & ~right), ("right_only", right & ~left),
                ("both", left & right), ("neither", ~left & ~right))}
            # Scale before summation so finite large forces do not overflow.
            mean = (forces / len(window)).sum(axis=0)
            largest = float(forces.max())
            scaled = forces / largest if largest else forces
            total = float(scaled.sum())
            entry.update(
                support_sample_counts=counts,
                support_sample_fractions={name: count / len(window) for name, count in counts.items()},
                mean_normal_force_n=mean.tolist(),
                left_normal_force_share=float(scaled[:, 0].sum() / total) if total else None,
            )
        result.append(entry)
    return result
