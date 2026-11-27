"""Analyze a saved native trajectory without loading a policy or simulating."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .speed_metrics import walking_speed_windows
from .stop_metrics import stop_windows
from .support_metrics import support_windows
from .posture_metrics import posture_windows


def _rows(data):
    if not isinstance(data, list):
        raise ValueError("trajectory must be a JSON array of post-step states")
    rows = []
    for index, row in enumerate(data):
        try:
            # Copy only diagnostic fields; actor/joint layouts are irrelevant.
            scalars = {key: float(row[key]) for key in
                       ("t", "speed_m_s", "horizontal_speed_m_s", "angular_speed_rad_s")}
            command = np.asarray(row["command"], dtype=float)
            position = np.asarray(row["qpos"], dtype=float)
            if command.shape != (3,) or position.ndim != 1 or position.size < 2:
                raise ValueError("command needs three values and qpos needs root XY")
            if (not np.isfinite(command).all() or not np.isfinite(position[:2]).all()
                    or not all(np.isfinite(scalars[key]) for key in
                               ("t", "horizontal_speed_m_s", "angular_speed_rad_s"))):
                raise ValueError("time, command, root XY and stop speeds must be finite")
            rows.append({**scalars, "command": command.tolist(), "qpos": position[:2].tolist(),
                         "normal_force": row.get("normal_force"),
                         "tilt_rad": row.get("tilt_rad"),
                         "heading_error_rad": row.get("heading_error_rad")})
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"trajectory row {index}: {error}") from error
    return rows


def analyze_trajectory(path, *, start_immediately=False):
    """Return diagnostics for the evaluator's fixed walk/stop schedule.

    Preserve sample order/gaps. Invalid speed/force samples keep the individual
    metric's invalid_samples/null behavior; no policy qualification is inferred.
    """
    return _analyze(Path(path).read_bytes(), start_immediately=start_immediately)[0]


def _analyze(raw, *, start_immediately=False):
    rows = _rows(json.loads(raw.decode("utf-8-sig")))
    report = {
        "schema": "whalechan-trajectory-diagnostics-v1",
        "analysis": "offline_recorded_trajectory",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_samples": len(rows),
        "initial_stand_s": 0 if start_immediately else 2,
        "stop_windows": stop_windows(rows),
        "walking_speed_windows": walking_speed_windows(rows, start_immediately=start_immediately),
        "support_windows": support_windows(rows, start_immediately=start_immediately),
        "posture_windows": posture_windows(rows, start_immediately=start_immediately),
    }
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new JSON report path")
    parser.add_argument("--html", type=Path, help="also save a new self-contained HTML chart")
    parser.add_argument("--start-immediately", action="store_true",
                        help="recording used 0-8 s first walk instead of 2-8 s")
    args = parser.parse_args()
    try:
        report, rows = _analyze(args.trajectory.read_bytes(), start_immediately=args.start_immediately)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.html:
            from .trajectory_html import trajectory_html
            if args.html.resolve() == args.out.resolve():
                raise ValueError("JSON and HTML output paths must differ")
            html = trajectory_html(report, rows, source_name=args.trajectory.name)
            for path in (args.out, args.html):
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            with args.html.open("x", encoding="utf-8") as stream:
                stream.write(html)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    print(f"Analyzed {report['recorded_samples']} recorded samples; saved {args.out}")
    if args.html:
        print(f"Saved offline charts: {args.html}")


if __name__ == "__main__":
    main()
