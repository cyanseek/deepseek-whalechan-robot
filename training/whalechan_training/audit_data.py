"""Check retained online samples and report actual command coverage."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(run: Path) -> dict:
    stats = json.loads((run / "data-summary.json").read_text())
    errors = []
    with np.load(run / "on-policy-samples.npz", allow_pickle=False) as data:
        for name in data.files:
            if not np.isfinite(data[name]).all():
                errors.append("nonfinite:" + name)
        for name, size in (("actor", 78), ("critic", 81), ("action_applied", 23)):
            if data[name].shape[-1] != size:
                errors.append("dimension:" + name)
        command = data["actor"][..., -3:]
        walking = command[..., 0] > 0.005
        speeds = command[..., 0][walking]
        if np.any(command[..., 1:] != 0):
            errors.append("non-forward command")
        if np.any((speeds < 0.025) | (speeds > 0.08)):
            errors.append("command range")
        if not np.allclose(
            np.clip(data["action_sampled"], -3, 3), data["action_applied"]
        ):
            errors.append("action clipping mismatch")
        if stats["command_observation_mismatch"] or stats["nonfinite_next_actor"]:
            errors.append("whole-run observation integrity")
        result = {
            "sampled_transitions": int(np.prod(data["actor"].shape[:-1])),
            "total_transitions": int(stats["transitions"]),
            "sampled_zero_fraction": float((~walking).mean()),
            "whole_run_zero_fraction": stats["zero_command"] / stats["transitions"],
            "sampled_forward_range": [float(speeds.min()), float(speeds.max())]
            if speeds.size
            else None,
            "errors": errors,
            "passed": not errors,
            "post_done_next_observation": "reset observation, not terminal state",
        }
    coverage = run / "command-coverage.json"
    if coverage.exists():
        result["command_coverage"] = json.loads(coverage.read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
