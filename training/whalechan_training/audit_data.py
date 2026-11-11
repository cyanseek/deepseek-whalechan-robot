"""Check retained online samples and report actual command coverage."""

import argparse
import json
from pathlib import Path

import numpy as np


def _dimensions(run: Path) -> tuple[dict, str]:
    """Use recorded widths; older runs predate explicit dimension metadata."""
    legacy = {"actor": 78, "critic": 81, "action": 23}
    path = run / "contract.json"
    if not path.exists():
        return legacy, "legacy 78/81/23"
    contract = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(contract, dict):
        raise ValueError("contract.json must contain an object")
    explicit = "actor_dim" in contract or "critic_dim" in contract
    order = contract.get("observation_order")
    if explicit or order is not None:
        if not isinstance(order, list) or not order or order[-1] != "command":
            raise ValueError("contract observation_order must end with command")
    if not explicit:
        return legacy, "legacy 78/81/23"
    actor, critic = contract.get("actor_dim"), contract.get("critic_dim")
    if type(actor) is not int or actor < 3 or type(critic) is not int or critic < 1:
        raise ValueError("contract actor_dim and critic_dim must be valid integer widths")
    names = contract.get("action_joint_names")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("contract action_joint_names must be nonempty and unique")
    return {"actor": actor, "critic": critic, "action": len(names)}, "contract.json"


def audit(run: Path) -> dict:
    dimensions, source = _dimensions(run)
    stats = json.loads((run / "data-summary.json").read_text(encoding="utf8"))
    errors = []
    with np.load(run / "on-policy-samples.npz", allow_pickle=False) as data:
        for name in data.files:
            if not np.isfinite(data[name]).all():
                errors.append("nonfinite:" + name)
        for name, size in (("actor", dimensions["actor"]),
                           ("critic", dimensions["critic"]),
                           ("action_sampled", dimensions["action"]),
                           ("action_applied", dimensions["action"])):
            if data[name].shape[-1] != size:
                errors.append("dimension:" + name)
            if data[name].shape[:-1] != data["actor"].shape[:-1]:
                errors.append("sample shape:" + name)
        command = data["actor"][..., -3:]
        standing = (command == 0).all(axis=-1)
        forward_only = (command[..., 1:] == 0).all(axis=-1)
        in_range = (command[..., 0] >= 0.025) & (command[..., 0] <= 0.08)
        walking = forward_only & in_range
        speeds = command[..., 0][walking]
        if np.any(~forward_only):
            errors.append("non-forward command")
        if np.any((command[..., 0] != 0) & ~in_range):
            errors.append("command range")
        if data["action_sampled"].shape != data["action_applied"].shape or not np.allclose(
            np.clip(data["action_sampled"], -3, 3), data["action_applied"]
        ):
            errors.append("action clipping mismatch")
        if stats["command_observation_mismatch"] or stats["nonfinite_next_actor"]:
            errors.append("whole-run observation integrity")
        result = {
            "expected_dimensions": dimensions,
            "dimension_source": source,
            "sampled_transitions": int(np.prod(data["actor"].shape[:-1])),
            "total_transitions": int(stats["transitions"]),
            "sampled_zero_fraction": float(standing.mean()),
            "sampled_invalid_command_count": int((~(standing | walking)).sum()),
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
