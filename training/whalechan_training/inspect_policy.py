"""Inspect a sealed policy's joint wiring without inference or hardware access."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .bundle import verify

OBSERVATION_ORDER = [
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"
]


def inspect_policy(run: Path) -> dict:
    run = Path(run)
    verify(run)
    c = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    m = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    names, actions = c["joints"], c["action_joint_names"]
    n = len(names)
    if (len(set(names)) != n or len(actions) != n or set(actions) != set(names)
            or m.nu != n):
        raise ValueError("Joint, action and actuator names must map one-to-one")
    if c["observation_order"] != OBSERVATION_ORDER:
        raise ValueError("Unsupported observation layout; use the matching runtime")
    for field in ("joint_home", "action_offset", "action_scale", "kp", "kd", "caps"):
        values = np.asarray(c[field], dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError("Invalid contract vector: " + field)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(run / "policy.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].shape != [1, 9 + 3 * n]
            or len(outputs) != 1 or outputs[0].shape != [1, n]):
        raise ValueError("Policy shape does not match the recorded observation/action layout")

    rows = []
    for motor in range(m.nu):
        if m.actuator_trntype[motor] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Only direct hinge-joint transmissions are supported")
        joint = int(m.actuator_trnid[motor, 0])
        full_name = m.joint(joint).name
        name = full_name.removeprefix("robot/")
        if (name not in names or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                or not np.array_equal(m.actuator_gear[motor], [1, 0, 0, 0, 0, 0])):
            raise ValueError("Expected a named hinge with unit transmission gear")
        j, a = names.index(name), actions.index(name)
        rows.append({
            "joint": name,
            "actuator": m.actuator(motor).name,
            "actuator_index": motor,
            "action_index": a,
            "joint_observation_index": j,
            "position_observation_index": 6 + j,
            "velocity_observation_index": 6 + n + j,
            "previous_action_observation_index": 6 + 2 * n + a,
            "qpos_index": int(m.jnt_qposadr[joint]),
            "qvel_index": int(m.jnt_dofadr[joint]),
            "body": m.body(int(m.jnt_bodyid[joint])).name,
            "axis_in_body_frame": m.jnt_axis[joint].tolist(),
            "joint_range_rad": m.jnt_range[joint].tolist() if m.jnt_limited[joint] else None,
            "observation_home_rad": c["joint_home"][j],
            "action_offset_rad": c["action_offset"][a],
            "action_scale_rad": c["action_scale"][a],
            "estimated_kp_nm_per_rad": c["kp"][j],
            "estimated_kd_nm_s_per_rad": c["kd"][j],
            "estimated_torque_cap_nm": c["caps"][j],
        })
    if {row["joint"] for row in rows} != set(names):
        raise ValueError("Actuators do not cover each joint exactly once")

    widths = [3, 3, n, n, n, 3]
    sources = [
        "body-frame gyro", "body-frame gravity estimate (unit vector)",
        "encoder radians minus joint_home", "joint velocity rad/s",
        "previous clipped policy action in action order (software)",
        "forward/lateral m/s and yaw rad/s (software request)",
    ]
    start, layout = 0, []
    for name, width, source in zip(OBSERVATION_ORDER, widths, sources):
        layout.append(dict(name=name, start=start, stop=start + width, source=source))
        start += width
    return {
        "schema": "whalechan-policy-interface-v1",
        "files": json.loads((run / "bundle-manifest.json").read_text())["files"],
        "mujoco": mujoco.__version__,
        "estimated_model_mass_kg": float(m.body_mass.sum()),
        "model_timestep_s": float(m.opt.timestep),
        "recorded_policy_dt_s": c["policy_dt"],
        "recorded_delay_steps": c["delay_steps"],
        "recorded_actuator_model": c["actuator_model"],
        "hardware_calibrated": False,
        "input_shape": inputs[0].shape,
        "output_shape": outputs[0].shape,
        "observation_slices": layout,
        "action_formula": "target_rad = action_offset_rad + action_scale_rad * clip(action, -3, 3)",
        "index_convention": "zero-based; observation slice stop is exclusive",
        "joints_in_actuator_order": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_policy(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(report['joints_in_actuator_order'])} joint mappings to {args.out}")


if __name__ == "__main__":
    main()
