import json
import sys

import mujoco
import onnx
import pytest
from onnx import TensorProto, helper

from whalechan_training.bundle import seal
from whalechan_training.inspect_policy import OBSERVATION_ORDER, inspect_policy, main


def make_bundle(path, *, change=None, input_width=15):
    path.mkdir()
    xml = '''<mujoco><compiler angle="radian"/><worldbody>
      <body name="robot/one"><joint name="robot/one" axis="1 0 0" range="-1 1"/>
        <geom size=".05" mass=".1"/></body>
      <body name="robot/two" pos="1 0 0"><joint name="robot/two" axis="0 1 0"/>
        <geom size=".05" mass=".2"/></body></worldbody><actuator>
      <position name="motor-two" joint="robot/two" kp="4"/>
      <position name="motor-one" joint="robot/one" kp="3"/>
      </actuator></mujoco>'''
    (path / "model.xml").write_text(xml)
    mujoco.mj_saveModel(mujoco.MjModel.from_xml_string(xml), str(path / "model.mjb"))
    c = dict(joints=["one", "two"], action_joint_names=["two", "one"],
             observation_order=OBSERVATION_ORDER, joint_home=[.1, .2],
             action_offset=[.3, .4], action_scale=[.05, .1], kp=[3., 4.],
             kd=[.1, .2], caps=[.5, .6], policy_dt=.02, delay_steps=2,
             actuator_model="uncalibrated test fixture")
    c.update(change or {})
    (path / "contract.json").write_text(json.dumps(c))
    node = helper.make_node("Constant", [], ["actions"],
                            value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 2], [0, 0]))
    graph = helper.make_graph([node], "fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, input_width])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save(model, path / "policy.onnx")
    seal(path)
    return path


def test_distinct_joint_action_and_motor_orders(tmp_path):
    report = inspect_policy(make_bundle(tmp_path / "policy"))
    two, one = report["joints_in_actuator_order"]
    assert (two["joint"], two["action_index"], two["joint_observation_index"]) == ("two", 0, 1)
    assert (two["position_observation_index"], two["velocity_observation_index"],
            two["previous_action_observation_index"]) == (7, 9, 10)
    assert (one["actuator_index"], one["action_index"], one["joint_observation_index"]) == (1, 1, 0)
    assert one["action_offset_rad"] == .4 and one["observation_home_rad"] == .1
    assert two["estimated_kp_nm_per_rad"] == 4.
    assert one["joint_range_rad"] == [-1, 1] and two["joint_range_rad"] is None
    assert two["axis_in_body_frame"] == [0, 1, 0]
    assert report["estimated_model_mass_kg"] == pytest.approx(.3)
    assert report["observation_slices"][-1]["start"] == 12
    assert report["hardware_calibrated"] is False


@pytest.mark.parametrize("change,match", [
    ({"action_joint_names": ["one", "one"]}, "one-to-one"),
    ({"joints": ["one", "one"]}, "one-to-one"),
    ({"observation_order": list(reversed(OBSERVATION_ORDER))}, "observation layout"),
    ({"action_scale": [1.]}, "contract vector"),
    ({"joint_home": [float("nan"), 0.]}, "contract vector"),
])
def test_invalid_contract_is_not_reported_as_wiring(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        inspect_policy(make_bundle(tmp_path / "policy", change=change))


def test_different_policy_observations_need_matching_inspector(tmp_path):
    with pytest.raises(ValueError, match="Policy shape"):
        inspect_policy(make_bundle(tmp_path / "policy", input_width=21))


def test_tampered_bundle_is_rejected(tmp_path):
    path = make_bundle(tmp_path / "policy")
    (path / "contract.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_policy(path)


def test_cli_preserves_prior_output(tmp_path, monkeypatch):
    path = make_bundle(tmp_path / "policy")
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["inspect-policy", str(path), "--out", str(out)])
    main()
    saved = out.read_bytes()
    assert json.loads(saved)["output_shape"] == [1, 2]
    with pytest.raises(FileExistsError):
        main()
    assert out.read_bytes() == saved
