import json
import sys

import mujoco
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from whalechan_training import evaluate
from whalechan_training.bundle import seal
from whalechan_training.evaluate import Runtime


def runtime():
    # The target transform only needs the recorded action mapping, not physics.
    r = Runtime.__new__(Runtime)
    r.offset = np.array([.1, -.2])
    r.scale = np.array([.05, .1])
    r.action_to_motor = [1, 0]
    r.previous = np.array([.4, -.6])
    return r


def test_finite_action_clipping_and_motor_order_are_preserved():
    r = runtime()
    np.testing.assert_allclose(r.targets([4., -2.]), [-.4, .25])
    np.testing.assert_array_equal(r.previous, [3., -2.])


@pytest.mark.parametrize("action", [
    [float("nan"), 0.], [float("inf"), 0.], [0., -float("inf")],
    .5, [.5], [[.5, .2]], [.5, .2, .1],
])
def test_invalid_policy_action_leaves_history_unchanged(action):
    r = runtime()
    with pytest.raises(ValueError, match="Policy action"):
        r.targets(action)
    np.testing.assert_array_equal(r.previous, [.4, -.6])


def test_overflowing_target_leaves_history_unchanged():
    r = runtime()
    r.scale[0] = np.finfo(float).max
    with pytest.raises(ValueError, match="joint targets"):
        r.targets([3., 0.])
    np.testing.assert_array_equal(r.previous, [.4, -.6])


@pytest.mark.parametrize("after_one_tick", [False, True])
def test_native_evaluation_retains_action_failure(tmp_path, monkeypatch, capsys, after_one_tick):
    # Small native body and a diagnostic ONNX, not a trained walking policy.
    # Relative paths also work with MuJoCo's Windows native file opener.
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "policy"
    bundle.mkdir()
    xml = '''<mujoco><option timestep=".001"/><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="robot/base" pos="0 0 .3"><freejoint/>
        <geom size=".04" mass=".1"/>
        <body pos="-.06 0 0"><joint name="robot/left"/>
          <geom name="robot/left_sole" type="box" size=".02 .03 .01" mass=".05"/></body>
        <body pos=".06 0 0"><joint name="robot/right"/>
          <geom name="robot/right_sole" type="box" size=".02 .03 .01" mass=".05"/></body>
      </body></worldbody><actuator>
      <position joint="robot/left" kp="3"/><position joint="robot/right" kp="3"/>
      </actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    mujoco.mj_saveModel(model, "policy/model.mjb")
    (bundle / "model.xml").write_text(xml)
    c = dict(joints=["left", "right"], action_joint_names=["left", "right"],
             qpos0=model.qpos0.tolist(), qpos=model.qpos0.tolist(), joint_home=[0., 0.],
             kp=[3., 3.], kd=[.1, .1], caps=[.5, .5],
             action_offset=[0., 0.], action_scale=[.1, .1])
    (bundle / "contract.json").write_text(json.dumps(c))
    tensors = [helper.make_tensor("bad", TensorProto.FLOAT, [1, 2], [float("nan"), 0.])]
    if after_one_tick:
        tensors += [helper.make_tensor("index", TensorProto.INT64, [1], [10]),
                    helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.]),
                    helper.make_tensor("good", TensorProto.FLOAT, [1, 2], [.5, .5])]
        nodes = [helper.make_node("Gather", ["obs", "index"], ["previous"], axis=1),
                 helper.make_node("Greater", ["previous", "zero"], ["later"]),
                 helper.make_node("Where", ["later", "bad", "good"], ["actions"])]
    else:
        nodes = [helper.make_node("Identity", ["bad"], ["actions"])]
    graph = helper.make_graph(nodes, "action-failure-fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 15])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])], tensors)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                                ir_version=10), bundle / "policy.onnx")
    seal(bundle)
    monkeypatch.setattr(sys, "argv", ["evaluate", "policy", "--out", "result", "--seconds", ".1"])
    evaluate.main()
    report = json.loads((tmp_path / "result/report.json").read_text())
    rows = json.loads((tmp_path / "result/trajectory.json").read_text())
    assert len(rows) == int(after_one_tick)
    assert report["sim_seconds"] == pytest.approx(.02 if after_one_tick else 0.)
    assert report["failure"]["stage"] == "policy_action" and not report["qualified"]
    assert report["failure"]["time_s"] == report["sim_seconds"]
    assert "finite" in report["failure"]["reason"]
    assert (report["max_heading_error_deg"] is None) == (not after_one_tick)
    assert len(report["policy_sha256"]) == 64
    assert json.loads(capsys.readouterr().out)["failure"] == report["failure"]
    assert all(np.isfinite(row["action"]).all() for row in rows)
