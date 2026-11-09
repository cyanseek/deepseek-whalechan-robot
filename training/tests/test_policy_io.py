import numpy as np
import pytest

from whalechan_training.policy_io import EncoderCalibration, PolicyIO


def contract():
    return dict(joints=["left", "right"], action_joint_names=["right", "left"],
                observation_order=["base_ang_vel", "projected_gravity", "joint_pos",
                                   "joint_vel", "actions", "command"],
                joint_home=[.1, .2], action_offset=[.3, .4], action_scale=[.05, .1])


def sensors():
    return dict(gyro_body=[.1, .2, .3], gravity_body=[0., 0., -1.],
                joint_position={"right": .5, "left": .2},
                joint_velocity={"right": 2., "left": 1.}, command=[.05, 0., 0.])


def test_distinct_joint_action_orders_and_history_reset():
    io = PolicyIO(contract())
    assert io.targets([4., -2.]) == pytest.approx({"right": .45, "left": .2})
    obs = io.observe(**sensors())
    # Observation positions use joint_home, not the distinct action offset.
    np.testing.assert_array_equal(obs, np.array([
        .1, .2, .3, 0., 0., -1., .1, .3, 1., 2., 3., -2., .05, 0., 0.
    ], dtype=np.float32))
    io.reset()
    np.testing.assert_array_equal(io.observe(**sensors())[10:12], [0., 0.])


@pytest.mark.parametrize("direction", [-1, 1])
def test_encoder_zero_direction_ratio_roundtrip(direction):
    encoder = EncoderCalibration(zero_rad=.25, direction=direction,
                                 encoder_radians_per_joint_radian=2.)
    assert encoder.joint_position(.25 + direction * .8) == pytest.approx(.4)
    assert encoder.joint_velocity(direction * 1.2) == pytest.approx(.6)
    assert encoder.encoder_target(.4) == pytest.approx(.25 + direction * .8)


@pytest.mark.parametrize("change", [
    {"direction": 0}, {"zero_rad": float("nan")},
    {"encoder_radians_per_joint_radian": 0.},
])
def test_invalid_calibration(change):
    values = dict(zero_rad=0., direction=1)
    values.update(change)
    with pytest.raises(ValueError, match="calibration"):
        EncoderCalibration(**values)


@pytest.mark.parametrize("change", [
    {"joints": ["left", "left"]}, {"action_joint_names": ["right", "tail"]},
    {"observation_order": ["different"]}, {"actor_dim": 21},
    {"joint_home": [0.]}, {"action_scale": [1., float("nan")]},
])
def test_incompatible_contract(change):
    c = contract()
    c.update(change)
    with pytest.raises(ValueError):
        PolicyIO(c)


@pytest.mark.parametrize("change", [
    {"joint_position": {"left": .2}},
    {"joint_velocity": {"left": 1., "right": [2.]}},
    {"gyro_body": [0., 0., float("nan")]},
    {"gravity_body": [0., 0., -9.81]}, {"command": [0., 0.]},
    {"joint_position": {"left": 1e100, "right": .5}},
])
def test_invalid_sensor_packet(change):
    packet = sensors()
    packet.update(change)
    with pytest.raises(ValueError):
        PolicyIO(contract()).observe(**packet)


@pytest.mark.parametrize("action", [[1.], [float("nan"), 0.], [[0., 1.]]])
def test_bad_action_does_not_replace_history(action):
    io = PolicyIO(contract())
    io.targets([1., 2.])
    with pytest.raises(ValueError):
        io.targets(action)
    np.testing.assert_array_equal(io.previous, [1., 2.])


def test_encoder_rejects_nonfinite_measurements():
    encoder = EncoderCalibration(0., 1)
    for method in (encoder.joint_position, encoder.joint_velocity, encoder.encoder_target):
        with pytest.raises(ValueError):
            method(float("nan"))
