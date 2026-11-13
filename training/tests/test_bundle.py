import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from whalechan_training.bundle import FILES, load_model, seal, verify


def test_bundle_covers_each_runtime_artifact(tmp_path):
    for name in FILES:
        (tmp_path / name).write_bytes(name.encode())
    seal(tmp_path)
    assert verify(tmp_path)["sealed"]
    (tmp_path / "policy.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)


def test_omitting_an_artifact_cannot_bypass_verification(tmp_path):
    with pytest.raises(ValueError, match="Export"):
        verify(tmp_path)
    for name in FILES:
        (tmp_path / name).write_bytes(name.encode())
    seal(tmp_path)
    path = tmp_path / "bundle-manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["policy.onnx"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete"):
        verify(tmp_path)


@pytest.mark.parametrize("directory", ["policy", "鲸鱼 模型"])
def test_model_loading_preserves_compiled_options_and_files(tmp_path, directory):
    path = tmp_path / directory
    path.mkdir()
    xml = '<mujoco><option timestep=".001"/><worldbody><geom size=".1"/></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = .003
    model.opt.gravity[:] = [0., 0., -1.23]
    data = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=data)
    (path / "model.mjb").write_bytes(data.tobytes())
    (path / "model.xml").write_text(xml)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    cwd = Path.cwd()
    loaded = load_model(path)
    assert loaded.opt.timestep == .003
    np.testing.assert_array_equal(loaded.opt.gravity, [0., 0., -1.23])
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert Path.cwd() == cwd


@pytest.mark.parametrize("corrupt", [False, True])
def test_model_loading_does_not_fall_back_to_xml(tmp_path, corrupt):
    (tmp_path / "model.xml").write_text('<mujoco/>')
    if corrupt:
        (tmp_path / "model.mjb").write_bytes(b"not a model")
    with pytest.raises(ValueError if corrupt else FileNotFoundError):
        load_model(tmp_path)
