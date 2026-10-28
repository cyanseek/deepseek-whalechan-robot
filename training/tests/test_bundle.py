import json
import pytest
from whalechan_training.bundle import FILES, seal, verify


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
