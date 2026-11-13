"""Bind deployment artifacts and require the recorded native-engine version."""

import hashlib, json, platform
from pathlib import Path
import mujoco

FILES = ("model.mjb", "model.xml", "contract.json", "policy.onnx")


def load_model(path):
    """Read a previously verified bundle's MJB, including Unicode disk paths.

    MuJoCo opens an ASCII virtual filename; Python handles the actual disk path.
    Load the compiled bytes directly so binary-only options remain intact.
    """
    return mujoco.MjModel.from_binary_path(
        "model.mjb", assets={"model.mjb": (Path(path) / "model.mjb").read_bytes()}
    )


def seal(path):
    p = Path(path)
    target = p / "bundle-manifest.json"
    if target.exists():
        raise FileExistsError(target)
    data = {
        "schema": "whalechan-mjlab-policy-v1",
        "mujoco": mujoco.__version__,
        "python": platform.python_version(),
        "files": {n: hashlib.sha256((p / n).read_bytes()).hexdigest() for n in FILES},
        "primary_physics_artifact": "model.mjb",
        "model_xml_role": "scene construction; compiled runtime options are in MJB",
        "initialization": "canonical home; policy at first tick",
        "fallback": None,
    }
    target.write_text(json.dumps(data, indent=2))
    return data


def verify(path):
    p = Path(path)
    manifest = p / "bundle-manifest.json"
    if not manifest.exists():
        raise ValueError(
            "Export the checkpoint with whalechan-export before evaluation"
        )
    data = json.loads(manifest.read_text())
    if data.get("schema") != "whalechan-mjlab-policy-v1" or set(
        data.get("files", {})
    ) != set(FILES):
        raise ValueError("Incomplete policy bundle manifest")
    if data["mujoco"] != mujoco.__version__:
        raise RuntimeError("Use the bundle MuJoCo version: " + data["mujoco"])
    for name, digest in data["files"].items():
        if hashlib.sha256((p / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Bundle hash mismatch: " + name)
    return {"sealed": True, "schema": data["schema"]}
