"""Export a trusted local RSL-RL checkpoint with its observation normalizer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .bundle import seal


def export(run: Path, checkpoint: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    actor = MLPModel(
        obs=TensorDict({"actor": torch.zeros(1, 78)}, batch_size=[1]),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=23,
        hidden_dims=(256, 128, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    model = actor.as_onnx(verbose=False).eval()
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),
        str(out / "policy.onnx"),
        opset_version=18,
        input_names=model.input_names,
        output_names=model.output_names,
        dynamo=False,
        dynamic_axes={},
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(out / "policy.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    errors = []
    with torch.inference_mode():
        for _ in range(16):
            x = torch.randn(1, 78)
            actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
            errors.append(float(np.max(np.abs(model(x).numpy() - actual))))
    if max(errors) >= 1e-4:
        raise ValueError(f"ONNX parity error: {max(errors)}")
    for name in ("contract.json", "model.mjb", "model.xml"):
        shutil.copy2(run / name, out / name)
    seal(out)
    report = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": saved["iter"],
        "max_torch_onnx_error": max(errors),
        "normalized_actor": True,
        "input_shape": [1, 78],
        "action_shape": [1, 23],
    }
    (out / "export.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.checkpoint, args.out), indent=2))


if __name__ == "__main__":
    main()
