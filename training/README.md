# Train the Whale-chan body

Experimental **mjlab / RSL-RL PPO → normalized ONNX → native CPU MuJoCo**
workflow, following [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl).
The package includes our VD32 dynamics candidate, online sampling and independent
walk/stop evaluation. It runs separately from the VD24 character application.

Training data comes from the policy interacting with this body in simulation.
There is no downloaded motion dataset, Duck policy or QP teacher in this recipe.
The default character demonstration continues to use its existing controller;
learned locomotion is experimental.

## Install

Use Linux or WSL, Python 3.12 and a CUDA GPU. From the repository root:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e './training[test]'
python -m pytest -q training/tests
```

CUDA 12.8 is the tested RTX 5070 Ti setup. Match the Torch CUDA build to your
driver before installing the package; CPU inference needs no GPU. Dependencies
are separate from the character viewer's MuJoCo 3.3.7 environment. The training
body uses primitive geometry, so the visual asset ZIP is not needed.

## Run a small experiment

```bash
# Compile, sample, update, checkpoint and export: start with this small run.
whalechan-train --out runs/train-smoke-01 --envs 64 --iterations 5
whalechan-audit-data runs/train-smoke-01 --out runs/train-smoke-01/data-audit.json

# A bounded training run; use a fresh output directory for each experiment.
whalechan-train --out runs/train-01 --envs 256 --iterations 200

# Export a selected checkpoint with its model, contract and baked-in normalizer.
whalechan-export runs/train-01 runs/train-01/checkpoint-final.pt --out runs/policy-01

# Start from home and drive physics solely with the exported policy.
whalechan-evaluate runs/policy-01 --out runs/eval-01
whalechan-evaluate runs/policy-01 --speed .065 --out runs/eval-faster-01
whalechan-evaluate runs/policy-01 --start-immediately --out runs/eval-start-01
```

The native evaluator uses MuJoCo 3.10 and ONNXRuntime on CPU. Read `report.json`
and `trajectory.json`: step counts, contact slip, heading, stop latency and falls
are measured independently of training rewards. A completed command can report
`qualified: false`; inspect that field rather than treating process exit 0 as a
walking result. Evaluate intermediate checkpoints as well as the final one.

## Resume and retain experiments

```bash
whalechan-train --out runs/train-02 --envs 256 --iterations 100 \
  --resume runs/train-01/checkpoint-final.pt
```

Every 50 updates and at the end, the runner saves weights, observation
normalizers, optimizer, counters and RNG state. Creating `STOP` in the active
output directory requests exit after the next scheduled save; `SIGUSR1` does the
same. On resume, physical episodes restart. Each run retains its source, config,
compiled model, checkpoints, data samples and summary. Do not overwrite runs.

`--consolidate` is an optional low-exploration continuation: cap the loaded
initial action standard deviation at 0.2, remove the entropy bonus and use a
fixed learning rate of 1e-5. It changes optimization, not the observation or
motor interface. `--recipe fullsole` measures the minimum of the oriented sole
box for clearance/landing rewards. Recipe changes require a new experiment and
the same independent evaluation.

## Physical and data contract

| Component | Current configuration |
| --- | --- |
| Body | VD32 candidate; estimated 1.267 kg; 23 actuators; 30 body collision geoms |
| Actor | 78 values: angular velocity, gravity, 23 joint positions/velocities, previous actions, 3 command values |
| Critic | Actor inputs plus privileged body linear velocity; 81 values |
| Control | 50 Hz policy, 1 kHz implicit PD, 2 ms target delay |
| Motor model | Estimated speed-dependent torque limits; calibration remains future work |
| Data | Online interaction; seed 42; forward commands 0.025–0.08 m/s and exact-zero commands |
| Variation | Initial pose, encoder/observation noise, sole friction and base COM |
| Evaluation | 2 s stand, 6 s walk, 3 s stop, 6 s walk, 3 s stop |

Joint, action and motor orders are mapped by name. The body faces local **−Y**.
The actor has no privileged linear velocity, world position or reference gait.
The evaluator has no teacher startup, QP fallback or automatic episode reset.

`data-summary.json` counts all transitions. `command-coverage.json` separates
standing/walking exposure, termination causes, completed command segments and
reset action history. These groups use the command **before** a step; a fall
followed by a new walking command is still attributed to its original command.
Ongoing segments are censored. Termination reasons can overlap.

`on-policy-samples.npz` keeps four worlds every 200 control ticks. It is a small
provenance sample, not the full PPO rollout buffer. After `done`, `next_actor`
is the framework's reset observation. The audit checks integrity and reports
observed coverage; the configured standing probability is not a guarantee of
the fraction of surviving training transitions.

## Evaluate the motion

Confirmed steps require ≥3 mm whole-sole clearance, ≥5 mm forward travel and
≥30 ms flight and loaded landing. Stops require both feet loaded, horizontal
speed <8 mm/s, angular speed <0.2 rad/s and tilt <12° for 0.5 s. The default
20-second evaluation also checks alternating steps, net displacement, slip and
heading. These are development criteria for this flat-ground body.

The task registers as `Mjlab-Velocity-Flat-WhaleChan-VD39`. The framework provides
GPU physics and PPO; this package supplies the body/control mapping, data
instrumentation and CPU deployment evaluation. For physical hardware, measured
mass/inertia and actuator calibration are still required. See
[body notes](whalechan_training/body/README.md) and the [roadmap](../docs/ROADMAP.md).
