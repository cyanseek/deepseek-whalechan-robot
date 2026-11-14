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

### CPU evaluation and inspection

Use a separate Python 3.12+ environment on Windows or Linux. No CUDA, Torch,
mjlab, API key or character asset download is needed for these tools.
From the repository root, on Windows PowerShell:

```powershell
py -3.13 -m venv .venv-policy
./.venv-policy/Scripts/python.exe -m pip install ./training
./.venv-policy/Scripts/python.exe -m pip check
./.venv-policy/Scripts/whalechan-inspect-policy.exe runs/policy-01 --out runs/interface-01.json
./.venv-policy/Scripts/whalechan-evaluate-suite.exe runs/policy-01 --out runs/eval-suite-01
```

On Linux:

```bash
python3.12 -m venv .venv-policy
source .venv-policy/bin/activate
python -m pip install ./training
python -m pip check
whalechan-inspect-policy runs/policy-01 --out runs/interface-01.json
whalechan-evaluate-suite runs/policy-01 --out runs/eval-suite-01
```

`runs/policy-01` is a placeholder for a complete, compatible exported bundle;
the repository does not include trained weights. Export one with the training
environment below, then copy the whole sealed bundle to the CPU computer.
These commands inspect or simulate the policy and write JSON results; they do
not launch the character viewer. `whalechan-audit-data` also runs in this CPU
environment against retained training samples. Training and checkpoint export
require the **`train` extra**; upgrading an older full setup should use
`'./training[train,test]'` instead of `'./training[test]'`.

仅检查策略接口、运行独立 CPU 仿真或审计训练样本时，安装 `./training` 即可，
无需 GPU、Torch、mjlab、API 或角色资产。上面的 `runs/policy-01` 需替换为完整、
匹配的已导出策略包；仓库不附带训练权重，可在训练机器上导出后整包复制。
输出为评估 JSON 与轨迹，不会打开角色窗口。请单独建环境，不要更改角色查看器
的 MuJoCo 3.3.7。继续训练或导出检查点须安装 `[train]`；原先完整安装命令中的
`[test]` 现在应改为 `[train,test]`。

For CPU tool development, install `'./training[test]'` and run:

```bash
python -m pytest -q training/tests --ignore=training/tests/test_data.py --ignore=training/tests/test_training_contract.py
```

The two excluded modules exercise the mjlab/Torch recorder and training task;
run the full suite in the training environment. CI checks both installations,
including the CPU commands on Windows and Linux without training dependencies.

CPU 工具测试使用 `[test]`；上面明确排除的两个模块依赖 mjlab/Torch，完整训练
环境仍运行全部测试。CI 分别验证两种安装，并在 Windows/Linux 检查 CPU 入口。

### GPU training and checkpoint export

Use Linux or WSL, Python 3.12 and a CUDA GPU. From the repository root:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e './training[train,test]'
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

Evaluation and interface inspection accept bundle directories containing Unicode
characters or spaces, including on Windows. They read the verified `model.mjb`
bytes into MuJoCo's virtual filesystem, preserving compiled physics options.
There is no need to relocate the bundle, change working directory or rebuild it
from XML. Keep the bundle's recorded MuJoCo version and file hashes matched.

评估与接口检查支持中文、空格等路径，Windows 下也可直接读取原策略包。
工具通过内存加载已校验的 MJB，保留编译后的物理参数，无需移动策略包、切换
工作目录或改用 XML 重建；仍须匹配包内记录的 MuJoCo 版本和文件哈希。

Policy actions must contain one finite value per action joint before clipping.
NaN/Inf, wrong-sized actions and nonfinite joint targets end evaluation before
applying that tick. The report records `failure.stage: policy_action`, its time
and reason, with `qualified: false`; the partial trajectory and bundle hash are
retained. Failure on the first tick produces an empty trajectory and a `null`
heading error. Finite actions still use the recorded transform and ±3 clipping.

策略动作在裁剪前须逐关节对应且数值有限。NaN/Inf、尺寸不符或目标溢出时，评估在
执行该步前结束，报告保留失败时间、原因和 `policy_action` 阶段，结果为未达标。
已有轨迹和策略哈希保留；首步失败时轨迹为空，朝向误差为 `null`。正常有限动作的
映射与 ±3 裁剪不变。

Run all three 20-second cases with a single command:

```bash
whalechan-evaluate-suite runs/policy-01 --out runs/eval-suite-01
# Also available without reinstalling an existing editable checkout:
python -m whalechan_training.evaluate_suite runs/policy-01 --out runs/eval-suite-02
```

Each case starts a separate native CPU process from the same bundle and home
state. The suite retains per-case logs, reports and full trajectories, then
collects step counts, slip, stops, drift and heading in `summary.json`. A failed
case does not skip the remaining cases. Existing output directories are refused.
The suite exits **0** only when all three reports qualify, **1** for completed
evaluations with unqualified motion, or **2** for execution/report errors.
The individual evaluator's criteria and exit behavior are unchanged.

三场景评估依次检查默认速度、0.065 m/s 和立即起步，均由真实策略从初始姿态运行，
每场计划20秒，物理失败时保留提前结束的轨迹。各场独立保存日志、轨迹和报告，
失败后仍继续其他场景；`summary.json`
集中呈现左右脚步数、足滑、停稳与漂移等结果。退出码 0 表示三场全部达标，1 表示
评估完成但动作未全部达标，2 表示执行或报告错误。请为每轮指定新的输出目录。

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

### Audit retained training samples

`whalechan-audit-data RUN --out NEW_REPORT.json` checks the saved samples against
`contract.json`. Runs with explicit `actor_dim` / `critic_dim` use those widths
and the length of `action_joint_names`; the actor's final three values must be
the command. Original runs without dimension metadata retain the 78/81/23 layout.
The report states the expected widths and their source, checks both sampled and
applied actions, and rejects sample-shape mismatches before clipping comparison.
Incomplete or invalid dimension metadata is an error, not a legacy fallback.

Command coverage, finite-value and clipping checks apply to the retained sample;
whole-run counters are reported separately. A passing data audit checks recording
integrity. Walking and stopping are evaluated with the independent CPU scenarios.
Use a new report filename to preserve previous audits. Reading an expanded data
contract does not change the policy runtime's supported observation layout.

数据检查按 `contract.json` 中明确保存的观测维度与动作关节数检查样本，保留原先无维度
字段运行的 78/81/23 兼容方式，并在报告中标明依据。actor 最后三项须为运动指令；
采样动作与执行动作都检查维度和样本形状，避免数组广播掩盖记录错位。残缺或无效的
维度定义会报错。检查针对保留样本，全程计数单列；行走和停止能力仍由独立物理评估
确认。请使用新的报告文件名；数据检查支持扩展维度不等于策略运行时自动支持新接口。

### Inspect the simulation-to-hardware interface

```bash
whalechan-inspect-policy runs/policy-01 --out runs/interface-01.json
# An existing editable installation can also use:
python -m whalechan_training.inspect_policy runs/policy-01 --out runs/interface-02.json
```

The inspector verifies the sealed bundle and reads the compiled model and ONNX
shapes on CPU, without running the policy. Its JSON maps each actuator to its
action and joint-observation indices, axis in the attached body's frame, range,
observation home, target offset/scale and estimated gains/torque cap. It also
lists observation slices and their intended sensor/software sources. All indices
are zero-based. Joint observations, previous actions and actuators have distinct
orders: do not wire them by matching array positions. The position-observation
home and action-target offset are also separate values.

Use this report when defining encoder zero/sign, IMU mounting and actuator
commands. Model axes are in local body frames; hardware sign/zero still need
measurement. The report describes the current training interface, not a servo
driver or calibrated motor specification. Position firmware gains are not
interchangeable with the model's Nm/rad gains. Encoder/IMU calibration, measured
actuator response and mass/inertia remain part of hardware bring-up. Different
observation layouts require a matching inspector/runtime. Existing reports are
preserved; choose a fresh output filename.

接口检查在CPU上读取已校验的策略包，导出每轴的动作/观测/执行器序号、局部轴线、
角度范围、观测零位、目标偏置/缩放及估计增益/力矩上限，同时列出观测分段和来源。
三种数组顺序不相同，观测零位也不等于动作偏置；对接编码器、IMU和舵机时应按名称
核对，再实测零位、方向、响应及质量惯量。工具不执行策略或连接硬件，输出不代表
已标定的电机参数；已有报告不会覆盖，其他观测布局需使用匹配的工具和运行时。

### Map named sensors and action targets

`whalechan_training.policy_io.PolicyIO` implements the current observation/target
transform using NumPy, with no device access. It takes calibrated, named joint
measurements and body-frame IMU values; dictionary insertion order does not
affect policy wiring. Use the inspector above to verify the bundle before
constructing the adapter:

```python
import json
from pathlib import Path
from whalechan_training.inspect_policy import inspect_policy
from whalechan_training.policy_io import PolicyIO, EncoderCalibration

bundle = Path("runs/policy-01")
inspect_policy(bundle)
interface = PolicyIO(json.loads((bundle / "contract.json").read_text()))

# At each policy tick, supply synchronized, calibrated measurements:
# obs = interface.observe(gyro_body=gyro_rad_s, gravity_body=gravity_unit,
#     joint_position=q_by_name, joint_velocity=qd_by_name,
#     command=[forward_m_s, 0.0, 0.0])
# action = session.run(None, {input_name: obs[None]})[0][0]
# targets_by_name = interface.targets(action)
```

Joint positions are absolute model radians; the adapter subtracts `joint_home`.
Gyro and gravity must already be expressed in the base frame, with gravity a
downward unit vector rather than raw accelerometer readings. The command is
`[forward m/s, lateral m/s, yaw rad/s]`; the current recipe trains only forward
and zero commands. `targets()` clips actions to ±3, applies the separate action
offset/scale and retains previous actions in **action order**. Call it once per
accepted policy tick. Reset history only when starting a new physical episode;
pausing does not reset it. Missing joints, nonfinite values and other observation
layouts are rejected. The current body produces 78 observation values.

`EncoderCalibration(zero_rad, direction, encoder_radians_per_joint_radian)`
converts each measured encoder position/velocity into model coordinates and a
joint target back into encoder radians. Supply measured zero, sign (±1) and
ratio; use ratio 1 for joint-side feedback. This conversion does not account for
backlash, wrapping, sensor latency or servo firmware control. The caller still
owns clock synchronization, actuator limits, matching control mode and fault
handling. No serial transport, motor calibration or hardware execution is added
by this adapter; the existing simulator remains unchanged.

命名接口接收已标定的关节角度/角速度与机身坐标系IMU，按策略约定生成观测、解码
目标并保存上一步动作。字典顺序不影响关节对应；观测零位、动作偏置与编码器零位
分别处理。`EncoderCalibration` 用实测零位、正负方向和传动比换算编码器与模型
坐标。暂停时保留动作历史，新的物理回合才重置；缺少关节、非有限值或其他观测
布局会报错。当前为78维接口，仍需驱动层完成时间同步、电机模式/限幅和故障处理，
没有连接硬件或替换既有仿真控制器。

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

The sample audit accepts exact-zero commands or forward commands within
0.025–0.08 m/s with zero lateral/yaw components. `sampled_zero_fraction`
counts only exact-zero commands; `sampled_invalid_command_count` reports
samples outside that contract. `sampled_forward_range` uses valid forward
samples. The whole-run fraction remains the recorder's standing count; the
audit does not rewrite retained data or whole-run counters.

样本审计将全零指令与有效前进指令分开统计；倒退、过小的非零速度、横移或转向
不会被算作站立。`sampled_invalid_command_count` 给出不符合当前训练指令范围的
样本数；原始样本和整轮训练计数保持不变。

## Evaluate the motion

Confirmed steps require ≥3 mm whole-sole clearance, ≥5 mm forward travel and
≥30 ms flight and loaded landing. Stops require both feet loaded, horizontal
speed <8 mm/s, angular speed <0.2 rad/s and tilt <12° for 0.5 s. The default
20-second evaluation also checks alternating steps, net displacement, slip and
heading. These are development criteria for this flat-ground body.

`report.json.stop_windows` adds diagnostics for the 8–11 s and 17–20 s stop
windows: net and peak horizontal root displacement since the stop command,
plus horizontal/angular speed at the last observed state. `complete` means
the full window was recorded, not that the robot stopped successfully. Early
termination leaves a partial window; missing boundary data gives `null` drift.
Displacements are sampled at 50 Hz in world XY, independently of walking foot
slip. These fields do not change `stops` or `qualified`.

停止窗口同时报告下令后的身体净漂移、最大漂移及结束时速度。`complete` 仅表示
该时间段记录完整；提前跌倒的窗口会标记未完成，未观测到的位移保留为 `null`。
这些诊断用于定位停止失衡，不替代停稳与整体步态验收。

The task registers as `Mjlab-Velocity-Flat-WhaleChan-VD39`. The framework provides
GPU physics and PPO; this package supplies the body/control mapping, data
instrumentation and CPU deployment evaluation. For physical hardware, measured
mass/inertia and actuator calibration are still required. See
[body notes](whalechan_training/body/README.md) and the [roadmap](../docs/ROADMAP.md).
