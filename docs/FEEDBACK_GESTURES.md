# VD29 feedback gestures / 反馈动作

Optional source profile: **nod, shake, look left, look right**, on the unchanged VD24 body and VD28 contact controller. Existing launchers and frozen versions remain available. These motions use the two existing head joints; they do not add gaze tracking, whole-body turning or a hardware driver.

可选源码入口：**点头、摇头、向左看、向右看**。沿用现有头部两个关节和 VD28 接触步态。左右以角色自身为准，动作结束会回正。主展示保持简短，扩展动作可单独体验。

## Try it / 运行

Install the matching character assets as in the repository quickstart, and install the optional contact-controller dependencies:

```powershell
./.venv-sim/Scripts/python.exe -m pip install -r requirements-contact.txt
./Start-WhaleChan-Gestures.ps1 -Backend commands -Record
```

For the existing private DeepSeek configuration:

```powershell
./Start-WhaleChan-Gestures.ps1 -Backend deepseek -Record
```

The native panel accepts text and has four new buttons. The offline vocabulary is exactly `点头`, `摇头`, `向左看`, `向右看` (English: `nod`, `shake`, `look left`, `look right`). The real API accepts paraphrases through the same validated high-level intent schema. Existing `过来`, `挥手`, `摆尾`, `停一下` remain available. Stop phrases bypass inference. Keep filled API configuration private; see [INTERACTION.md](INTERACTION.md).

原生面板可以输入文字或点击四个新按钮。离线模式使用上面的明确指令；DeepSeek 模式可用自然说法。任何新请求会取消当前任务，并等身体支撑和头部回正后再执行。API 故障不会自动冒充离线成功。

## Motion contract / 动作约束

| Action | Planned head angles | Behavior |
| --- | --- | --- |
| Nod / 点头 | pitch +0.10 / −0.06 rad (about +5.7° / −3.4°) | Two nods, return neutral |
| Shake / 摇头 | yaw +0.18 / −0.18 / +0.12 rad | Alternating turns, return neutral |
| Look left/right / 左右看 | yaw ±0.22 rad (about 12.6°) | Turn, briefly hold, return neutral |

Targets use quintic transitions followed by a **0.45 rad/s velocity and 1.8 rad/s² acceleration limiter**. These are command limits; physical measured velocity can differ under contact and gravity. The model's original joint, actuator-force, mass and collision limits remain unchanged.

The controller waits for double support and low body velocity. Completion additionally requires commanded head position within 0.004 rad, command velocity below 0.025 rad/s, measured head position within 0.018 rad (about 1°), and measured velocity below 0.05 rad/s, continuously for 0.15 s. The measured position tolerance accommodates the existing gravity-loaded position actuators. Cancellation returns smoothly to neutral and uses the same check. An 8-second gesture timeout reports a failure rather than inventing completion.

动作等待双脚支撑和身体低速，完成还检查实际头部角度、角速度与连续稳定时间。回正容差约 1°，保留现有位置执行器受重力后的静态误差。停止仍保持支撑力，不等同于实体急停或断电。

VD29 retains VD28's straight-corridor limit. Side targets and jumps remain unsupported in this profile; VD26 retains its separate turn/hop experiments. Head turns are not visual localization or a promise to follow a person.

## Toward a physical prototype / 面向实体化

The reusable increment is a bounded action contract: finite motion, explicit support preconditions, cancellation with recentering, and completion from state feedback. A physical port still needs measured geometry/mass, servo angle and torque calibration, measured joint feedback, contact/IMU estimates, communication/watchdog handling and an independent emergency stop. Simulation command limits are not verified hardware operating limits.

实体化可复用有限动作、支撑前置条件、可打断回正和状态反馈；仍需实测结构/质量、舵机与力矩标定、关节反馈、接触和 IMU 估计、通信超时保护以及独立急停。本版本没有完成这些硬件工作，也没有移植学习策略。

## Validation / 验证

```powershell
./.venv-sim/Scripts/python.exe -m pytest tests/test_feedback_gestures.py -q
```

Native tests cover four gestures, cancellation while turning, walking-to-nod-to-walking, measured motion, contacts, actuator-force limits and absence of live body-state/external-force writes. A skipped optional dependency is not validation. Recorded runs retain source/model snapshots and simulated versus wall time. Automated text inputs are labelled separately from human or microphone interaction.
