# VD28 contact walking / 接触约束步态

VD28 is an optional experimental profile on the unchanged VD24 body. It combines prescribed COM/foot trajectories, contact-constrained inverse dynamics, and the existing asynchronous DeepSeek text interaction. VD24/VD25/VD26 entry points remain available. This profile is under active development.

## Run locally

Install the matching character assets using [ASSETS.md](ASSETS.md), then add the optional solver:

```powershell
./.venv-sim/Scripts/python.exe -m pip install -r requirements-contact.txt
./Start-WhaleChan-ContactWalk.ps1 -Backend commands -Record
```

For real DeepSeek, fill the private configuration described in [INTERACTION.md](INTERACTION.md), then run:

```powershell
./Start-WhaleChan-ContactWalk.ps1 -Backend deepseek -Record
```

The panel accepts **过来、挥手、摆尾、停一下**. With DeepSeek, “继续走到我这里” starts a new approach to the red marker. The controller finishes a supported stop before accepting the next motion. Exact stop words bypass model inference; cancelled-generation replies cannot restart movement. These are text inputs, not microphone commands.

本入口支持正前方可见标记的走近、停止、继续、挥手和摆尾。步态与上肢、鲸尾及表情协调；目标方位超过 0.25 rad 时停止并说明当前不支持侧向走近。红/绿球是相机识别的测试标记，不是身体的一部分。VD28 不支持转向和小跳；这些功能保留在 VD26 入口，不能视作已经迁移到新控制器。

## Recorded validation

All figures below are native MuJoCo results on level ground, with estimated dynamics and ideal simulated proprioception. Video playback uses simulated time, not wall-clock throughput.

| Same 12-second forward-command window | VD26 | VD27 research baseline | VD28 |
| --- | ---: | ---: | ---: |
| Forward displacement | 243.3 mm | 21.4 mm | 103.3 mm |
| Mean forward speed | 20.27 mm/s | 1.78 mm/s | 8.61 mm/s |
| Completed steps within the window | 19 | 3 | 7 |
| Peak sole clearance, L / R | 0.50 / 2.08 mm | 4.15 / 4.57 mm | 5.47 / 5.19 mm |
| Mean loaded contact-point speed | 20.49 mm/s | 1.35 mm/s | 3.72 mm/s |

VD28 is about 4.8× faster than the VD27 lifted-foot baseline in this fixed window. It remains slower than VD26. The contact-speed metric includes rolling and landing; this comparison does not measure slip at equal speed or prove zero slip.

A separate 42-second run completed 21 steps. Every completed swing reached 4.51–5.47 mm sole clearance; maximum body tilt was 3.30°. Stops requested during weight transfer, swing and landing reached sustained double support in 0.454 / 1.240 / 0.718 simulated seconds, including a 150 ms stability confirmation.

The real-API demonstration completed 20 steps, an interrupted approach, resumed approach, wave, tail gesture, final arrival and another wave. The 80-second native run had no non-foot floor contact or actuator-limit violation; maximum tilt was 3.19°. Its continuous 55.5-second presentation excerpt ends after the action sequence. Inputs came from an automated external text client; this is not a human/voice trial. The full recording took 101.6 wall seconds, including rendering, so this configuration does not sustain 1× real-time recording on the tested machine.

## Engineering scope

The controller runs at 500 Hz, with 1 kHz native integration and bounded actuator targets. It regularizes double-support contact forces, constrains friction/support moments and joint limits, and checks the sole normal when confirming flat landing. It never writes the live root pose, velocities, mocap state or external forces. The LLM selects supported high-level intents and cannot emit joint controls.

[MicroDuck](https://github.com/pollen-robotics/microduck/tree/942e58e3aa6dbb28e7037203fff11866122c7567) and [microduck_rl](https://github.com/pollen-robotics/microduck_rl/tree/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3) informed the separation of motion, zero-command standing, foot clearance and slip evaluation. Their policy, weights and robot morphology are not reused by this controller.

Limitations: low absolute speed, lateral drift, flat-ground/specific initial-state validation, ideal sensing, expensive CPU optimization, and no transferred turning/hopping or hardware validation. Faster timings still produced failures; the passing timings are intentionally retained. This is not learned or robust general locomotion.

Run `python -m pytest -q -ra` with the optional dependencies and assets installed. Native VD28 tests check foot lift, stopping/resuming, body-state write restrictions, force limits, gesture completion and cancellation. Without `clarabel`, these optional tests skip; that skip does not validate VD28.
