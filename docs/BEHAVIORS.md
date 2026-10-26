# Feedback-driven behaviors / 动作组合

Available from the current source checkout; the packaged Release keeps the VD24 demo. Uses the VD29 contact controller on the VD24 physical body.

```powershell
./Start-WhaleChan-Behaviors.ps1 -Backend commands -Record
```

Run a complete 46-second walk → pause → resume → wave/tail/nod sequence with an automated external text client:

```powershell
./.venv-sim/Scripts/python.exe tools/run_behavior_demo.py --out runs/behavior-demo-01 --video
```

The offline demo needs the character assets and `requirements-contact.txt`. It records the native video and physical measurements in `native/`, and the inputs/outcome in `client.json`. A fresh output directory is required. Simulation time and elapsed wall time are recorded separately. The interactive panel accepts your own text inputs instead of this prescribed test schedule.

The panel adds **打个招呼** (wave → tail → nod), **看看左右**, **继续**, and **取消任务**. “停一下” requests supported stopping and retains remaining intentions. “继续” reacquires an approach target or restarts the interrupted gesture from its beginning after settling. “取消任务” discards future actions. A new ordinary instruction replaces both active and paused intentions.

组合动作按实际完成反馈推进，不按固定播放时长切换。每项动作有仿真时间上限：接近60秒、手势20秒；超时收步并清空任务，不自动重试。停止不是硬件急停或断电。接近仍使用机器人RGB相机中的颜色标记，不是声音定位。最大6项，支持情况由当前身体控制器决定。

The opt-in runner accepts `POST /sequence` with `{"actions":["wave","tail","nod"]}` in addition to `/command`. Invalid or unsupported actions are rejected before changing the current plan. `/state` includes `behavior.remaining`, `paused_actions`, `can_resume` and the current deadline. Events and source snapshots remain in each fresh run directory. No raw joints or executable commands are accepted.

`-Backend deepseek` retains the existing asynchronous API route for individual instructions and the private config file. Named routines and pause/resume/cancel use explicit local semantics; they are not presented as LLM-generated sequences. Multi-action language generation, microphone input and hardware execution are not implemented here.

## Learned-policy boundary / 学习策略边界

`opendafeiyu.policy_bundle` and `tools/check_policy_bundle.py` bind a TorchScript policy to body/home SHA256, MuJoCo version, actuator joint order, observation schema, frequency and action scaling. Wrong bindings and nonfinite observations/actions are rejected; no fallback controller is silently substituted. Torch is optional and imported only by `TorchPolicy`.

```bash
python tools/check_policy_bundle.py PATH_TO_BUNDLE --body PATH_TO_TRAINING_BODY --allow-experimental --infer
```

Experimental weights require explicit opt-in. Contract validation checks interface compatibility and one inference; locomotion requires a separate physical evaluation. The VD29 behavior launcher uses its matched contact controller and 0.972 kg VD24 body with MuJoCo3.3.7. Training bodies and checkpoints use their own bound models and environments.

## Measured footfalls / 独立落步计数

`run.json.metrics` includes left/right loaded-contact slip and `confirmed_steps_by_foot`, separate from the planner's `steps_completed`. A measured forward step requires at least 3 mm clearance, 5 mm sole travel, 30 ms flight and 30 ms loaded landing. Short contact chatter cannot confirm another step; sample gaps over 50 ms discard pending footfalls. `alternating_transitions` and timestamped `footstep_events` allow bilateral gait inspection. These measurements describe the run; a counter alone does not establish overall walking qualification.
