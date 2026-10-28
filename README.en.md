# DeepSeek Whale-chan Robot 🐋

**Bringing DeepSeek Whale-chan into the real world.**

An open-source embodied character project for AI companionship and human-robot interaction, featuring the DeepSeek community's Whale-chan (大肥鱼 / 鲸鱼娘). The current MuJoCo implementation combines bipedal small steps, turning, coordinated tail motion and expressions. We are building her into an embodied companion that can converse, accept tasks and gain new behaviors through continued development, progressing toward a physical robot.

Actively developed · Current stage: MuJoCo simulation.

The current source includes [DeepSeek text interaction](docs/INTERACTION.md), [contact walking and supported stops](docs/CONTACT_WALK.md), and [head gestures, behavior sequences, pause and resume](docs/BEHAVIORS.md). The default entry point and 39-second demo below remain VD24; the additional entry points run from source.

[简体中文](README.md) · **English** · [日本語](README.ja.md)

**Training workflow available:** [own-body mjlab / PPO training and data tools](training/README.md), with online sampling, checkpoints, ONNX export and independent CPU walk/stop evaluation. Learned locomotion remains experimental; the demos below use the existing controller.

[Watch the demo](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html) · [Quick start](#quick-start) · [Roadmap](#roadmap) · [For agents](#for-agents) · [Developer guide](docs/CHARACTER.md)

[![DeepSeek Whale-chan Robot: DeepSeek Whale-chan walking with coordinated tail and arms in MuJoCo; real-time-speed simulation excerpt](docs/media/vd24/walk.gif)](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html)

**MuJoCo physics · Three-axis whale tail · Coordinated motion and expressions · VD2.4**

The GIF loops seconds 6–12 of the recording at 1× speed. Click for the complete **[39-second / 60fps](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html)** sequence: greeting, walking, settling, a small hop, turning, waving with tail movement, and resting. Actuators and ground contact drive the physical body; the high-level demonstration follows a prescribed sequence.

[▶ Watch the full 39 s / 60fps demo](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html) · [Download the recording](https://github.com/cyanseek/deepseek-whalechan-robot/releases/download/vd24-ar2/DeepSeek-Whalechan-Robot-VD24-demo.mp4)

## Features

| A little wave | Smiles and expressions |
| :---: | :---: |
| ![Whale-chan waves in native physics, at original speed](docs/media/vd24/wave.gif) | ![Blinking, smiling and expression changes in the live runtime](docs/media/vd24/expressions.gif) |
| **Three-axis whale-tail gesture** | **Small steps and coordinated motion** |
| ![Tail gesture from a native simulation, viewed from behind](docs/media/vd24/tail.gif) | ![Walking with coordinated arms and tail, at original speed](docs/media/vd24/walk-detail.gif) |

All clips play at the recorded speed. The wave comes from the full demo; expressions and tail gestures were recorded as separate native scenarios. [Sources and rerun commands](docs/DEMO.md)

- **Motion and expression:** small steps, turns, waves and experimental low hops, with coordinated arms and a three-axis whale tail.
- **Visuals and facial expression:** an open-eyed smile, natural blinks, smooth expression transitions, clothing and display adaptation.
- **Perception and interaction:** colored-marker detection and approach using the robot RGB camera; cancellation finishes steps according to support state.
- **Simulation and reproduction:** 23 actuators, a floating base, ground contacts and torque limits; saved video, trajectories, results and source snapshots.

The controller uses explicit IK/PD with ideal simulated proprioception, verified in flat-ground scenarios. Hops remain low; faster walking and landing control are still being improved. [Demo and measurements](docs/DEMO.md)

## Quick start

**Watch first, without installing anything:** [full 39-second video](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html).

**Run locally:** these commands start in the project source directory. Verified environment: Windows, Python **3.13.5**, MuJoCo **3.3.7**. Physics and control run on CPU; OpenGL renders the scene. The demo needs no API key or language model. This is a source-run demo.

### 1. Set up the environment

Open PowerShell in the project root:

```powershell
git clone https://github.com/cyanseek/deepseek-whalechan-robot.git
cd deepseek-whalechan-robot
py -3.13 -m venv .venv-sim
./.venv-sim/Scripts/python.exe -m pip install -r requirements-character.txt
```

### 2. Install the character assets

The matched **AR2 runtime package** (about 29.5 MB) contains the model, skins, textures and expression binding. No manual rig editing is needed:

```powershell
./.venv-sim/Scripts/python.exe tools/install_character_assets.py
./.venv-sim/Scripts/python.exe tools/character_preflight.py --delivery assets/local/BlueFatFish_Visual_Delivery
```

The installer downloads from [GitHub Releases](https://github.com/cyanseek/deepseek-whalechan-robot/releases/tag/vd24-ar2), verifies hashes and installs into `assets/local/BlueFatFish_Visual_Delivery/`, preserving existing files. For offline use, [download the ZIP](https://github.com/cyanseek/deepseek-whalechan-robot/releases/download/vd24-ar2/DeepSeek-Whalechan-Robot-Assets-AR2-VD24.zip) and pass it with `--archive`. [Asset installation](docs/ASSETS.md)

`ready_for_native_check` confirms dependency metadata and asset checks. Next, run the native simulation.

### 3. Bring her into motion

```powershell
# Open the interactive viewer
./Start-BlueFatFish.ps1 -Delivery assets/local/BlueFatFish_Visual_Delivery

# Run the complete demonstration and record a 60fps video at 1x playback speed
./Start-BlueFatFish.ps1 -Delivery assets/local/BlueFatFish_Visual_Delivery -Scenario coordinated -Seconds 39 -Record -Presentation -Fps 60
```

| Key | Behavior | Key | Behavior |
| --- | --- | --- | --- |
| W / A | Forward / turn | J | Experimental small hop |
| T / G | Tail gesture / wave | E | Change expression |
| R | Approach the red marker | C / Space | Cancel the current task |
| V / D | Camera / collision view | Close window | End the run |

Each launch creates a unique directory under `runs/`. Recordings are named `simulation.mp4`; results are in `run.json`. See the [output reference](docs/AGENT_GUIDE.md#outputs) for trajectories and snapshots. Cancellation uses a contact-aware finishing sequence.

### Try interruptible interaction

Open the offline panel and enter “过来”, “停一下”, “继续” or “打个招呼”. Approach uses the robot camera to locate the red marker; sequences advance on completion feedback.

```powershell
./.venv-sim/Scripts/python.exe -m pip install -r requirements-contact.txt
./Start-WhaleChan-Behaviors.ps1 -Backend commands -Record

./.venv-sim/Scripts/python.exe tools/run_behavior_demo.py --out runs/behavior-demo-01 --video
```

The automated text-client demo records about 46 simulated seconds of walking, pause/resume and wave → tail → nod. See [DeepSeek API setup](docs/INTERACTION.md) and the [behavior interface](docs/BEHAVIORS.md).

## Roadmap

We aim to build a companion that can converse, act and express a character of her own. A concrete experience: ask her to come closer and she greets you; say “wait” on the way and she settles and responds; change the target and she continues with the new task.

| Stage | Scope |
| --- | --- |
| **Available: body and expression** | MuJoCo small-step locomotion, tail coordination, expressions, colored-marker approach and cancelable tasks, with a matching asset installer and real recordings. |
| **Available from source: text and actions** | DeepSeek text intents, interruptible approach and supported stopping, head feedback, behavior sequences and pause/resume; explicit offline commands are supported. |
| **Next: motion and voice** | Velocity tracking, foot slip and disturbance balance, body-matched learned policies, and voice input. |
| **Expand: teach her new behaviors** | Turn greetings, following and companionship into reproducible examples, with a stable task interface for developers and agents to add, test and share behaviors. |
| **Build the physical companion** | Develop mechanics, actuators, sensors and calibration so the same task logic can drive a physical body; progressively validate greetings, desk-side companionship and reminders. |

The DeepSeek interface handles text and high-level intent; local control handles balance, actions and result feedback. Source includes reproducible behavior sequences. Default VD24 uses IK/PD; optional contact walking uses constrained inverse dynamics. Learned locomotion, voice input and hardware are future milestones. [Development milestones and acceptance goals](docs/ROADMAP.md)

## For agents

Give this prompt to a coding agent with access to the project files:

> Read README.en.md, docs/AGENT_GUIDE.md and project-info.json. Check the character assets and dependencies. Run a 2-second stand check, then the 39-second VD24 coordinated scenario. Use fresh output directories, report the result in run.json and link the recording. If assets are missing, list the missing files.

Use the [agent guide](docs/AGENT_GUIDE.md), [machine-readable project information](project-info.json) and [llms.txt documentation index](llms.txt). The interface is the existing Python CLI, which agents can invoke and inspect; the demonstration itself does not require an LLM.

## Develop and contribute

See [body and control](docs/CHARACTER.md) for the implementation and [measurements](docs/DEMO.md) for the validation scope.

Contributions are welcome in walking and landing control, DeepSeek conversation and speech, interaction design, appropriately licensed model adaptation, and robot mechanics and control. Start with one concrete scenario and include input, expected behavior and a rerun command. [Contributing](CONTRIBUTING.md)

Project path: **S1 design and simulation → S2 physical robot**.

Aliases: **DeepSeek 大肥鱼 / DeepSeek 鲸鱼娘 / Whale-chan / whale girl / DeepSeek-chan / OpenDafeiyu**.

## Credits and licenses

Thanks to the creators and open-source communities behind the following work.

- **Anime character and generated-image references:** [Neko3000 / DeepSeek Whale-chan](https://github.com/Neko3000/deepseek-whalechan) and [YunYueSama / codex-deepseek-pet](https://github.com/YunYueSama/codex-deepseek-pet). Neko's [character source notice](https://github.com/Neko3000/deepseek-whalechan/blob/main/THIRD-PARTY-NOTICES.md) credits Bilibili creators [ZipZipPipe](https://space.bilibili.com/4168597) and [上善无形](https://space.bilibili.com/4456176). Building on these references, we also used GPT to develop additional character-design ideas and refine details.
- **Base model and artwork:** **木子恶魔**, from the [original Dafeiyu work](https://www.aplaybox.com/details/model/ayXkRzVflmnd); the original file was obtained through [dsh-plugin-vrmpet3d](https://github.com/drliuhuan/dsh-plugin-vrmpet3d/tree/13cb2a2532ce34fa7dfd6ff2715da4f50565b161).
- **This project's embodied implementation:** building on the received pose and display adaptation, we implement the dynamic body, contact and torque constraints, motion control, task interaction, gait and whale-tail coordination, plus clothing, expression and visual runtime improvements. The original model and expression morphs retain their original authorship.
- **Robotics references**: [MicroDuck](https://github.com/pollen-robotics/microduck) / [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) as primary references for the embodied product, training and deployment path; [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) / [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground).

Project-authored code uses [MIT](LICENSE). Third-party models, art and demonstration media containing them have separate terms; the code license does not grant redistribution rights. See [assets](docs/ASSETS.md) and [NOTICE](NOTICE.md). This is an independent community project with no official affiliation with DeepSeek.
