# VD24 body and control / 当前身体与控制

This guide describes the VD2.4 character shown in the README. See [ROADMAP.md](ROADMAP.md) for planned extensions.

## One physical state

The body has a floating base and 23 actuators. MuJoCo integrates body dynamics with gravity, ground contact, collision envelopes and actuator force limits. Estimated total mass is 0.972 kg; this is a simulation model, not a measured hardware specification.

The supplied skins follow articulated bodies. The original visual delivery's mocap preview is an input asset conversion stage; the current character itself uses dynamic bodies. Two external target markers use mocap positioning to represent changes in the environment.

## Runtime path

```text
Keyboard / prescribed demonstration / color-target task
  → task and motion state machines
  → contact-aware IK targets and PD actuation
  → MuJoCo integration
  → proprioception, robot RGB observations and task feedback
  → synchronized character skin, expressions and director recording
```

The director camera records the scene. Target detection uses a separate robot RGB camera. Ideal simulated IMU and model-based center-of-mass feedback are available to the controller; sensor-noise robustness is a future validation task.

The controller writes actuator targets. It does not move the live root pose to create locomotion. Physics tests check live state and applied-force arrays around controller calls.

## Motion and expression

Manual walking and the coordinated demo use the faster small-step mode. RGB target approach explicitly uses a slower precision mode; the selected mode is logged. A hop includes stability checks, preparation, bounded launch, flight foot placement and contact-aware recovery. Tail yaw plus two pitch joints form a three-axis rigid chain; flukes follow the distal segment.

The default smile is a spatially selected derivative of the existing happy morph, preserving open eyes. Semantic changes blend over 280 ms. Blinking and expression weights are coordinated to avoid stacking the previous eyelid artifacts. Clothing is skinned visual geometry, including a fitted surface derived from the actual thigh mesh; there is no cloth solver.

## Code map

| Area | Entry |
| --- | --- |
| Current model, gait, landing and cheerful expression | [bluefatfish_v24.py](../src/opendafeiyu/bluefatfish_v24.py) |
| Base dynamic model and IK | [bluefatfish.py](../src/opendafeiyu/bluefatfish.py) |
| Hop and landing state machine | [bluefatfish_v2.py](../src/opendafeiyu/bluefatfish_v2.py) |
| Tail articulation and gestures | [bluefatfish_v21.py](../src/opendafeiyu/bluefatfish_v21.py) |
| Coordinated demo and expression phases | [bluefatfish_v22.py](../src/opendafeiyu/bluefatfish_v22.py) |
| Small-step baseline | [bluefatfish_v23.py](../src/opendafeiyu/bluefatfish_v23.py) |
| Original morph extraction and transitions | [bluefatfish_expressions.py](../src/opendafeiyu/bluefatfish_expressions.py) |
| RGB targets and task cancellation | [tasks.py](../src/opendafeiyu/tasks.py) |
| Viewer, CLI and recording | [run_bluefatfish_v2.py](../tools/run_bluefatfish_v2.py) |
| Clothing parameters | [clothing.json](../assets/characters/bluefatfish-v24/clothing.json) |

## Development scope

Start with a reproducible input and inspect task state, physical outcome and video together. A foot-clearance measurement is not the same as center-of-mass jump height. Playback fps does not measure control throughput. Cancellation finishes a supported motion; hardware emergency stop and power-off are separate future interfaces.

Current VD24 uses explicit IK/PD and state machines. The older language/voice pipeline and Open Duck learned policy belong to different profiles; they are not automatically capabilities of this body. S2 needs mechanical design, measured inertias, actuator selection and hardware validation.
