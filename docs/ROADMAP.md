# Development roadmap / 开发路线

The goal is an expressive DeepSeek Whale-chan companion whose actions respond to people, with behaviors that developers can reproduce and extend. Work follows **S1 design and simulation → S2 physical robot**. Hardware design can inform S1 before a complete robot is built.

This page describes planned work. Current commands and measured capabilities remain in the [agent guide](AGENT_GUIDE.md), [body and control](CHARACTER.md) and [demonstration](DEMO.md). VD24 uses explicit IK/PD with ideal simulated proprioception; the demonstration runs without an LLM.

## 1. Make the same character easy to run

The matching AR2 asset ZIP and verified installer are available from the public GitHub Release. The repository quickstart downloads the assets and runs the VD24 demonstration.

Keep fresh-source setup checks that obtain matching assets, pass preflight and produce the same demonstration without a developer's private directories. Installation reports useful errors and preserves existing user modifications. Record dependency versions, command, outcome and video. Keep CPU inference/rendering separate from GPU training environments.

## 2. Improve movement while preserving expression

Measure commanded versus achieved velocity, heading drift, foot slip, upright time, contact transitions and landing behavior. Evaluate turning, start/stop, changed commands and bounded disturbances in declared scenarios. Keep the accepted VD24 controller as a comparison baseline and preserve failed trials.

Evaluate learned locomotion alongside the current controller. Begin with a model and actuator audit, then a small training smoke run and measured resource cost. A candidate must improve the declared motion tests while retaining task cancellation, tail/arm coordination and expressions. Publish an evaluation with its model, observation/action layout, normalizer, control rate, checkpoint and exact run command when that work is ready.

The [experimental training package](../training/README.md) now implements the
own-body mjlab/RSL-RL workflow, online data collection, normalized ONNX export and
independent CPU evaluation. It uses the separate VD32 physical candidate. Current
optimization focuses on foot clearance, contact slip, heading and supported stops.

MicroDuck's training/deployment workflow is a useful reference. Its existing policy is tied to a different body; adapting our character requires a matching controller or retraining. Body proportions, inertia, contact geometry and articulation stay part of the engineering problem. The current environment is Python 3.13; the referenced MicroDuck RL revision requires Python 3.12, so training exploration needs a separate environment.

## 3. Make conversation change what she does

The source [text interface](INTERACTION.md) connects DeepSeek high-level intents to local control, with cancellation and observable feedback. The next target extends this interaction to voice and broader perception: **come closer → wait → change target → greet**. Local control owns balance and actuator output.

Accept the feature when a person can interrupt or change a task during a live run, the character responds through physical motion and expression, and observable events agree with the video. Measure request-to-response and cancellation latency, target-arrival error and task outcome. Include a missing-target case and a model/network failure case. Separate operator input, scripted input, actual model output and sensor observations in the record.

Start with the current visible markers; extend perception as each target class is validated. The planned voice experience should identify what was actually heard and observed.

## 4. Let others add behaviors

The source [behavior interface](BEHAVIORS.md) provides a greeting sequence, bounded head feedback, supported pause/resume and an offline external-client demonstration. Extend these examples using the existing task and recording code.

A proposed behavior package contains its input schema, supported body/controller version, completion and cancellation conditions, asset/code terms, a short example and a rerun command. Learned policies additionally need their complete model binding and evaluation. A stable interface should let keyboard input, a coding agent and a language model request the same behavior and receive the same task outcome.

The next milestone is an independently reproduced behavior contribution. A gallery of compatible behaviors can grow from that evidence. A policy installer and gallery remain planned; current behavior and Agent entry points are documented in [AGENT_GUIDE.md](AGENT_GUIDE.md).

## 5. Transfer the experience to hardware

Choose the mechanism and actuators around the approved character design. Measure mass distribution, actuator response, delays and contact behavior; use the measurements to improve simulation. Evaluate whether the tail affects balance in the actual design instead of assuming that visible tail motion provides stabilization.

Progress through an actuator bench, supported body tests and bounded free-body trials. Map the same task semantics to the real sensors and motor interface. Hardware acceptance includes measured motion, task interruption, power and thermal behavior, and hardware stop/recovery procedures. Record simulation and physical results separately.

The intended experiences include greetings, following, desk-side companionship and reminders. Their scope will be determined by demonstrated perception, mobility and interaction on the physical body.

## Engineering references

- [MicroDuck runtime and product](https://github.com/pollen-robotics/microduck/tree/942e58e3aa6dbb28e7037203fff11866122c7567): reference for usable robot software and developer entry points.
- [MicroDuck simulation interface](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/docs/robot/simulation.md): reference for using the same task/control path with a simulated body.
- [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl/tree/b02fdea4902c2b93b7d34a83a6f05da4ececb7d3): reference for training, actuator modeling, evaluation and policy delivery.
- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) and [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground): mechanical and locomotion references retained from earlier work.

These references guide future engineering; their software and policies have not been integrated into VD24 by this documentation revision. Source attribution and licenses remain in the [README](../README.en.md#credits-and-licenses).
