# Development guide

Read README.md, docs/AGENT_GUIDE.md and docs/CHARACTER.md. Current profile is VD24.
Use the matching asset installer, then run the native stand check before changing behavior.
Run `python -m pytest -q -ra` after installing requirements-dev.txt and the character assets.
For control changes, also run affected native scenarios and inspect physical outcomes and video.
Keep model, controller and observation/action bindings matched. Preserve original asset attribution.
Use a fresh runs/ directory. Keep simulation, replay, scripted inputs and model outputs distinguishable.
Never replace physics with live base-pose or force writes, removed contacts or mislabeled replay.
VD25 adds the opt-in text/API interaction entry point in docs/INTERACTION.md; the physical body remains VD24.
VD26 adds opt-in measured-step navigation and interactive hops; read docs/EMBODIMENT.md for exact validation scope and retained failures. Do not describe it as robust locomotion. Keep VD25 available.
Run tests/test_dialogue.py and a native external-client scenario for interaction changes. Voice and hardware remain roadmap work.
Keep API calls off the physics thread, reject cancelled-generation replies, and distinguish input receipt from supported stopping.
VD28 is an optional contact-constrained straight-walk profile; read docs/CONTACT_WALK.md and install requirements-contact.txt. It runs at 500 Hz with 1 kHz native actuation, supports straight RGB approach/stop/gestures, and declines side targets and jumps. Preserve the older defaults. Run tests/test_contact_walk.py with the solver and assets installed; an optional-dependency skip does not validate VD28. Do not imply hardware, learned motion, universal steering or speed superiority over VD26.

VD29 is optional bounded head feedback on VD28: read docs/FEEDBACK_GESTURES.md; use Start-WhaleChan-Gestures.ps1. Nod/shake/left/right wait for support, return to neutral on cancellation, and confirm measured settling. The main showcase stays short; extended actions belong in separate footage or the interactive panel. No added physical joints or hardware validation. Run tests/test_feedback_gestures.py with native assets/solver; preserve prior versions and failed validation records.
Do not commit secrets, personal data, local environments, installed assets or raw run directories.

Motion metrics include independent contact-confirmed footfalls (3mm lift, 5mm forward travel, 30ms flight/landing). Preserve planner counters separately. Run tests/test_footsteps.py and native contact-walk tests after changing detection. tools/run_behavior_demo.py provides an offline external-client sequence; use a fresh output directory and inspect both client.json and native/run.json. A successful scripted demo uses the stated contact controller, not a learned policy or microphone input.

VD35 adds opt-in feedback-driven routines, supported pause/resume and intent cancellation on VD29 via Start-WhaleChan-Behaviors.ps1. Read docs/BEHAVIORS.md. Resume restarts gestures/reacquires RGB targets; it never restores live state. Keep training weights separate until physically qualified. policy_bundle/check_policy_bundle bind model, home, engine version, joint order and control contract; experimental opt-in is not locomotion acceptance. Run tests/test_behaviors.py, tests/test_policy_bundle.py and a native external-client sequence; preserve failures and old inputs.

VD39 adds the separate experimental training/ package; read training/README.md.
It uses the VD32 dynamics candidate, mjlab1.3/RSL-RL5, 78 actor inputs/23 actions,
50Hz policy/1kHz implicit motor control and independent MuJoCo3.10/ONNX evaluation.
Run training/tests in its own Python3.12 environment, then a 64-world/5-update
smoke, export and CPU evaluation for training/control changes. Do not install its
physics dependencies into the character viewer environment. Keep checkpoints,
raw runs and local credentials out of Git. Preserve all failure records locally;
describe learned motion according to independent physical outcomes, separately
from the existing character demos. No trained policy is installed by default.
