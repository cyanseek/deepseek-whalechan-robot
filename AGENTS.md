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
Do not commit secrets, personal data, local environments, installed assets or raw run directories.
