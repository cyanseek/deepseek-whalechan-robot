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
Do not commit secrets, personal data, local environments, installed assets or raw run directories.
