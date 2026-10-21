# Development guide

Read README.md, docs/AGENT_GUIDE.md and docs/CHARACTER.md. Current profile is VD24.
Use the matching asset installer, then run the native stand check before changing behavior.
Run `python -m pytest -q -ra` after installing requirements-dev.txt and the character assets.
For control changes, also run affected native scenarios and inspect physical outcomes and video.
Keep model, controller and observation/action bindings matched. Preserve original asset attribution.
Use a fresh runs/ directory. Keep simulation, replay, scripted inputs and model outputs distinguishable.
Never replace physics with live base-pose or force writes, removed contacts or mislabeled replay.
Dialogue/voice and hardware milestones in docs/ROADMAP.md are planned capabilities.
Do not commit secrets, personal data, local environments, installed assets or raw run directories.
