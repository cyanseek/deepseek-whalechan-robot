# Contributing

Start with a reproducible behavior: input, expected outcome, actual result and rerun command.
See [the current body and controller](docs/CHARACTER.md) and [development milestones](docs/ROADMAP.md).

Install `requirements-dev.txt`, `requirements-contact.txt` and the matching character assets, then run:

```powershell
./.venv-sim/Scripts/python.exe -m pytest -q -ra
```

For motion, perception, cancellation or timing changes, also run affected native MuJoCo scenarios.
Inspect physical outcomes, task events and video together. Use new output directories and report
untested scopes accurately. Policy changes need matching model and observation/action bindings.

Keep simulation, replay, scripted input and model output distinguishable. Preserve source attribution
and asset terms. Do not commit credentials, personal recordings, installed assets or run directories.
