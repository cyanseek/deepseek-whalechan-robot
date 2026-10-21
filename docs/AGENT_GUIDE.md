# Use DeepSeek Whale-chan Robot from a coding agent

Read [project-info.json](../project-info.json) for a compact capability and command index. This guide describes local CLI use of VD24; it does not install a plugin or expose an MCP server.

## Inputs and setup

- Run commands from the project root.
- Use the virtual environment described in [README.en.md](../README.en.md#quick-start).
- Install the matching AR2 runtime ZIP with `tools/install_character_assets.py` as described in [ASSETS.md](ASSETS.md). The installer populates `assets/local/BlueFatFish_Visual_Delivery`; no manual model editing is required.
- Each native invocation needs a fresh `--out` directory. The runner rejects an existing output directory.
- No API key, network call, language model or microphone is needed for these scenarios.

## Readiness

```powershell
./.venv-sim/Scripts/python.exe tools/character_preflight.py --delivery assets/local/BlueFatFish_Visual_Delivery
```

The stdout is JSON. Exit `0` with `status: ready_for_native_check` means dependency metadata and asset hashes passed. Exit `2` means `action_required`; inspect `errors` and `missing_files`. A valid preflight is not a dynamics/rendering test or a license grant.

## Minimal native check

```powershell
./.venv-sim/Scripts/python.exe tools/run_bluefatfish_v2.py --profile VD24 --delivery assets/local/BlueFatFish_Visual_Delivery --scenario stand --seconds 2 --out runs/agent-stand-01
```

This invokes real MuJoCo and the target-camera renderer without opening a viewer. It still needs a working OpenGL environment.

## Record the demonstration

```powershell
./.venv-sim/Scripts/python.exe tools/run_bluefatfish_v2.py --profile VD24 --delivery assets/local/BlueFatFish_Visual_Delivery --scenario coordinated --seconds 39 --video --fps 60 --presentation --realtime --out runs/agent-demo-01
```

`--realtime` paces the run; it does not guarantee real-time throughput on every computer. Video uses simulated time at 1× playback. `--presentation` changes the director camera/overlay, not the robot observations or physics.

For the interactive viewer, use `Start-BlueFatFish.ps1 -Delivery ...`. For focused checks, choose `walk`, `turn`, `tail`, `jump`, `expressions`, `approach` or `coord-cancel`. The raw runner defaults to the older `VD2` profile, so **always specify `--profile VD24`** when reproducing the current README.

## Outputs

| Output | Read it for |
| --- | --- |
| `run.json` | Run/profile/scenario, failure, final status, simulated and wall time, source fingerprints. |
| `samples.json` | Task and motion samples, support/contact and motion-mode observations. |
| `trajectory.npz` | Recorded physical and expression state for analysis or labelled replay. |
| `simulation.mp4` | Native recording, present when `--video` is used. |
| `binding.json` | Model binding and simulation assumptions. |
| `scene.xml`, `model-assets.zip` | Frozen scene/assets; include separately licensed character data. |
| `source-snapshot.zip` | Code/parameter snapshot associated with the run. |

Inspect both the process exit code and the final task outcome. For target approach, successful arrival is required; an arbitrary final pose is insufficient. Preserve failed outputs when reporting a problem.

## What to report

Return the exact command, profile, scenario, exit code, `failure`, final status/phase, simulated/wall time and recording path. Describe the controller as explicit IK/PD with ideal simulated proprioception. Distinguish a newly integrated run from a trajectory replay. Large jumps, running, noisy-sensor robustness, current-body language/voice end-to-end behavior and physical hardware are future work.

Future motion learning, task extensions and hardware milestones are described in [ROADMAP.md](ROADMAP.md); those interfaces are planned, not additional commands available in VD24.

For code changes, follow [CONTRIBUTING.md](../CONTRIBUTING.md). Preserve prior runs and snapshots. Never obtain motion by teleporting the base, removing relevant contacts or relabelling a replay as a new simulation. User authorization and third-party asset terms still govern any external upload or publication.
