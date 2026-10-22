# Measured embodiment development profile — VD26

This is a local development increment on the unchanged VD24 physical body. It adds fixed small-step navigation, continuous turning endpoints, a more consistent hop preparation posture, and interruptible text-triggered hops. VD25 remains selectable. No learned policy, voice input or hardware result is implied.

## Run

Install the dependencies and matching character assets as described in the README. Configure the private API file as described in [INTERACTION.md](INTERACTION.md). Never distribute the filled configuration.

```powershell
./Start-WhaleChan-Embodiment.ps1 -Record
```

```bash
python tools/run_dialogue.py --profile VD26 --backend deepseek --config Local/deepseek.local.json --target-x .65 --target-y .02 --panel --realtime --foot-view --video --fps 60 --seconds 180 --out runs/embodiment-example
```

Use `--backend commands` for the explicitly labelled offline vocabulary: `过来`, `去绿色标记`, `挥手`, `摆尾`, `跳一下`, `停一下`. A new input cancels the preceding intent. Cancellation in flight allows landing and recovery; it cannot stop gravity or freeze the body. Hops wait for support and can take several seconds to prepare. VD25 does not expose the new interactive hop action.

The **red ball** is a known-size RGB marker representing the user in these tests. The **green ball** is an alternative selectable target. They are not balance aids, rewards, robot sensors, or a substitute for demonstrated person tracking. The director camera is for presentation. Navigation sees only the robot RGB image and its color detections.

## Changes and measurements

- Forward requests use the previously tested 26 mm planned small step and lively timing. A visual-navigation request no longer continuously scales the stride into unvalidated intermediate values. This is a bounded step command, not metric velocity tracking.
- Straight and turning steps share one continuous endpoint formula. A tiny yaw request no longer changes the endpoint from swing-relative to stance-relative placement. Yaw increments are limited to 0.012 rad per step.
- VD26 stops at an observed marker distance of 0.35 m and bearing within 0.50 rad, followed by supported settling. VD25 retains 0.32 m / 0.15 rad. “Come” means approach to the marker's vicinity, not exact facing or positioning. The RGB range estimator uses a known 9 cm sphere and is approximate.
- Hop preparation converges to a consistent planned support height; the crouch is reduced to 4 mm to avoid excessive compression. Upward contact separation confirms takeoff. Only actuator controls are written; estimated masses, friction, collisions, joint ranges and force bounds remain unchanged.
- `MotionMetrics` measures loaded contact-point velocity, actual sole clearance, tilt and airborne time. It does not substitute requested foot lift for achieved clearance. Contact-point velocity includes touchdown and rolling, and is not a pure friction-identification measurement.

In retained 12-second straight-command checks, the earlier precision gait advanced 0.162 m and the lively gait advanced 0.243 m (about 0.0135 → 0.0203 m/s). VD26 brings this small-step mode into navigation. Loaded-contact speed stayed near 0.020 m/s: **foot sliding remains unresolved**, despite better progress per step. Actual walking sole clearance remains around 1–2 mm.

The local 55-second / 60 fps real-API recording completed a hop, 17 walking steps, a supported interruption, resumed approach, tail gesture and wave. It advanced about 0.199 m during the measured walking intervals, reached 8.6 mm minimum clearance across both soles during the hop, and had about 80 ms without foot contact. Maximum body tilt was 4.95°. These are small-hop and flat-ground results, not running or prominent jumping.

## Reproduce checks and recording

```bash
python tools/validate_embodiment.py --case front --out runs/check-front
python tools/validate_embodiment.py --case left --out runs/check-left
python tools/validate_embodiment.py --case right --out runs/check-right
python tools/validate_embodiment.py --case far --out runs/check-far
python tools/validate_embodiment.py --case stop-swing --out runs/check-interruption
python tools/validate_embodiment.py --case stop-flight --out runs/check-flight-cancel
python tools/record_interaction_demo.py --backend deepseek --target-x .65 --stop-step 8 --seconds 55 --fps 60 --realtime --video --out runs/recorded-text-demo
python -m pytest -q
```

Other cases: `stop-shift`, `stop-land`, `hop`, `stop-crouch`. Each writes a new directory, report, 100 Hz trajectory and exact source snapshot. The recording driver is an **automated external text client**, not microphone or human-input evidence. A user can send arbitrary text through the native panel or the same loopback API.

Four separate marker positions and three walking interruption phases passed the retained checks. The longer combination of a prior hop, an 0.80 m target and interruption failed in an earlier retained run; the 0.65 m combined demonstration passed. Passing individual cases does not establish arbitrary compositional robustness. Larger strides, stronger launch settings, and experimental whole-body optimization also produced failures and were not promoted. Wider turns, disturbances, terrain changes, high-clearance walking and larger jumps remain research work. Continue toward contact-aware dynamics control or a trained policy with a quantitative validation suite; more GPU compute alone does not repair the current gait model.
