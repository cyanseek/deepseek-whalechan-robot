# Interruptible text interaction (VD25)

For the newer **VD26** navigation/hop development profile, measurements and native validation commands, see [EMBODIMENT.md](EMBODIMENT.md) and `Start-WhaleChan-Embodiment.ps1`. The existing VD25 entry point and four-button interface remain available.

VD25 is an opt-in development entry point using the VD24 MuJoCo body and IK/PD controller. Type a request, watch its interpretation and physical execution, and send a new request to interrupt it. The existing VD24 launchers remain available.

## Start

Install the Python dependencies and matching character assets using the repository quickstart first. The native panel also requires Python's Tkinter (included in standard Windows Python installers; some Linux distributions package it separately).

Create `Local/deepseek.local.json` from `configs/deepseek.example.json`, then fill its `api_key` locally. `DEEPSEEK_API_KEY` is also supported when the JSON key is empty. Keep the real configuration under the Git-ignored `Local/` directory. Do not replace an existing configuration with the template.

```powershell
./Start-WhaleChan-Interaction.ps1 -Record
```

The panel accepts text and provides four buttons: **过来 / 挥手 / 摆尾 / 停一下**. Each run uses a new output directory. `-Seconds 180` controls the duration (maximum 600 seconds). The panel closes when the simulation ends; closing it early only closes the input window. Use **停一下** to request a supported stop.

Direct Python entry point:

```bash
python tools/run_dialogue.py --backend deepseek --config Local/deepseek.local.json --panel --realtime --video --seconds 180 --out runs/my-interaction
```

For explicit offline commands, use `--backend commands`, or PowerShell `-Backend commands`. This mode has a small fixed vocabulary and is visibly labelled. API failures do not silently switch to offline interpretation.

## What the requests do

| Request | Execution |
| --- | --- |
| 过来 / 靠近我一点 | Approach the visible red marker using robot-camera RGB detections. The marker represents the user's position in this demonstration. |
| 去绿色标记 | Select the green marker through the same RGB controller. |
| 挥手 / 跟我打个招呼 | Wait for supported stance, then perform the physical arm gesture. |
| 摆尾 / 摆摆鲸鱼尾巴 | Wait for supported stance, then articulate the three-joint whale tail. |
| 停一下 | Cancel the active request immediately and let the controller finish its current step and settle. |
| Other text | Short language reply or clarification; unsupported and multi-step requests do not start an arbitrary action. |

Every new input cancels the previous request. A generation counter rejects model replies belonging to cancelled inputs. Common exact stop phrases bypass the API. Language requests run in at most two background workers; only the simulation thread writes actuator targets. New gestures wait for measured support, and completion messages follow controller feedback. Request cancellation is distinct from a supported physical stop and from a hardware emergency stop.

Default idle expression is an open-eyed smile. Thinking and physical gestures change expression; completion briefly celebrates before returning to idle.

## Local adapter contract

The panel and test clients use the same loopback HTTP service (default port `18852`, adjustable with `--port`).

```http
POST http://127.0.0.1:18852/command
Content-Type: application/json

{"text":"挥手"}
```

The response is `202 {"status":"queued"}`. This acknowledges input queuing, not physical completion. Read `GET /state` for the current generation, input, reply, body phase, support contacts and recent events. `GET /frame.jpg` provides the current rendered preview. Text is limited to 500 characters; raw joints and shell commands are not part of this interface. Browser requests from other origins are rejected. The unauthenticated endpoint is intended only for trusted local clients; keep it on loopback.

`run.json`, `samples.json`, `trajectory.npz` and the optional `simulation.mp4` preserve actual behavior, simulated/wall time and the exact source/model snapshots. Input text and observable model decisions are recorded locally. API keys, provider headers, raw provider responses and hidden reasoning are excluded. Review input text before sharing a recording or run.

## Evidence and current scope

The 2026-10-22 native integration check used a separate automated text client through this HTTP interface and a real DeepSeek API. It exercised approach → walking cancellation → wave → tail cancellation → completed tail → cancellation while inference was pending → rejected stale reply → approach and supported arrival → wave. Recordings are native physics, with a separate director camera and robot RGB inset. This is text interaction; microphone input and speech output are future adapters.

The controller still uses ideal simulated proprioception and marker-based perception on flat ground. Multi-turn memory, compound-task planning, learned locomotion and hardware validation are separate work. The new interaction layer does not change the VD24 mass, collision geometry, actuator limits or low-hop capability.

## References

- [MicroDuck architecture](https://github.com/pollen-robotics/microduck/blob/942e58e3aa6dbb28e7037203fff11866122c7567/docs/design/architecture.md): separation between the robot-control owner and command transports. This upstream design document includes planned architecture; no MicroDuck policy was ported here.
- [DSH Pet](https://github.com/PC2005-cloud/dsh-pet#api--二次开发): separate command and state endpoints, plus visible interaction states.
- [DeepSeek's AstrBot integration](https://api-docs.deepseek.com/zh-cn/quick_start/agent_integrations/astrbot/): model-provider configuration separate from messaging-platform adapters.
- [Bilibili Live Assistant](https://github.com/YUKIQVQ/Live-assistant): an example of text-stream input, model integration and a desktop interface.
- [DeepSeek API](https://api-docs.deepseek.com/): the implemented chat-completion transport.

These are architectural references. This increment is implemented in this repository; no QQ/Bilibili account, message relay, upstream desktop-pet runtime or trained motion policy is required.
