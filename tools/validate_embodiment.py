"""Deterministic RGB approach, phase interruption and hop validation (no API)."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import mujoco
import numpy as np
from opendafeiyu.bluefatfish_v24 import build_scene, LivelyController
from opendafeiyu.bluefatfish_v26 import MeasuredStepController
from opendafeiyu.dialogue import InteractionController
from opendafeiyu.motion_metrics import MotionMetrics
from opendafeiyu.tasks import detect_targets

CASES = {
    'front': (.65, .02, 32), 'left': (.65, .12, 32),
    'right': (.65, -.10, 32), 'far': (.80, .02, 45),
    'stop-shift': (.65, .02, 22), 'stop-swing': (.65, .02, 22),
    'stop-land': (.65, .02, 22), 'hop': (.65, .02, 18),
    'stop-flight': (.65, .02, 18), 'stop-crouch': (.65, .02, 18),
}


def run(out, case, profile='VD26', submit_at=3.):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    x, y, seconds = CASES[case]
    _, xml, assets, binding = build_scene()
    tree = ET.fromstring(xml)
    tree.find(".//body[@name='target_red']").set('pos', f'{x} {y} .24')
    tree.find(".//body[@name='target_green']").set('pos', '.65 -.22 .24')
    xml = ET.tostring(tree, encoding='unicode')
    m = mujoco.MjModel.from_xml_string(xml, assets)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    low = (MeasuredStepController if profile == 'VD26' else LivelyController)(m, d, binding)
    interaction = InteractionController(low)
    metrics = MotionMetrics(m, low)
    renderer = mujoco.Renderer(m, 240, 320)
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    detections = {}; rows = []; trajectory = []; checks = 0
    sent = stopped = resumed = False
    failed = None; stop_time = None; started = time.perf_counter()
    jump_case = case in ('hop', 'stop-flight', 'stop-crouch')
    wanted_phase = case.removeprefix('stop-') if case.startswith('stop-') else None
    sources = list((ROOT/'src/opendafeiyu').glob('*.py')) + [Path(__file__)]
    with zipfile.ZipFile(out/'source-snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sources: z.write(p, p.relative_to(ROOT))
    (out/'scene.xml').write_text(xml, 'utf8')
    try:
        for step in range(round(seconds / m.opt.timestep)):
            now = float(d.time)
            if step % 100 == 0 and not jump_case:
                renderer.update_scene(d, camera='robot_camera', scene_option=option)
                detections = detect_targets(renderer.render())
            if now >= submit_at and not sent:
                interaction.submit('跳一下' if jump_case else '过来', now, source='automated_validation')
                sent = True
            if wanted_phase and not stopped and low.phase == wanted_phase and (jump_case or low.steps_completed >= 8):
                interaction.submit('停一下', now, source='automated_phase_interrupt')
                stopped = True; stop_time = now
            if stopped and not jump_case and not resumed and interaction.status == 'stopped' and now >= stop_time + 2:
                interaction.submit('过来', now, source='automated_resume')
                resumed = True
            if step % 10 == 0:
                arrays = (d.qpos, d.qvel, d.mocap_pos, d.mocap_quat, d.xfrc_applied, d.qfrc_applied)
                before = [a.copy() for a in arrays]
                interaction.tick(detections, now)
                for old, current in zip(before, arrays):
                    if not np.array_equal(old, current): raise RuntimeError('live_state_write')
                checks += 1
            mujoco.mj_step(m, d)
            if np.any(np.abs(d.actuator_force) > m.actuator_forcerange[:, 1] + 1e-7):
                raise RuntimeError('actuator_force_limit')
            if step % 10 == 0:
                detail = metrics.sample(d, low.phase in ('shift', 'swing', 'land'))
                rows.append(dict(sim_s=float(d.time), phase=low.phase, status=interaction.status,
                                 root=d.qpos[:3].tolist(), steps=low.steps_completed,
                                 feet_contact=low.contacts(), **detail))
                trajectory.append(d.qpos.copy())
                if detail['tilt_rad'] > .15: raise RuntimeError('tilt_exceeds_validation_envelope_8.6deg')
                if metrics.other_floor: raise RuntimeError('nonfoot_floor_contact')
            if low.phase == 'failed': raise RuntimeError('controller_failed')
    except Exception as exc:
        failed = str(exc)
    finally:
        renderer.close(); interaction.close()
    complete = interaction.status == ('stopped' if case in ('stop-flight', 'stop-crouch') else 'complete' if case == 'hop' else 'arrived')
    stop_events = [e for e in interaction.events if e['event'] == 'supported_stop' and stop_time is not None and e['sim_s'] >= stop_time]
    passed = failed is None and complete and low.phase == 'ready' and all(low.contacts())
    if wanted_phase: passed = passed and stopped and bool(stop_events)
    report = dict(profile=profile, case=case, submit_at=submit_at, target_red_xyz=[x, y, .24], sim_s=float(d.time),
                  wall_s=time.perf_counter()-started, passed=bool(passed), failure=failed,
                  final_status=interaction.status, final_phase=low.phase, final_support=low.contacts(),
                  steps=low.steps_completed, metrics=metrics.report(), state_write_checks=checks,
                  stop_requested=stopped, stop_phase=wanted_phase, stop_events=stop_events,
                  events=interaction.events, motion_events=low.events,
                  model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                  evidence='native MuJoCo; automated text client; RGB marker navigation; no model/microphone claim')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf8')
    (out/'samples.json').write_text(json.dumps(rows, ensure_ascii=False), 'utf8')
    np.savez_compressed(out/'trajectory.npz', qpos=np.asarray(trajectory))
    print(json.dumps({k:report[k] for k in ('profile', 'case', 'passed', 'failure', 'final_status', 'steps')}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=list(CASES), required=True)
    parser.add_argument('--profile', choices=['VD25', 'VD26'], default='VD26')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.out, args.case, args.profile)['passed'] else 1)
