"""Drive the real loopback text endpoint; explicitly automated input, not speech."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['commands', 'deepseek'], default='commands')
    p.add_argument('--config', type=Path, default=Path('Local/deepseek.local.json'))
    p.add_argument('--seconds', type=int, default=55)
    p.add_argument('--target-x', type=float, default=.65)
    p.add_argument('--target-y', type=float, default=.02)
    p.add_argument('--stop-step', type=int, default=8)
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--port', type=int, default=18856)
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--video', action='store_true')
    a = p.parse_args()
    command = [sys.executable, '-X', 'utf8', str(ROOT/'tools/run_dialogue.py'), '--out', str(a.out),
               '--profile', 'VD26', '--input-source', 'automated_demo', '--backend', a.backend, '--config', str(a.config),
               '--seconds', str(a.seconds), '--target-x', str(a.target_x), '--target-y', str(a.target_y),
               '--foot-view', '--fps', str(a.fps), '--port', str(a.port)]
    if a.realtime: command.append('--realtime')
    if a.video: command.append('--video')
    startup = subprocess.STARTUPINFO() if sys.platform == 'win32' else None
    if startup: startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    process = subprocess.Popen(command, cwd=ROOT, startupinfo=startup)
    url = f'http://127.0.0.1:{a.port}'
    stage = 'jump'; events = []; last_sent = 0.; completed = None
    def send(text, state):
        nonlocal last_sent
        req = urllib.request.Request(url+'/command', data=json.dumps({'text':text}).encode(),
                                     headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=2) as response:
            assert response.status == 202
        last_sent = state['sim_s']
        events.append(dict(text=text, sent_at_sim_s=last_sent, phase=state['phase'], steps=state['steps_completed']))
        print(json.dumps(events[-1], ensure_ascii=False), flush=True)
    while process.poll() is None:
        try:
            with urllib.request.urlopen(url+'/state', timeout=2) as response: state = json.load(response)
            now = state.get('sim_s', 0)
            if stage == 'jump' and now >= 3:
                send('跳一下', state); stage = 'wait-jump'
            elif stage == 'wait-jump' and state['status'] == 'complete':
                completed = now; stage = 'walk'
            elif stage == 'walk' and now > completed + 2:
                send('请到我这里来' if a.backend == 'deepseek' else '过来', state); stage = 'interrupt'
            elif stage == 'interrupt' and state['steps_completed'] >= a.stop_step and state['phase'] == 'swing':
                send('停一下', state); stage = 'resume'
            elif stage == 'resume' and state['status'] == 'stopped' and now > last_sent + 2:
                send('再到我这里来吧' if a.backend == 'deepseek' else '过来', state); stage = 'arrive'
            elif stage == 'arrive' and state['status'] == 'arrived':
                completed = now; stage = 'tail'
            elif stage == 'tail' and now > completed + 1:
                send('摆尾', state); stage = 'wait-tail'
            elif stage == 'wait-tail' and state['status'] == 'complete':
                send('挥手', state); stage = 'wait-wave'
            elif stage == 'wait-wave' and state['status'] == 'complete':
                completed = now; stage = 'complete'
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.015)
    result = dict(input_source='automated external loopback text client; not human or microphone input',
                  backend=a.backend, stage=stage, completed_at_sim_s=completed if stage=='complete' else None,
                  exit_code=process.returncode, commands=events)
    (a.out/'client.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if process.returncode == 0 and stage == 'complete' else 1


if __name__ == '__main__': raise SystemExit(main())
