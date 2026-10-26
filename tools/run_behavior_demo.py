"""Run an offline external-client walk/pause/resume/gesture demonstration."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=int, choices=(30, 60), default=30)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'client-source.py')
    command = [sys.executable, '-X', 'utf8', str(ROOT/'tools/run_dialogue.py'),
               '--profile', 'VD29', '--behaviors', '--backend', 'commands',
               '--seconds', '46', '--target-x', '.55', '--target-y', '.02',
               '--port', '0', '--input-source', 'automated_demo', '--foot-view',
               '--fps', str(args.fps), '--out', str(out/'native')]
    if args.video:
        command.append('--video')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    schedule = [(3., 'command', {'text': '过来'}), (7., 'command', {'text': '停一下'}),
                (9., 'command', {'text': '继续'}),
                (15., 'sequence', {'actions': ['wave', 'tail', 'nod']}),
                (22., 'command', {'text': '停一下'}), (24., 'command', {'text': '继续'})]
    sent, failure, url = [], None, None
    start = time.monotonic()
    with (out/'console.log').open('w', encoding='utf8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while process.poll() is None:
                if time.monotonic()-start > 300:
                    raise TimeoutError('Native demo exceeded 300 seconds; inspect console.log')
                try:
                    if url is None:
                        url = json.loads((out/'native/endpoint.json').read_text('utf8'))['url']
                    with opener.open(url+'/state', timeout=2) as response:
                        state = json.load(response)
                except (OSError, json.JSONDecodeError, urllib.error.URLError):
                    time.sleep(.05)
                    continue
                if schedule and state.get('sim_s', 0) >= schedule[0][0]:
                    _, route, payload = schedule.pop(0)
                    request = urllib.request.Request(url+'/'+route, data=json.dumps(payload).encode(),
                                                     headers={'Content-Type': 'application/json'})
                    with opener.open(request, timeout=2) as response:
                        if response.status != 202:
                            raise RuntimeError('Command was not queued')
                    sent.append(dict(sim_s=state['sim_s'], route=route, payload=payload))
                time.sleep(.03)
            if process.returncode:
                raise RuntimeError('Native process failed; inspect native/run.json and console.log')
            run = json.loads((out/'native/run.json').read_text('utf8'))
            events = run['events']
            if (schedule or run['failure'] is not None or run['final_phase'] != 'ready' or
                    sum(e['event'] == 'sequence_resumed' for e in events) != 2 or
                    not any(e['event'] == 'sequence_completed' for e in events) or
                    min(run['metrics']['confirmed_steps_by_foot']) < 1):
                raise RuntimeError('Demo did not meet its measured completion conditions')
        except (Exception, KeyboardInterrupt) as exc:
            failure = type(exc).__name__+': '+str(exc)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    report = dict(failure=failure, inputs=sent, wall_s=time.monotonic()-start,
                  command=command, evidence='offline automated external text client; native prescribed contact controller')
    (out/'client.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'failure': failure, 'output': str(out)}, ensure_ascii=False))
    return 1 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
