"""Native VD24 physics with VD25 interruptible text interaction and recording."""
from pathlib import Path
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import queue
import subprocess
import sys
import threading
import time
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from opendafeiyu.bluefatfish import DELIVERY, R_FORWARD
from opendafeiyu.bluefatfish_v24 import build_scene, LivelyController, CheerfulExpressions
from opendafeiyu.bluefatfish_v26 import MeasuredStepController
from opendafeiyu.motion_metrics import MotionMetrics
from opendafeiyu.dialogue import DeepSeekClient, InteractionController
from opendafeiyu.tasks import detect_targets

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input-source', choices=['live_text','automated_demo'], default='live_text')
    parser.add_argument('--profile', choices=['VD25','VD26','VD28','VD29'], default='VD25')
    parser.add_argument('--target-x', type=float, default=.48)
    parser.add_argument('--target-y', type=float, default=.06)
    parser.add_argument('--foot-view', action='store_true')
    parser.add_argument('--backend', choices=['commands', 'deepseek'], default='commands')
    parser.add_argument('--config', type=Path, default=Path('Local/deepseek.local.json'))
    parser.add_argument('--delivery', type=Path, default=DELIVERY)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--port', type=int, default=18852)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--fps', type=int, choices=[30,60], default=30)
    parser.add_argument('--panel', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--behaviors', action='store_true', help='Enable feedback-driven routines and pause/resume')
    args = parser.parse_args()
    if not 0 < args.seconds <= 600: parser.error('seconds must be in (0,600]')
    translator = DeepSeekClient(args.config) if args.backend == 'deepseek' else None
    args.out.mkdir(parents=True, exist_ok=False)
    m, xml, assets, binding = build_scene(args.delivery)
    if not np.isfinite([args.target_x,args.target_y]).all() or not .35<=args.target_x<=2 or abs(args.target_y)>1:
        parser.error('Target must be finite, x in [.35,2], |y| <= 1')
    tree=ET.fromstring(xml)
    tree.find(".//body[@name='target_red']").set('pos',f'{args.target_x} {args.target_y} .24')
    xml=ET.tostring(tree,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets)
    d = mujoco.MjData(m); mujoco.mj_forward(m,d)
    if args.profile=='VD29':
        from opendafeiyu.bluefatfish_v29 import FeedbackController
        low=FeedbackController(m,d,binding)
    elif args.profile=='VD28':
        from opendafeiyu.bluefatfish_v28 import ContactWalkController
        low=ContactWalkController(m,d,binding)
    else:
        low = (MeasuredStepController if args.profile=='VD26' else LivelyController)(m,d,binding)
    control_stride=getattr(low,'control_stride',10)
    metrics = MotionMetrics(m,low)
    face = CheerfulExpressions(m,args.delivery,assets['expression-cutout-map.npz'])
    if args.behaviors:
        from opendafeiyu.behaviors import BehaviorController
        interaction = BehaviorController(low, translator)
    else:
        interaction = InteractionController(low, translator)
    sources = sorted((ROOT/'src/opendafeiyu').glob('bluefatfish*.py')) + [ROOT/'src/opendafeiyu/tasks.py', ROOT/'src/opendafeiyu/dialogue.py', ROOT/'src/opendafeiyu/motion_metrics.py', Path(__file__), ROOT/'tools/dialogue_panel.py']
    sources += list((ROOT/'assets/characters').glob('bluefatfish-v*/clothing.json'))
    sources += [ROOT/'src/opendafeiyu/footsteps.py']
    if args.behaviors: sources += [ROOT/'src/opendafeiyu/behaviors.py']
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with zipfile.ZipFile(args.out/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(args.out/'model-assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('scene.xml',xml)
        for name,data in assets.items():z.writestr(name,data)
    (args.out/'binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2),encoding='utf-8')
    messages = queue.Queue(maxsize=32)
    shared = {'state': {'status':'starting'}, 'frame': b''}
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def respond(self,status,data,ctype='application/json'):
            raw = data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',ctype)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            with lock:
                state,frame=shared['state'],shared['frame']
            # A slow reader must never hold the simulation's shared-state lock.
            if self.path=='/state':return self.respond(200,state)
            if self.path=='/frame.jpg':return self.respond(200,frame,'image/jpeg')
            self.respond(404,{'error':'not_found'})
        def do_POST(self):
            if self.path not in ('/command', '/sequence') or (self.path=='/sequence' and not args.behaviors):return self.respond(404,{'error':'not_found'})
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{args.port}':return self.respond(403,{'error':'origin'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError()
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError()
                request=json.loads(self.rfile.read(length))
                if self.path=='/sequence':
                    if not isinstance(request,dict) or set(request)!={'actions'}:raise ValueError()
                    payload=interaction._validate(request['actions']);kind='sequence'
                else:
                    if not isinstance(request,dict) or set(request)!={'text'}:raise ValueError()
                    payload=request['text'];kind='text'
                    if not isinstance(payload,str) or not payload.strip() or len(payload)>500:raise ValueError()
                messages.put_nowait((kind,payload,time.perf_counter()))
            except (ValueError,KeyError,TypeError):return self.respond(400,{'error':'invalid_text_request'})
            except queue.Full:return self.respond(429,{'error':'queue_full'})
            self.respond(202,{'status':'queued'})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    args.port=server.server_address[1]
    (args.out/'endpoint.json').write_text(json.dumps({'url':f'http://127.0.0.1:{args.port}'}),encoding='utf8')
    threading.Thread(target=server.serve_forever,daemon=True).start()
    panel=None
    if args.panel:
        startup=subprocess.STARTUPINFO() if sys.platform=='win32' else None
        if startup:startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
        panel=subprocess.Popen([sys.executable,str(ROOT/'tools/dialogue_panel.py'),'--port',str(args.port),'--profile',args.profile]+(['--behaviors'] if args.behaviors else []),startupinfo=startup)
    sensor=mujoco.Renderer(m,240,320);renderer=mujoco.Renderer(m,720,1280)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    camera=mujoco.MjvCamera();camera.azimuth=210;camera.elevation=-10;camera.distance=.80
    foot_renderer=mujoco.Renderer(m,200,320) if args.foot_view else None
    foot_camera=mujoco.MjvCamera();foot_camera.azimuth=90;foot_camera.elevation=-8;foot_camera.distance=.24
    foot_option=mujoco.MjvOption();foot_option.geomgroup[3]=1
    foot_option.flags[mujoco.mjtVisFlag.mjVIS_SKIN]=False
    foot_option.geomgroup[2]=0
    detail={'clearance_m':[0,0]}
    font_path='C:/Windows/Fonts/msyh.ttc'
    font=ImageFont.truetype(font_path,20) if Path(font_path).exists() else ImageFont.load_default(size=20)
    small=ImageFont.truetype(font_path,16) if Path(font_path).exists() else ImageFont.load_default(size=16)
    writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(str(args.out/'simulation.mp4'),fps=args.fps,codec='libx264',quality=7)
    wall0=time.perf_counter();next_sensor=next_frame=next_sample=0.
    detections={};rgb=np.zeros((240,320,3),np.uint8);samples=[];qpos=[];ctrl=[];expressions=[]
    peak_tilt=0.;peak_force=np.zeros(m.nu);state_checks=0;failure=None;received=0;frames=0
    physics_wall=control_wall=render_wall=0.
    names={'idle':'等待指令','thinking':'正在理解','waiting_for_support':'先站稳，再执行','approach':'看着标记走近',
           'wave':'挥手打招呼','tail':'摆动鲸尾','jump':'准备、起跳与落地','nod':'轻轻点头','shake':'轻轻摇头',
           'look_left':'向自己的左边看','look_right':'向自己的右边看','stopping':'正在收步与回正','stopped':'已经停稳','complete':'动作完成',
           'arrived':'已走近目标','settling_at_target':'到附近了，正在站稳','lost':'目标丢失，停下','reply':'回应','model_error':'请重试'}
    try:
        for step in range(round(args.seconds/m.opt.timestep)):
            now=float(d.time)
            if now>=next_sensor:
                sensor.update_scene(d,camera='robot_camera',scene_option=option)
                rgb=sensor.render();detections=detect_targets(rgb);next_sensor+=.1
            while not messages.empty():
                kind,payload,received_wall=messages.get_nowait()
                if kind=='sequence':interaction.submit_sequence(payload,now,source='loopback_sequence_endpoint')
                else:interaction.submit(payload,now,source='live_text_endpoint')
                received+=1
                interaction.event(now,'input_delivery',queue_latency_wall_s=time.perf_counter()-received_wall)
            if step%control_stride==0:
                before=[v.copy() for v in (d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)]
                tick=time.perf_counter();command=interaction.tick(detections,now)
                for previous,current in zip(before,(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)):
                    if not np.array_equal(previous,current):raise RuntimeError('Controller modified live state or external forces')
                state_checks+=1
                if step%10==0:
                    emotion=interaction.expression(now)
                    face.update(now,emotion)
                control_wall+=time.perf_counter()-tick
            if hasattr(low,'apply_actuation'):
                before=[v.copy() for v in (d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)]
                low.apply_actuation()
                for previous,current in zip(before,(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)):
                    if not np.array_equal(previous,current):raise RuntimeError('Actuation modified live state or external forces')
            tick=time.perf_counter();mujoco.mj_step(m,d);physics_wall+=time.perf_counter()-tick
            orientation=d.xmat[low.base].reshape(3,3)@R_FORWARD.T
            tilt=float(np.arccos(np.clip(orientation[2,2],-1,1)));peak_tilt=max(peak_tilt,tilt)
            peak_force=np.maximum(peak_force,np.abs(d.actuator_force))
            if tilt>(.15 if args.profile in ('VD28','VD29') else .65) or low.phase=='failed':raise RuntimeError('Stability check failed')
            if np.any(np.abs(d.actuator_force)>m.actuator_forcerange[:,1]+1e-7):raise RuntimeError('Actuator limit exceeded')
            if step%10==0:
                detail=metrics.sample(d,low.phase in ('shift','swing','land'))
                if metrics.other_floor:raise RuntimeError('Non-foot floor contact')
            if now>=next_sample:
                snapshot={'sim_s':float(d.time),'wall_s':time.perf_counter()-wall0,'generation':interaction.generation,
                    'status':interaction.status,'action':interaction.action,'phase':low.phase,'root':d.qpos[:3].tolist(),
                    'feet_contact':low.contacts(),'steps_completed':low.steps_completed,'expression':face.weights.copy(),
                    'detections':{k:vars(v) for k,v in detections.items()},'tilt_rad':tilt,'motion_mode':low.motion_mode,
                    'sole_clearance_m':detail['clearance_m'],
                    'loaded_contact_speed_by_foot_m_s':detail['loaded_contact_speed_by_foot_m_s'],
                    'confirmed_steps_by_foot':detail['confirmed_steps_by_foot'],
                    'wave_active':low.wave_start is not None,'tail_active':low.tail_started is not None}
                if args.behaviors:snapshot['behavior']=interaction.behavior_state()
                samples.append(snapshot);qpos.append(d.qpos.copy());ctrl.append(d.ctrl.copy());expressions.append(face.weights.copy())
                if hasattr(low,'head_ids'):
                    snapshot.update(head_action=low.head_action,head_returning=low.head_returning,
                                    head_angles=d.qpos[low.qaddresses[low.head_ids]].tolist(),
                                    head_velocity=d.qvel[low.dofs[low.head_ids]].tolist(),
                                    head_target=low.channels.get('head_feedback',np.zeros(2)).tolist())
                with lock:shared['state']={**snapshot,'input':interaction.text,'reply':interaction.reply,
                    'backend':args.backend,'events':interaction.events[-24:]}
                next_sample+=.05
            if now>=next_frame:
                tick=time.perf_counter()
                # Recording camera is separate from RGB observations used for navigation.
                yaw=float(np.arctan2(orientation[1,0],orientation[0,0]))
                desired=(-65 if interaction.action=='tail' else 210)+np.degrees(yaw)
                camera.azimuth+=((desired-camera.azimuth+180)%360-180)*.08
                camera.lookat[:]=[d.qpos[0]+.01,d.qpos[1],.19]
                renderer.update_scene(d,camera=camera,scene_option=option)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                reply_lines = [interaction.reply[i:i+56] for i in range(0,len(interaction.reply),56)] or ['']
                status_y = 69 + 25*len(reply_lines)
                draw.rectangle((0,0,1280,status_y+26),fill=(20,29,43))
                mode='DeepSeek API' if args.backend=='deepseek' else '离线明确指令'
                draw.text((20,8),f'DeepSeek Whale-chan Robot · {args.profile} | {mode} · 真实物理仿真',font=font,fill='white')
                draw.text((20,39),'输入：'+interaction.text[:48],font=font,fill=(150,220,255))
                for index, line in enumerate(reply_lines):
                    draw.text((20,69+25*index),('回应：' if index==0 else '　　　')+line,font=font,fill='white')
                draw.text((20,status_y),f'{names.get(interaction.status,interaction.status)} · {low.phase} | 模拟 {now:.2f}s / 实际 {time.perf_counter()-wall0:.2f}s',font=small,fill=(160,215,230))
                frame.paste(Image.fromarray(rgb),(944,460));draw.rectangle((944,430,1264,460),fill=(20,29,43))
                draw.text((950,434),'机器人相机：过来 = 红色标记',font=small,fill='white')
                if foot_renderer:
                    foot_camera.lookat[:]=[d.qpos[0],d.qpos[1],.045]
                    foot_renderer.update_scene(d,camera=foot_camera,scene_option=foot_option)
                    frame.paste(Image.fromarray(foot_renderer.render()),(16,480))
                    draw.rectangle((16,446,336,480),fill=(20,29,43))
                    draw.text((22,450),'物理脚部：同一实时仿真状态',font=small,fill='white')
                draw.rectangle((0,690,1280,720),fill=(20,29,43))
                draw.text((20,694),f'步数 {low.steps_completed} | 脚底离地 L/R {detail["clearance_m"][0]*1000:.1f}/{detail["clearance_m"][1]*1000:.1f} mm | {"自动文本测试客户端" if args.input_source=="automated_demo" else "本地文本输入"} · 无麦克风 · 原速',font=small,fill='white')
                if frames==0:frame.save(args.out/'frame-start.png')
                if writer:writer.append_data(np.asarray(frame))
                stream=io.BytesIO();frame.save(stream,format='JPEG',quality=82)
                with lock:shared['frame']=stream.getvalue()
                frames+=1;next_frame+=1/args.fps;render_wall+=time.perf_counter()-tick
            if args.realtime or args.panel:
                remaining=d.time-(time.perf_counter()-wall0)
                if remaining>0:time.sleep(min(remaining,.005))
    except KeyboardInterrupt:
        interaction.submit('停一下',float(d.time),source='process_interrupt')
        failure='user_process_interrupt_before_supported_stop_validation'
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)
    finally:
        if writer:writer.close()
        if frames:frame.save(args.out/'frame-end.png')
        interaction.close();sensor.close();renderer.close();server.shutdown();server.server_close()
        if foot_renderer:foot_renderer.close()
        if panel is not None and panel.poll() is None:
            panel.terminate();panel.wait(timeout=5)
        report={'version':args.profile+' interaction / VD24 physical body','profile':args.profile,'behaviors':args.behaviors,'target_red_xyz':[args.target_x,args.target_y,.24],'metrics':metrics.report(),'backend':args.backend,'sim_s':float(d.time),
            'wall_s':time.perf_counter()-wall0,'physics_wall_s':physics_wall,'control_wall_s':control_wall,
            'render_wall_s':render_wall,'video_fps':args.fps,'video_frames':frames,'inputs_received':received,
            'failure':failure,'final_phase':low.phase,'final_status':interaction.status,'max_tilt_rad':peak_tilt,
            'actuators':m.nu,'steps_completed':low.steps_completed,'controller_state_write_checks':state_checks,
            'root_writes_after_initialization':0,'max_actuator_force':peak_force.tolist(),'source_hashes':hashes,
            'observation':'robot RGB color markers; ideal simulated proprioception','input_source':args.input_source+' via external loopback text endpoint',
            'control':'one simulation-thread owner; asynchronous language intents; '+('contact-constrained inverse dynamics / prescribed straight gait' if args.profile in ('VD28','VD29') else 'explicit IK/PD'),
            'max_solver_constraint_residual':getattr(low,'solver_residual',None),
            'model':translator.model if translator else None,'events':interaction.events,'motion_events':low.events}
        (args.out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (args.out/'samples.json').write_text(json.dumps(samples,ensure_ascii=False),encoding='utf-8')
        if failure and hasattr(low,'solver_debug'):np.savez_compressed(args.out/'qp-at-failure.npz',**low.solver_debug)
        np.savez_compressed(args.out/'trajectory.npz',qpos=np.asarray(qpos),ctrl=np.asarray(ctrl))
        print(json.dumps({k:report[k] for k in ('version','backend','sim_s','wall_s','failure','final_phase','inputs_received')},ensure_ascii=False))
    return 1 if failure else 0

if __name__=='__main__':raise SystemExit(main())
