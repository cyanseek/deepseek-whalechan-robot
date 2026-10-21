#!/usr/bin/env python3
"""Run the delivered skinned character as an independent dynamic robot."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from opendafeiyu.bluefatfish import build_scene, NativeController, R_FORWARD, DELIVERY
from opendafeiyu.bluefatfish_v2 import EmbodiedController, build_scene
from opendafeiyu.bluefatfish_expressions import Expressions
from opendafeiyu.tasks import TaskController, detect_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--scenario',choices=['stand','greet','walk','approach','cancel','loss','switch','look','push','freefall','interactive','jump','turn','side','expressions','showcase','tail','tail-cancel','coordinated','coord-cancel'],default='showcase')
    p.add_argument('--profile',choices=['VD2','VD21','VD22','VD23','VD24'],default='VD2')
    p.add_argument('--balance-off',action='store_true')
    p.add_argument('--push-force',type=float,default=1.2)
    p.add_argument('--push-axis',choices=['x','y'],default='y')
    p.add_argument('--video',action='store_true')
    p.add_argument('--fps',type=int,choices=[30,60],default=30,help='Native recording rate; playback remains one simulated second per second')
    p.add_argument('--presentation',action='store_true',help='Continuous native physics recording with coordinated director/detail cameras')
    p.add_argument('--live',action='store_true')
    p.add_argument('--collision-view',action='store_true')
    p.add_argument('--realtime',action='store_true')
    p.add_argument('--delivery',type=Path,default=DELIVERY)
    a=p.parse_args()
    if not 0<a.seconds<=180:p.error('seconds must be in (0,180]')
    a.out.mkdir(parents=True,exist_ok=False)
    builder,controller=build_scene,EmbodiedController
    if a.profile=='VD24':
        from opendafeiyu.bluefatfish_v24 import build_scene as builder, LivelyController as controller
    elif a.profile=='VD23':
        from opendafeiyu.bluefatfish_v23 import build_scene as builder, SmallStepController as controller
    elif a.profile=='VD22':
        from opendafeiyu.bluefatfish_v22 import build_scene as builder, CoordinatedController as controller
    elif a.profile=='VD21':
        from opendafeiyu.bluefatfish_v21 import build_scene as builder, ExpressiveController as controller
    elif a.scenario.startswith('tail'):p.error('Tail gestures require --profile VD21')
    if a.scenario in ('coordinated','coord-cancel') and a.profile not in ('VD22','VD23','VD24'):p.error('Coordinated demo requires --profile VD22 or VD23')
    m,xml,assets,binding=builder(a.delivery)
    binding['limitations']=[x for x in binding['limitations'] if 'Expression morphs' not in x]
    if a.scenario=='freefall':
        import xml.etree.ElementTree as ET
        root=ET.fromstring(xml);ET.SubElement(root.find('option'),'flag',actuation='disable')
        xml=ET.tostring(root,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets=assets)
        binding['negative_control']='Actuation disabled explicitly; contacts and gravity retained.'
    face_type=Expressions
    if a.profile=='VD24':
        from opendafeiyu.bluefatfish_v24 import CheerfulExpressions as face_type
    face=face_type(m,a.delivery,assets.get('expression-cutout-map.npz'))
    binding['expressions']=face.report
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    (a.out/'binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2),'utf8')
    (a.out/'scene.xml').write_text(xml,'utf8')
    with zipfile.ZipFile(a.out/'model-assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('scene.xml',xml)
        for name,raw in assets.items():z.writestr(name,raw)
        z.write(a.delivery/'credits/THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md')
    sources=['src/opendafeiyu/bluefatfish.py','src/opendafeiyu/bluefatfish_v2.py','src/opendafeiyu/bluefatfish_expressions.py','src/opendafeiyu/tasks.py','tools/run_bluefatfish_v2.py']
    if a.profile in ('VD21','VD22','VD23','VD24'):sources.append('src/opendafeiyu/bluefatfish_v21.py')
    if a.profile in ('VD22','VD23','VD24'):sources.append('src/opendafeiyu/bluefatfish_v22.py')
    if a.profile in ('VD23','VD24'):sources.extend(['src/opendafeiyu/bluefatfish_v23.py','assets/characters/bluefatfish-v23/clothing.json'])
    if a.profile=='VD24':sources.extend(['src/opendafeiyu/bluefatfish_v24.py','assets/characters/bluefatfish-v24/clothing.json'])
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    with zipfile.ZipFile(a.out/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    low=controller(m,d,binding,balance=not a.balance_off)
    demo=None
    if a.scenario in ('coordinated','coord-cancel'):
        from opendafeiyu.bluefatfish_v22 import CoordinatedDemo
        demo=CoordinatedDemo(walk_steps=16,turn_steps=6) if a.profile=='VD24' else (CoordinatedDemo(walk_steps=12,turn_steps=6) if a.profile=='VD23' else CoordinatedDemo())
    high=TaskController()
    command=np.zeros(7)
    if a.scenario in ('approach','cancel','loss','switch','look'):
        token=high.request(0.)
        high.accept(token,{'action':'look' if a.scenario=='look' else 'approach','target':'red'},0.,'scripted_test_input')
    camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-12;camera.distance=1.2
    camera.lookat[:]=[.12,0,.20]
    if a.scenario.startswith('tail'):camera.azimuth=-65;camera.distance=.75;camera.lookat[:]=[-.065,0,.18]
    option=mujoco.MjvOption();option.geomgroup[3]=int(a.collision_view)
    if a.presentation:
        for name in ('target_red','target_green'):
            m.geom_group[m.geom_bodyid==m.body(name).id]=4
        option.geomgroup[4]=0  # Director view only; robot camera retains both markers.
    if a.collision_view:option.skingroup[2]=0
    sensor_option=mujoco.MjvOption();sensor_option.geomgroup[3]=0;sensor_option.geomgroup[4]=1
    sensor=mujoco.Renderer(m,240,320)
    renderer=mujoco.Renderer(m,720,1280)
    detail_renderer=mujoco.Renderer(m,240,320) if a.presentation else None
    detail_camera=mujoco.MjvCamera()
    if a.presentation:camera.azimuth=155;camera.elevation=-8;camera.distance=.76
    writer=viewer=None
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    if a.video:writer=imageio.get_writer(str(a.out/'simulation.mp4'),fps=a.fps,codec='libx264',quality=7)
    import queue
    messages=queue.Queue()
    if a.live:
        from mujoco import viewer as mjviewer
        def key(code):
            if code in (32,67):messages.put('stop')
            if code==82:messages.put('red')
            if code==71:messages.put('greet')
            if code==68:messages.put('debug')
            if code==74:messages.put('jump')
            if code==87:messages.put('forward')
            if code==65:messages.put('left')
            if code==83:messages.put('back')
            if code==69:messages.put('expression')
            if code==84:messages.put('tail')
            if code==86:messages.put('view')
        viewer=mjviewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False)
        viewer.cam.azimuth=135;viewer.cam.elevation=-12;viewer.cam.distance=1.2;viewer.cam.lookat[:]=[.12,0,.2]
        if a.scenario.startswith('tail'):viewer.cam.azimuth=-65;viewer.cam.distance=.75;viewer.cam.lookat[:]=[-.065,0,.18]
        viewer.opt.geomgroup[3]=int(a.collision_view)
        viewer.opt.skingroup[2]=not a.collision_view
        if a.presentation:viewer.opt.geomgroup[4]=0
    rgb=np.zeros((240,320,3),dtype=np.uint8)
    detections={};next_sensor=next_frame=next_gui=next_control=0.
    qposes=[];velocities=[];controls=[];samples=[];contact_pairs={}
    maximum_force=np.zeros(m.nu);max_tilt=0.;max_penetration=0.;steps=0;min_height=1.
    failure=None;intervened=False;events=[];end_frame_saved=False
    controller_state_checks=0
    wall0=time.perf_counter();physics_s=control_s=render_s=sensor_s=0.
    sim_command='idle';manual_emotion='neutral';jump_sent=False;jump_pending=False;tail_sent=False;tail_pending=False;view_index=0
    airborne_s=0.;longest_air_s=0.;air_start=None;peak_com_z=-1.;ready_com_z=None;min_soles_clearance=0.
    peak_air_clearance=0.;jump_metrics=[];air_phase=None;air_peak=0.
    try:
        for i in range(round(a.seconds/m.opt.timestep)):
            now=float(d.time)
            if now>=next_sensor:
                start=time.perf_counter();sensor.update_scene(d,camera='robot_camera',scene_option=sensor_option)
                rgb=sensor.render();detections=detect_targets(rgb);sensor_s+=time.perf_counter()-start;next_sensor+=.1
            if a.scenario in ('cancel','loss','switch') and now>=8. and not intervened:
                intervened=True
                if a.scenario=='cancel':high.cancel(now)
                elif a.scenario=='switch':
                    token=high.request(now);high.accept(token,{'action':'approach','target':'green'},now,'scripted_test_input')
                else:
                    d.mocap_pos[m.body_mocapid[m.body('target_red').id]]=[-2,0,.24]
                    events.append({'time':now,'event':'external_marker_moved'})
            while not messages.empty():
                msg=messages.get()
                if msg=='stop':
                    high.cancel(now);sim_command='idle';low.cancel();jump_pending=False;tail_pending=False
                    if demo:demo.cancel(low)
                elif msg=='red':
                    sim_command='idle';low.wave_start=None;low.wave_done=False
                    token=high.request(now);high.accept(token,{'action':'approach','target':'red'},now,'keyboard')
                elif msg=='greet':
                    high.cancel(now,'greeting_requested');sim_command='greet';low.wave_start=None;low.wave_done=False
                elif msg=='debug' and viewer:
                    visible=not bool(viewer.opt.geomgroup[3]);viewer.opt.geomgroup[3]=visible;viewer.opt.skingroup[2]=not visible
                elif msg=='jump':
                    high.cancel(now,'jump_requested');sim_command='idle';jump_pending=True
                elif msg in ('forward','left','back'):
                    high.cancel(now,'manual_motion');sim_command=msg
                elif msg=='expression':
                    emotions=['neutral','happy','angry','sad','relaxed']
                    manual_emotion=emotions[(emotions.index(manual_emotion)+1)%len(emotions)]
                elif msg=='tail' and a.profile in ('VD21','VD22','VD23','VD24'):
                    high.cancel(now,'tail_requested');sim_command='idle';tail_pending=True
                elif msg=='view' and viewer:
                    view_index=(view_index+1)%3
                    viewer.cam.azimuth=[135,-90,0][view_index];viewer.cam.distance=.75
                    viewer.cam.lookat[:]=[d.qpos[0]-.04,d.qpos[1],.18]
            if i%10==0:
                start=time.perf_counter()
                status=high.status;command=high.command(detections,now)
                status=high.status
                if a.scenario=='greet' or sim_command=='greet':status='arrived'
                elif a.scenario in ('walk','turn','side'):
                    status='approach';command[:3]=[.12 if a.scenario=='walk' else .015, .06 if a.scenario=='side' else 0., .35 if a.scenario=='turn' else 0.]
                if sim_command in ('forward','left','back'):
                    status='approach';command[:3]=[.12 if sim_command=='forward' else (-.06 if sim_command=='back' else .015),0,.35 if sim_command=='left' else 0]
                if a.scenario=='showcase':
                    if 3<now<12:status='approach';command[:3]=[.12,0,0]
                    elif 21<now<31:status='approach';command[:3]=[.015,0,.35]
                    elif now>=34:status='arrived'
                if demo:
                    command,status=demo.update(low)
                    if low.jump_preparing or demo.stage=='jump':jump_sent=True
                    if a.scenario=='coord-cancel' and now>=11 and not intervened:
                        demo.cancel(low);command[:]=0;status='stopped';intervened=True
                if a.scenario in ('jump','showcase') and i>=round((3. if a.scenario=='jump' else 16.)/m.opt.timestep) and not jump_sent:
                    jump_sent=low.request_jump()
                if jump_pending and low.request_jump():jump_sent=True;jump_pending=False
                if a.scenario.startswith('tail') and now>=3 and not tail_sent:
                    tail_sent=low.request_tail()
                if tail_pending and low.request_tail():tail_pending=False
                if a.scenario=='tail-cancel' and now>=5 and not intervened:
                    low.cancel();intervened=True
                if a.scenario=='freefall':d.ctrl[:]=0.
                else:
                    before=[v.copy() for v in (d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)]
                    if a.profile=='VD24':low.set_precision_motion(high.target is not None)
                    low.update(command,status,walking=a.scenario not in ('stand','greet','push','look','jump','expressions'))
                    for previous,current in zip(before,(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)):
                        if not np.array_equal(previous,current):raise RuntimeError('Controller wrote state or external forces')
                    controller_state_checks+=1
                emotion=low.emotion if low.phase in ('crouch','flight','recover') or low.wave_start is not None else manual_emotion
                if a.profile in ('VD22','VD23','VD24'):emotion=low.expression if low.activity!='idle' or manual_emotion=='neutral' else manual_emotion
                if a.scenario=='expressions':emotion=['neutral','happy','angry','sad','relaxed'][int(now//2.5)%5]
                face.update(now,emotion)
                control_s+=time.perf_counter()-start;next_control+=.01
            if a.scenario=='push':
                # Explicit external test disturbance; the controller never applies base wrenches.
                d.xfrc_applied[low.base,:3]=0
                if 4.<=now<4.15:d.xfrc_applied[low.base,0 if a.push_axis=='x' else 1]=a.push_force
            start=time.perf_counter();mujoco.mj_step(m,d);physics_s+=time.perf_counter()-start;steps+=1
            maximum_force=np.maximum(maximum_force,np.abs(d.actuator_force))
            orientation=d.xmat[low.base].reshape(3,3)@R_FORWARD.T
            tilt=float(np.arccos(np.clip(orientation[2,2],-1,1)))
            max_tilt=max(max_tilt,tilt)
            if low.phase=='ready' and ready_com_z is None:ready_com_z=float(d.subtree_com[low.base,2])
            if jump_sent:
                peak_com_z=max(peak_com_z,float(d.subtree_com[low.base,2]))
                floor=m.geom('floor').id
                ground=any(c.geom1==floor or c.geom2==floor for c in d.contact)
                if not ground and low.phase in ('launch','flight','recover'):
                    if air_start is None:air_start=float(d.time);air_phase=low.phase;air_peak=0.
                    airborne_s+=m.opt.timestep;longest_air_s=max(longest_air_s,float(d.time)-air_start+m.opt.timestep)
                    clearance=[]
                    for side in ('left','right'):
                        gi=m.geom(side+'_sole').id
                        extent=float(np.abs(d.geom_xmat[gi].reshape(3,3)[2])@m.geom_size[gi])
                        clearance.append(float(d.geom_xpos[gi,2])-extent)
                    peak_air_clearance=max(peak_air_clearance,min(clearance))
                    air_peak=max(air_peak,min(clearance))
                elif air_start is not None:
                    jump_metrics.append(dict(start_s=air_start,end_s=float(d.time),duration_s=float(d.time)-air_start,
                                             start_phase=air_phase,peak_both_soles_clearance_m=air_peak))
                    air_start=None
            head_z=float(d.body(binding['source_bones']['head']).xpos[2]);min_height=min(min_height,head_z)
            for c in d.contact:
                pair=' / '.join(sorted([m.geom(c.geom1).name,m.geom(c.geom2).name]))
                v=contact_pairs.setdefault(pair,{'points':0,'max_penetration_m':0.})
                v['points']+=1;v['max_penetration_m']=max(v['max_penetration_m'],-float(c.dist))
                max_penetration=max(max_penetration,-float(c.dist))
            if i%10==0:
                qposes.append(d.qpos.copy());velocities.append(d.qvel.copy());controls.append(d.ctrl.copy())
                samples.append(dict(sim_s=float(d.time),wall_s=time.perf_counter()-wall0,status=status,phase=low.phase,
                    root=d.qpos[:3].tolist(),tilt_rad=tilt,feet_contact=low.contacts(),head_z=head_z,
                    detections={k:vars(v) for k,v in detections.items()},
                    feet=[d.xpos[b].tolist() for b in low.feet],com=d.subtree_com[low.base].tolist(),
                    expression=face.weights.copy(),feedback_rad=low.feedback.tolist(),heading=low.heading,
                    coordination=getattr(low,'activity',None),demo_stage=demo.stage if demo else None,motion_mode=getattr(low,'motion_mode',None),
                    tail_target=[float(d.ctrl[m.actuator(n).id]) for n in ['tail_yaw','tail_pitch','tail_bend']] if a.profile in ('VD21','VD22','VD23','VD24') else None))
            if now>=next_frame and (writer or i==0 or (now>=a.seconds-.002 and not end_frame_saved)):
                start=time.perf_counter()
                camera.lookat[:]=[d.qpos[0]+.12,d.qpos[1],.20]
                if a.scenario.startswith('tail'):camera.lookat[:]=[d.qpos[0]-.065,d.qpos[1],.18]
                if a.presentation:
                    stage=demo.stage if demo else getattr(low,'activity',a.scenario)
                    angles={'settle':155,'hello':155,'walk':100,'stop':100,'jump':90,'landed':115,'turn':125,'stop_turn':135,'celebrate':150,'rest':155,'complete':155}
                    if a.profile=='VD24':angles.update(walk=140,stop=140,jump=115)
                    view_orientation=d.xmat[low.base].reshape(3,3)@R_FORWARD.T
                    view_yaw=float(np.arctan2(view_orientation[1,0],view_orientation[0,0])) if a.profile in ('VD23','VD24') else 0.
                    desired_azimuth=angles.get(stage,140)+np.degrees(view_yaw)
                    camera.azimuth+=((desired_azimuth-camera.azimuth+180)%360-180)*(1-(1-.045)**(30/a.fps))
                    camera.lookat[:]=[d.qpos[0]-.045*np.cos(view_yaw),d.qpos[1]-.045*np.sin(view_yaw),.19]
                renderer.update_scene(d,camera=camera,scene_option=option)
                frame=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(frame)
                if a.presentation:
                    titles={'settle':'准备站稳','hello':'问好 · 挥手与摆尾','walk':'行走 · 手臂与尾巴跟随步态','stop':'收步 · 平滑停稳','jump':'准备 · 小跳 · 落地恢复','landed':'落地后的舒展与回应','turn':'转向 · 视线与尾巴协调','stop_turn':'完成转向 · 收步','celebrate':'挥手回应 · 摆尾 · 微笑','rest':'自然回到休息姿态','complete':'动作完成','cancelled':'取消 · 接触感知停稳'}
                    if a.profile in ('VD23','VD24'):titles['walk']='轻快小碎步 · 手臂与鲸尾协调'
                    draw.rectangle((0,0,1280,73),fill=(20,29,43))
                    draw.text((22,9),'OpenDafeiyu · '+binding['revision']+' · 协调动作演示',font=font,fill='white')
                    draw.text((22,39),titles.get(stage,stage),font=font,fill=(160,220,255))
                    draw.text((916,15),f'真实物理仿真 · 编排任务 · {now:05.2f}s',font=font,fill=(210,225,240))
                    show_face=stage in ('hello','landed','celebrate','rest','complete')
                    detail_camera.azimuth=(165 if show_face else 0)+np.degrees(view_yaw);detail_camera.elevation=-5 if show_face else -18
                    detail_camera.distance=.38 if show_face else .33
                    detail_offset=.015 if show_face else .14
                    detail_camera.lookat[:]=[d.qpos[0]-detail_offset*np.cos(view_yaw),d.qpos[1]-detail_offset*np.sin(view_yaw),.265 if show_face else .075]
                    show_feet=a.profile=='VD24' and stage in ('jump','prepare','crouch','recover')
                    if show_feet:
                        detail_camera.azimuth=100+np.degrees(view_yaw);detail_camera.elevation=-5
                        detail_camera.distance=.30;detail_camera.lookat[:]=[d.qpos[0],d.qpos[1],.045]
                    detail_renderer.update_scene(d,camera=detail_camera,scene_option=option)
                    frame.paste(Image.fromarray(detail_renderer.render()),(944,450))
                    draw.rectangle((944,422,1264,450),fill=(20,29,43))
                    draw.text((954,425),'脚部离地 · 同一仿真状态' if show_feet else ('表情细节 · 同一仿真状态' if show_face else '尾部协调 · 同一仿真状态'),font=font,fill='white')
                    if show_feet:
                        draw.text((954,694),f'离地峰值 {peak_air_clearance*1000:.1f} mm · 原速',font=font,fill=(20,29,43))
                else:
                    frame.paste(Image.fromarray(rgb),(944,468))
                    draw.rectangle((0,0,1280,96),fill=(20,29,43))
                    draw.text((18,8),'OpenDafeiyu · 大肥鱼具身仿真 · '+binding['revision'],font=font,fill='white')
                    draw.text((18,37),f'{a.scenario} | {status} / {low.phase} | 模拟 {now:.2f}s | {m.nu}轴 · 自由基座 · 接触求解',font=font,fill=(140,214,255))
                    draw.text((18,65),f'仿真估计动力学 | 表情：{emotion} | 腾空 {longest_air_s*1000:.0f} ms | 双脚最低离地峰值 {peak_air_clearance*1000:.1f} mm',font=font,fill='white')
                    draw.text((944,439),'机器人实际 RGB 输入',font=font,fill=(30,40,60))
                if i==0:frame.save(a.out/'frame-start.png')
                if now>=a.seconds-.04:
                    frame.save(a.out/'frame-end.png');end_frame_saved=True
                if writer:writer.append_data(np.asarray(frame))
                next_frame+=1/a.fps;render_s+=time.perf_counter()-start
            if viewer and now>=next_gui:
                if not viewer.is_running():break
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'OpenDafeiyu / '+binding['revision']+'\nControls\nMore\nState\n',
                    'Native physics simulation\nR approach / G greet / C stop / J jump\nW forward / A turn / S back / E expression / T tail / V camera / D collisions\n'+status+' / '+low.phase+'\n'))
                viewer.sync();next_gui+=1/30
            if a.realtime or a.live:
                delay=wall0+float(d.time)-time.perf_counter()
                if delay>0:time.sleep(delay)
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite dynamics')
            if tilt>.9 or head_z<.10:
                failure={'reason':'fallen','sim_s':float(d.time)}
                break
            if low.phase=='failed':
                failure={'reason':'landing_timeout','sim_s':float(d.time)}
                break
    except Exception as exc:
        failure={'reason':type(exc).__name__,'detail':str(exc),'sim_s':float(d.time)}
    finally:
        sensor.close();renderer.close()
        if detail_renderer:detail_renderer.close()
        if writer:writer.close()
        if viewer:viewer.close()
    wall_s=time.perf_counter()-wall0
    stable_final=bool(samples and all(s['tilt_rad']<.10 and all(s['feet_contact']) for s in samples if s['sim_s']>float(d.time)-1.))
    if a.profile=='VD24' and failure is None:
        unsupported=[p for p in contact_pairs if 'floor' in p and p not in ('floor / left_sole','floor / right_sole')]
        if unsupported:failure=dict(reason='nonsole_ground_contact',pairs=unsupported)
        elif a.scenario=='approach' and (high.status!='arrived' or not stable_final):
            failure=dict(reason='approach_acceptance_not_met',status=high.status,stable=stable_final)
    if air_start is not None:
        jump_metrics.append(dict(start_s=air_start,end_s=float(d.time),duration_s=float(d.time)-air_start+m.opt.timestep,
                                 start_phase=air_phase,peak_both_soles_clearance_m=air_peak,incomplete=True))
    launch_intervals=[x for x in jump_metrics if x['start_phase'] in ('launch','flight')]
    launch_air_s=max((x['duration_s'] for x in launch_intervals),default=0.)
    launch_clearance=max((x['peak_both_soles_clearance_m'] for x in launch_intervals),default=0.)
    jump_success=bool(jump_sent and low.jump_airborne and low.jump_done and stable_final and longest_air_s>.012 and peak_air_clearance>.0005)
    if a.profile in ('VD22','VD23','VD24'):jump_success=jump_success and launch_air_s>.012 and launch_clearance>.0005
    if a.scenario in ('jump','showcase') and not jump_success and failure is None:
        failure=dict(reason='jump_acceptance_not_met',sim_s=float(d.time),recovered=low.jump_done,stable_final=stable_final)
    if demo and failure is None:
        expected='cancelled' if a.scenario=='coord-cancel' else 'complete'
        if demo.stage!=expected or not stable_final or (not demo.cancelled and not jump_success):
            failure=dict(reason='coordinated_acceptance_not_met',stage=demo.stage,jump=jump_success,stable=stable_final)
    np.savez_compressed(a.out/'trajectory.npz',qpos=qposes,qvel=velocities,ctrl=controls)
    (a.out/'samples.json').write_text(json.dumps(samples,ensure_ascii=False),'utf8')
    report=dict(scenario=a.scenario,engine=mujoco.__version__,python=platform.python_version(),
        sim_s=float(d.time),wall_s=wall_s,physics_s=physics_s,control_s=control_s,sensor_s=sensor_s,render_s=render_s,
        steps=steps,nq=m.nq,nv=m.nv,nu=m.nu,nmocap=m.nmocap,character_mocap=0,
        root_writes_after_initialization=0,controller_state_write_checks=controller_state_checks,
        root_write_evidence='Live qpos/qvel/mocap/external-force arrays compared before/after every controller update; all identical.',
        max_tilt_rad=max_tilt,min_head_z=min_height,max_contact_penetration_m=max_penetration,
        contact_pairs=contact_pairs,max_actuator_force={m.actuator(i).name:float(v) for i,v in enumerate(maximum_force)},
        final_root=d.qpos[:3].tolist(),final_status=status,final_phase=low.phase,steps_completed=low.steps_completed,
        max_ik_error_m=low.max_ik_error,events=high.events+low.events+events+(demo.events if demo else []),failure=failure,
        source_hashes=hashes,video_timing=f'simulation_time_{a.fps}fps' if writer else None,
        controller=a.profile+' prescribed contact-aware IK/PD; ideal IMU attitude + gyro feedback; VD24 flight landing uses model-based COM proprioception; bounded movement, original force limits; not learned',
        presentation=dict(enabled=a.presentation,source='continuous native integration',director_is_robot_observation=False,
                          follows_body_heading=a.profile in ('VD23','VD24'),
                          detail='same live physics state',runtime_render_override='target marker geom group 4 hidden from director only; robot sensor unchanged' if a.presentation else None),
        trajectory_hz=100,physics_hz=1000,control_hz=100,
        base_model_art_author='木子恶魔',project_contribution=binding['project_contribution'],
        balance_enabled=not a.balance_off,arguments=vars(a)|{'out':str(a.out),'delivery':str(a.delivery)},
        jump=dict(requested=jump_sent,takeoff=low.jump_airborne,recovered=low.jump_done,accepted=jump_success,stable_final_second=stable_final,
                  total_airborne_s=airborne_s,longest_airborne_s=longest_air_s,
                  air_intervals=jump_metrics,launch_longest_airborne_s=launch_air_s,launch_peak_both_soles_clearance_m=launch_clearance,
                  interval_definition='No floor contact at 1 kHz, classified by controller phase at interval start; recovery intervals are not accepted as launch flight in VD22/VD23.',
                  peak_com_rise_m=peak_com_z-ready_com_z if jump_sent and ready_com_z is not None else 0.,
                  peak_both_soles_clearance_m=peak_air_clearance),
        final_tilt_rad=tilt,expression_mapping=face.report,
        model_asset_license='embedded restrictive VRM fields retained; no redistribution permission inferred')
    if a.profile in ('VD21','VD22','VD23','VD24'):
        track=np.asarray(qposes)
        names=['tail_yaw','tail_pitch','tail_bend']
        report['tail']={name:dict(min_rad=float(track[:,m.joint(name).qposadr[0]].min()),
            max_rad=float(track[:,m.joint(name).qposadr[0]].max()),
            final_rad=float(d.joint(name).qpos[0]),max_torque_Nm=float(maximum_force[m.actuator(name).id])) for name in names}
        report['tail']['completed']=any(e['event']=='tail_gesture_completed' for e in low.events)
        report['tail']['cancelled']=any(e['event']=='tail_gesture_cancelled' for e in low.events)
        report['tail']['stable_final_second']=stable_final
        if a.scenario.startswith('tail'):
            expected='cancelled' if a.scenario=='tail-cancel' else 'completed'
            tail_contact=any('tail' in name for name in contact_pairs)
            report['tail']['accepted']=bool(report['tail'][expected] and stable_final and not tail_contact)
            if not report['tail']['accepted'] and failure is None:
                failure=dict(reason='tail_acceptance_not_met',sim_s=float(d.time))
                report['failure']=failure
    (a.out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('contact_pairs','max_actuator_force','events','source_hashes')},ensure_ascii=False,indent=2))
    return int(failure is not None or low.phase=='failed')


if __name__=='__main__':raise SystemExit(main())
