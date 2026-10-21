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
from opendafeiyu.tasks import TaskController, detect_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--scenario',choices=['stand','greet','walk','approach','cancel','loss','switch','look','push','freefall','interactive'],default='greet')
    p.add_argument('--video',action='store_true')
    p.add_argument('--live',action='store_true')
    p.add_argument('--collision-view',action='store_true')
    p.add_argument('--realtime',action='store_true')
    p.add_argument('--delivery',type=Path,default=DELIVERY)
    a=p.parse_args()
    if not 0<a.seconds<=180:p.error('seconds must be in (0,180]')
    a.out.mkdir(parents=True,exist_ok=False)
    m,xml,assets,binding=build_scene(a.delivery)
    if a.scenario=='freefall':
        import xml.etree.ElementTree as ET
        root=ET.fromstring(xml);ET.SubElement(root.find('option'),'flag',actuation='disable')
        xml=ET.tostring(root,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml,assets=assets)
        binding['negative_control']='Actuation disabled explicitly; contacts and gravity retained.'
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    (a.out/'binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2),'utf8')
    (a.out/'scene.xml').write_text(xml,'utf8')
    with zipfile.ZipFile(a.out/'model-assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('scene.xml',xml)
        for name,raw in assets.items():z.writestr(name,raw)
        z.write(a.delivery/'credits/THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md')
    sources=['src/opendafeiyu/bluefatfish.py','src/opendafeiyu/tasks.py','tools/run_bluefatfish.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    with zipfile.ZipFile(a.out/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    low=NativeController(m,d,binding)
    high=TaskController()
    command=np.zeros(7)
    if a.scenario in ('approach','cancel','loss','switch','look'):
        token=high.request(0.)
        high.accept(token,{'action':'look' if a.scenario=='look' else 'approach','target':'red'},0.,'scripted_test_input')
    camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-12;camera.distance=1.2
    camera.lookat[:]=[.12,0,.20]
    option=mujoco.MjvOption();option.geomgroup[3]=int(a.collision_view)
    if a.collision_view:option.skingroup[2]=0
    sensor_option=mujoco.MjvOption();sensor_option.geomgroup[3]=0
    sensor=mujoco.Renderer(m,240,320)
    renderer=mujoco.Renderer(m,720,1280)
    writer=viewer=None
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    if a.video:writer=imageio.get_writer(str(a.out/'simulation.mp4'),fps=30,codec='libx264',quality=7)
    import queue
    messages=queue.Queue()
    if a.live:
        from mujoco import viewer as mjviewer
        def key(code):
            if code in (32,67):messages.put('stop')
            if code==82:messages.put('red')
            if code==71:messages.put('greet')
            if code==68:messages.put('debug')
        viewer=mjviewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False)
        viewer.cam.azimuth=135;viewer.cam.elevation=-12;viewer.cam.distance=1.2;viewer.cam.lookat[:]=[.12,0,.2]
        viewer.opt.geomgroup[3]=int(a.collision_view)
        viewer.opt.skingroup[2]=not a.collision_view
    rgb=np.zeros((240,320,3),dtype=np.uint8)
    detections={};next_sensor=next_frame=next_gui=next_control=0.
    qposes=[];velocities=[];controls=[];samples=[];contact_pairs={}
    maximum_force=np.zeros(m.nu);max_tilt=0.;max_penetration=0.;steps=0;min_height=1.
    failure=None;intervened=False;events=[];end_frame_saved=False
    controller_state_checks=0
    wall0=time.perf_counter();physics_s=control_s=render_s=sensor_s=0.
    sim_command='idle'
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
                if msg=='stop':high.cancel(now);sim_command='idle'
                elif msg=='red':
                    sim_command='idle';low.wave_start=None;low.wave_done=False
                    token=high.request(now);high.accept(token,{'action':'approach','target':'red'},now,'keyboard')
                elif msg=='greet':
                    high.cancel(now,'greeting_requested');sim_command='greet';low.wave_start=None;low.wave_done=False
                elif msg=='debug' and viewer:
                    visible=not bool(viewer.opt.geomgroup[3]);viewer.opt.geomgroup[3]=visible;viewer.opt.skingroup[2]=not visible
            if now>=next_control:
                start=time.perf_counter()
                status=high.status;command=high.command(detections,now)
                status=high.status
                if a.scenario=='greet' or sim_command=='greet':status='arrived'
                elif a.scenario=='walk':status='approach';command[0]=.1
                if a.scenario=='freefall':d.ctrl[:]=0.
                else:
                    before=[v.copy() for v in (d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)]
                    low.update(command,status,walking=a.scenario not in ('stand','greet','push','look'))
                    for previous,current in zip(before,(d.qpos,d.qvel,d.mocap_pos,d.mocap_quat,d.xfrc_applied,d.qfrc_applied)):
                        if not np.array_equal(previous,current):raise RuntimeError('Controller wrote state or external forces')
                    controller_state_checks+=1
                control_s+=time.perf_counter()-start;next_control+=.02
            if a.scenario=='push':
                # Explicit external test disturbance; the controller never applies base wrenches.
                d.xfrc_applied[low.base,:3]=[0.,.6,0.] if 4.<=now<4.15 else [0.,0.,0.]
            start=time.perf_counter();mujoco.mj_step(m,d);physics_s+=time.perf_counter()-start;steps+=1
            maximum_force=np.maximum(maximum_force,np.abs(d.actuator_force))
            orientation=d.xmat[low.base].reshape(3,3)@R_FORWARD.T
            tilt=float(np.arccos(np.clip(orientation[2,2],-1,1)))
            max_tilt=max(max_tilt,tilt)
            head_z=float(d.body(binding['source_bones']['head']).xpos[2]);min_height=min(min_height,head_z)
            for c in d.contact:
                pair=' / '.join(sorted([m.geom(c.geom1).name,m.geom(c.geom2).name]))
                v=contact_pairs.setdefault(pair,{'points':0,'max_penetration_m':0.})
                v['points']+=1;v['max_penetration_m']=max(v['max_penetration_m'],-float(c.dist))
                max_penetration=max(max_penetration,-float(c.dist))
            if i%20==0:
                qposes.append(d.qpos.copy());velocities.append(d.qvel.copy());controls.append(d.ctrl.copy())
                samples.append(dict(sim_s=float(d.time),wall_s=time.perf_counter()-wall0,status=status,phase=low.phase,
                    root=d.qpos[:3].tolist(),tilt_rad=tilt,feet_contact=low.contacts(),head_z=head_z,
                    detections={k:vars(v) for k,v in detections.items()},
                    feet=[d.xpos[b].tolist() for b in low.feet],com=d.subtree_com[low.base].tolist()))
            if now>=next_frame and (writer or i==0 or (now>=a.seconds-.002 and not end_frame_saved)):
                start=time.perf_counter()
                camera.lookat[:]=[d.qpos[0]+.12,d.qpos[1],.20]
                renderer.update_scene(d,camera=camera,scene_option=option)
                frame=Image.fromarray(renderer.render());frame.paste(Image.fromarray(rgb),(944,468))
                draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,96),fill=(20,29,43))
                draw.text((18,8),'大肥鱼 · 真实 MuJoCo 动力学 | 原模型 / 原画：木子恶魔',font=font,fill='white')
                draw.text((18,37),f'{a.scenario} | {status} / {low.phase} | 模拟 {now:.2f}s | 21轴 · 自由基座 · 接触求解',font=font,fill=(140,214,255))
                draw.text((18,65),'VD1：质量 / 惯量为仿真估计；模型授权待澄清；本地验证素材',font=font,fill='white')
                draw.text((944,439),'机器人实际 RGB 输入',font=font,fill=(30,40,60))
                if i==0:frame.save(a.out/'frame-start.png')
                if now>=a.seconds-.04:
                    frame.save(a.out/'frame-end.png');end_frame_saved=True
                if writer:writer.append_data(np.asarray(frame))
                next_frame+=1/30;render_s+=time.perf_counter()-start
            if viewer and now>=next_gui:
                if not viewer.is_running():break
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'BlueFatFish / native physics\nControls\nState\nModel/art\n',
                    'Estimated dynamics - local verification\nR approach / G greet / C stop / D collisions\n'+status+' / '+low.phase+'\nMuzi Emo (original embedded credit retained)\n'))
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
        if writer:writer.close()
        if viewer:viewer.close()
    wall_s=time.perf_counter()-wall0
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
        max_ik_error_m=low.max_ik_error,events=high.events+low.events+events,failure=failure,
        source_hashes=hashes,video_timing='simulation_time_30fps' if writer else None,
        controller='explicit slow IK/PD with measured foot contact; not learned; camera pixels for targets; straight short steps, head aiming, no general turning planner',
        model_author='木子恶魔',model_asset_license='embedded restrictive VRM fields retained; no redistribution permission inferred')
    (a.out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('contact_pairs','max_actuator_force','events','source_hashes')},ensure_ascii=False,indent=2))
    return int(failure is not None or low.phase=='failed')


if __name__=='__main__':raise SystemExit(main())
