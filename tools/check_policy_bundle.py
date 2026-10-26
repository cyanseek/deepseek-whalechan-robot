"""Check an experimental policy/body binding without launching motion or networking."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
from opendafeiyu.policy_bundle import validate_bundle,TorchPolicy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('--body',type=Path,required=True)
    p.add_argument('--allow-experimental',action='store_true');p.add_argument('--infer',action='store_true');a=p.parse_args()
    try:
        m=mujoco.MjModel.from_xml_string((a.body/'robot.xml').read_text())
        names=[m.joint(int(i)).name for i in m.actuator_trnid[:,0]]
        spec,_=validate_bundle(a.bundle,a.body/'robot.xml',a.body/'home.npz',names,allow_experimental=a.allow_experimental)
        if a.infer:TorchPolicy(a.bundle,a.body/'robot.xml',a.body/'home.npz',names,allow_experimental=a.allow_experimental)
        result={'status':'binding_and_inference_checked' if a.infer else 'binding_checked','qualification':spec['qualification'],'walking_validated_here':False}
    except Exception as e:
        print(json.dumps({'status':'rejected','error_type':type(e).__name__,'reason':str(e)}));return 2
    print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
