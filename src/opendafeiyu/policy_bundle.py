"""Versioned, hash-bound learned-policy loading; never silently fall back to a controller."""
from pathlib import Path
import hashlib,json
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(directory, model_path, home_path, joint_names, *, allow_experimental=False):
    root=Path(directory).resolve()
    spec=json.loads((root/'policy.json').read_text('utf8'))
    import mujoco
    if spec.get('physics_engine_version')!=mujoco.__version__:raise ValueError('MuJoCo version mismatch')
    if spec.get('schema')!='whalechan-policy/1':raise ValueError('Unknown policy schema')
    expected={'observation_schema':'proprio-phase-v1','observation_size':80,'action_size':23,
              'physics_dt':.001,'control_steps':20,'delay_steps':2,'action_scale':.6,'previous_action_weight':.8}
    for key,value in expected.items():
        if spec.get(key)!=value:raise ValueError('Policy contract mismatch: '+key)
    if spec.get('joint_names')!=list(joint_names):raise ValueError('Actuator joint order mismatch')
    if not isinstance(spec.get('period_ticks'),int) or isinstance(spec['period_ticks'],bool) or not 20<=spec['period_ticks']<=1000:
        raise ValueError('Invalid phase period')
    if spec.get('model_sha256')!=sha256(model_path):raise ValueError('Physical body fingerprint mismatch')
    if spec.get('home_sha256')!=sha256(home_path):raise ValueError('Initial state fingerprint mismatch')
    name=spec.get('policy_file')
    if not isinstance(name,str) or Path(name).is_absolute():raise ValueError('Invalid policy path')
    policy=(root/name).resolve()
    if not policy.is_relative_to(root) or policy==root:raise ValueError('Policy path escapes bundle')
    if spec.get('policy_sha256')!=sha256(policy):raise ValueError('Policy fingerprint mismatch')
    qualification=spec.get('qualification')
    if qualification not in ('experimental','evaluated'):raise ValueError('Unknown qualification')
    if qualification=='experimental' and not allow_experimental:
        raise ValueError('Experimental policy requires explicit opt-in; walking is not qualified')
    if qualification=='evaluated':
        # Require a bound report instead of trusting a status string alone.
        report_path=(root/'evaluation.json')
        if spec.get('evaluation_sha256')!=sha256(report_path):raise ValueError('Evaluation fingerprint mismatch')
        report=json.loads(report_path.read_text('utf8'))
        if report.get('policy_sha256')!=spec['policy_sha256'] or report.get('model_sha256')!=spec['model_sha256'] or report.get('passed') is not True:
            raise ValueError('Evaluation does not qualify this policy/body')
    return spec,policy


def validate_action(action):
    action=np.asarray(action,dtype=np.float32)
    if action.shape!=(23,) or not np.isfinite(action).all():
        raise ValueError('Policy returned a nonfinite or incorrectly sized action')
    return np.clip(action,-1,1)


class TorchPolicy:
    def __init__(self,directory,model_path,home_path,joint_names,*,allow_experimental=False):
        self.spec,path=validate_bundle(directory,model_path,home_path,joint_names,allow_experimental=allow_experimental)
        import torch
        self.torch=torch
        self.model=torch.jit.load(str(path),map_location='cpu').eval()
        self(np.zeros(80,dtype=np.float32))  # Validate actual export I/O before control starts.

    def __call__(self,observation):
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape!=(80,) or not np.isfinite(observation).all():raise ValueError('Invalid policy observation')
        with self.torch.no_grad():output=self.model(self.torch.from_numpy(observation)[None])
        if tuple(output.shape)!=(1,23):raise ValueError('Invalid policy output shape')
        return validate_action(output[0].cpu().numpy())
