import json,mujoco
import numpy as np
import pytest
from opendafeiyu.policy_bundle import sha256,validate_bundle,validate_action

def bundle(tmp_path):
    # Byte fixtures validate the manifest boundary; they are not trained policies.
    model=tmp_path/'body.xml';model.write_text('body fixture')
    home=tmp_path/'home.npz';home.write_bytes(b'home fixture')
    policy=tmp_path/'policy.ts';policy.write_bytes(b'policy fixture')
    joints=['joint'+str(i) for i in range(23)]
    spec=dict(schema='whalechan-policy/1',physics_engine_version=mujoco.__version__,observation_schema='proprio-phase-v1',observation_size=80,action_size=23,
              physics_dt=.001,control_steps=20,delay_steps=2,action_scale=.6,previous_action_weight=.8,period_ticks=160,
              joint_names=joints,model_sha256=sha256(model),home_sha256=sha256(home),policy_file='policy.ts',policy_sha256=sha256(policy),qualification='experimental')
    (tmp_path/'policy.json').write_text(json.dumps(spec))
    return model,home,joints,spec

def test_unqualified_policy_requires_explicit_opt_in(tmp_path):
    m,h,j,s=bundle(tmp_path)
    with pytest.raises(ValueError,match='explicit opt-in'):validate_bundle(tmp_path,m,h,j)
    assert validate_bundle(tmp_path,m,h,j,allow_experimental=True)[0]==s

@pytest.mark.parametrize('fault',['body','home','policy','joint_order','frequency','escape','qualification'])
def test_incompatible_bundle_is_rejected(tmp_path,fault):
    m,h,j,s=bundle(tmp_path)
    if fault in ('body','home','policy'):
        {'body':m,'home':h,'policy':tmp_path/'policy.ts'}[fault].write_text('changed')
    elif fault=='joint_order':j=j[::-1]
    elif fault=='frequency':s['control_steps']=10
    elif fault=='escape':s['policy_file']='../outside.ts'
    else:s['qualification']='evaluated'
    (tmp_path/'policy.json').write_text(json.dumps(s))
    with pytest.raises((ValueError,FileNotFoundError)):validate_bundle(tmp_path,m,h,j,allow_experimental=True)

def test_invalid_action_does_not_become_zero_or_fallback():
    for action in [np.zeros(22),np.full(23,np.nan),np.full(23,np.inf)]:
        with pytest.raises(ValueError):validate_action(action)
    np.testing.assert_array_equal(validate_action(np.full(23,2)),np.ones(23))
