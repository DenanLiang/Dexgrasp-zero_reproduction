"""Isaac Sim integration check: a failed trial must not contaminate another."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':1,'multi_gpu':False,'fast_shutdown':True})
try:
 import numpy as np,json
 from dexgrasp_isaac.task import ObjectData
 from dexgrasp_isaac.simulation import CrossHandYCB
 env=CrossHandYCB([ObjectData(n) for n in ['003_cracker_box','025_mug']],'cuda:1')
 env.reset(0)
 # Inject a zero-norm object quaternion at the tensor boundary once.
 original=env.top.get_world_poses
 def corrupt_pose():
  pos,rot=original();rot[0]=0.;return pos,rot
 env.top.get_world_poses=corrupt_pose
 obs=env.observe()
 env.top.get_world_poses=original
 assert env.failed.tolist()==[True,False]
 assert env.failure_reason[0]=='invalid_object_pose'
 for s in range(10): obs=env.step(np.zeros((2,env.a)),s)
 assert np.isfinite(obs).all() and not env.failed[1]
 assert not env.results(0,np.ones(2,bool))[0]['success']
 env.reset(1)
 assert not env.failed.any()
 actions=np.zeros((2,env.a));actions[0,0]=np.nan
 obs=env.step(actions,0)
 assert env.failed.tolist()==[True,False]
 assert env.failure_reason[0]=='nonfinite_policy_action'
 assert np.isfinite(obs).all()
 result={'zero_quaternion_isolated':True,'invalid_action_isolated':True,'other_trial_continues':True,'failure_counts_as_unsuccessful':True,'next_reset_clears_failure':True}
 (ROOT/'results/numerical_isolation_validation.json').write_text(json.dumps(result,indent=2))
 print('NUMERICAL_ISOLATION_PASS',flush=True)
 env.close()
except BaseException:
 import os,traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
app.close()
