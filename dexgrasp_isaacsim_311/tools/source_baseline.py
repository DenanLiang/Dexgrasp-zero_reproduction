"""Small RaiSim reference rollout using the same saved labels and sampled geometry."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
import json
from scipy.spatial.transform import Rotation
from dexgrasp_isaac.task import DEX,hand_spec,ObjectData,load_policy,mv
from dexgrasp_zero.env.bin import cross_hand
from dexgrasp_zero.env.test_scripts.eval_utils import EvalSpec,compute_lifted_mask,compute_eval_metrics
name=sys.argv[1] if len(sys.argv)>1 else 'leap'
spec=hand_spec(name);objects=[ObjectData(n) for n in ['003_cracker_box','025_mug']]
n=len(objects);a=spec['act_dim'];k=len(spec['frames'])
cfg=yaml.safe_load((DEX/'env/envs/cross_hand/cfgs/cross_hand_base_cfg_reg_p2.yaml').read_text())['environment']
cfg.update(num_envs=n,num_threads=1,visualize=False,hand_model_r=spec['path'].name,load_set='ycb_crossDex')
env=getattr(cross_hand,name.capitalize()+'_Test')(str(DEX/'rsc'),yaml.safe_dump(cfg))
env.load_multi_articulated([f'{o.name}/{o.name}.urdf' for o in objects])
policy=load_policy('cuda:0',spec)
points=torch.tensor(np.stack([o.points for o in objects]),dtype=torch.float32,device='cuda:0')
ob=np.zeros((n,env.getRightObDim()),np.float32);left=np.zeros((n,1),np.float32);gs=np.zeros((n,env.getGSDim()),np.float32)
def observe():
 env.observe(ob,left);env.get_global_state(gs)
 joints=torch.tensor(gs[:,3+3*k:3+6*k].reshape(n,k,3),device='cuda:0')
 rot=Rotation.from_euler('XYZ',gs[:,:3]).as_matrix()
 vecs=[]
 for part in range(2):
  idx=torch.cdist(joints,points[:,part]).argmin(-1)
  near=torch.gather(points[:,part],1,idx[:,:,None].expand(-1,-1,3))
  vec=np.einsum('nij,nkj->nki',rot,(near-joints).cpu().numpy())
  if part==1: vec[[not o.has_nonaff for o in objects]]=0
  vecs.append(vec.reshape(n,-1))
 return np.concatenate([ob,*vecs],axis=1).astype(np.float32)
rows=[]
trace_obs=[];trace_actions=[]
for update in range(2):
 init=[o.initial(update,name) for o in objects]
 state=np.zeros((n,a),np.float32)
 if name in ['leap','allegro','shadow']:state[:,-4]=1.7
 elif name in ['schunk','inspire','ability','brainco']:state[:,6]={'schunk':1.,'inspire':1.3,'ability':-1.9,'brainco':.9}[name]
 state[:,:3]=np.stack([x[0] for x in init]);state[:,3:6]=Rotation.from_matrix(np.stack([x[1] for x in init])).as_euler('XYZ')
 op=np.array([[*x[2],1,0,0,0,0] for x in init],np.float32);zeros=np.zeros_like(state)
 env.reset_state(state,zeros,zeros,zeros,op)
 env.set_goals(np.stack([x[3] for x in init]).astype(np.float32),np.stack([o.aff.centroid for o in objects]).astype(np.float32),*[np.zeros((n,1),np.float32) for _ in range(8)])
 obs=observe()
 with torch.inference_mode():
  for step in range(200):
   if step==130:env.switch_root_guidance(True)
   actions=policy(torch.tensor(obs,device='cuda:0'),spec['hand_type']).cpu().numpy()
   trace_obs.append(obs.copy());trace_actions.append(actions.copy())
   env.step(actions,zeros,np.zeros(n,np.float32),np.zeros(n,np.float32),np.zeros(n,bool))
   obs=observe()
 es=EvalSpec.from_runtime(act_dim=a,num_bodyparts=k,obs_dim=obs.shape[1])
 success=compute_lifted_mask(es,obs,gs,op,.1,.1)
 for i,o in enumerate(objects):
  rows.append({'object':o.name,'update':update,'success':bool(success[i]),'lift_height_m':float(gs[i,es.obj_pos_z_idx]-op[i,2]),'center_error_m':float(np.linalg.norm(obs[i,:3]))})
 print(rows[-n:],flush=True)
(Path(__file__).resolve().parents[1]/f'results/source_baseline_{name}.json').write_text(json.dumps(rows,indent=2))

np.savez_compressed(Path(__file__).resolve().parents[1]/f'results/source_trace_{name}.npz',obs=trace_obs,actions=trace_actions)
