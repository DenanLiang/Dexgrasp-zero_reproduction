"""Compare migrated FK/observation geometry to the original RaiSim C++ environment.
Run in dexgrasp310, with its LD_LIBRARY_PATH, before trusting a cross-simulator run.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from dexgrasp_isaac.task import DEX,HANDS,hand_spec,Kinematics,ObjectData,quat,mv,transpose
from dexgrasp_zero.env.bin import cross_hand

results={}
for name in HANDS:
    spec=hand_spec(name); kin=Kinematics(spec)
    cfg=yaml.safe_load((DEX/'env/envs/cross_hand/cfgs/cross_hand_base_cfg_reg_p2.yaml').read_text())['environment']
    cfg.update(num_envs=1,num_threads=1,visualize=False,hand_model_r=spec['path'].name,load_set='ycb_crossDex')
    ctor=getattr(cross_hand,name.capitalize()+'_Test')
    env=ctor(str(DEX/'rsc'),yaml.safe_dump(cfg))
    env.load_multi_articulated(['003_cracker_box/003_cracker_box.urdf'])
    data=ObjectData('003_cracker_box')
    base,rotation,obj,bias=data.initial(0,name)
    a=spec['act_dim']; q=np.zeros((1,a),np.float32)
    if name in ['leap','allegro','shadow']: q[:,-4]=1.7
    elif name in ['schunk','inspire','ability','brainco']: q[:,6]={'schunk':1.,'inspire':1.3,'ability':-1.9,'brainco':.9}[name]
    state=q.copy();state[:,:3]=base;state[:,3:6]=Rotation.from_matrix(rotation).as_euler('XYZ')
    zeros=np.zeros_like(state); op=np.array([[*obj,1,0,0,0,0]],np.float32)
    env.reset_state(state,zeros,zeros,zeros,op)
    env.set_goals(bias.reshape(1,3).astype(np.float32),data.aff.centroid.reshape(1,3).astype(np.float32),*[np.zeros((1,1),np.float32) for _ in range(8)])
    # Refresh C++ cached observation after set_goals without integrating physics.
    env.reset_state(state,zeros,zeros,zeros,op)
    ob=np.zeros((1,env.getRightObDim()),np.float32); left=np.zeros((1,1),np.float32)
    gs=np.zeros((1,env.getGSDim()),np.float32)
    env.observe(ob,left);env.get_global_state(gs)
    positions,rots=kin.forward(q,base[None],rotation[None])
    k=len(spec['frames'])
    source_positions=gs[:,3+3*k:3+6*k].reshape(1,k,3)+obj
    fk_error=float(np.max(np.abs(positions-source_positions)))
    fingers=positions[:,spec['finger_center_ids']].mean(1); thumb=positions[:,spec['thumb_id']]
    if name=='inspire': fingers=(fingers+thumb)*.5
    center=(fingers+thumb)*.5
    hand_center=mv(transpose(rots[:,0]),center-positions[:,0])
    delta=mv(transpose(rots[:,0]),bias+obj-positions[:,0])-hand_center
    center_error=float(np.max(np.abs(delta-ob[:,:3])))
    joints_error=float(np.max(np.abs(q[:,6:]-ob[:,6:a])))
    results[name]={'fk_max_abs_error_m':fk_error,'center_max_abs_error_m':center_error,
                   'joint_order_max_abs_error_rad':joints_error,
                   'raw_observation_dim':int(ob.shape[1]),'policy_observation_dim':spec['obs_dim']}
    print(name,results[name],flush=True)
    assert max(fk_error,center_error,joints_error)<1e-5,results[name]
    del env
import json
output=Path(__file__).resolve().parents[1]/'results/contract_validation.json'
output.write_text(json.dumps(results,indent=2))
print('ALL_CONTRACTS_PASS',flush=True)
