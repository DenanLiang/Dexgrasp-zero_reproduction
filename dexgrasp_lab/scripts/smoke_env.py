import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/dexgrasp_lab'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=6);p.add_argument('--steps',type=int,default=140)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
launcher=AppLauncher(args);app=launcher.app
try:
 import torch
 from dexgrasp_lab.tasks.direct.dexgrasp_lab.dexgrasp_lab_env_cfg import DexgraspLabEnvCfg
 from dexgrasp_lab.tasks.direct.dexgrasp_lab.dexgrasp_lab_env import DexgraspLabEnv
 cfg=DexgraspLabEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device
 env=DexgraspLabEnv(cfg);obs,_=env.reset()
 report={'obs_shape':list(obs['policy'].shape),'action_dim':cfg.action_space,'hands':{}}
 for name,s in env.specs.items():
  report['hands'][name]={'env_ids':env.global_ids[name].tolist(),'joints':env.robots[name].joint_names,'policy_order':env.orders[name],'observation_dim':s['obs_dim']}
 for step in range(args.steps):
  action=torch.randn((env.num_envs,cfg.action_space),device=env.device)*.1
  obs,reward,term,trunc,info=env.step(action)
  assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
  if step%20==0:print('SMOKE',step,float(reward.mean()),env.numerical_failures,flush=True)
 report.update(steps=args.steps,numerical_failures=env.numerical_failures,resets=env.completed_episodes,passed=True)
 (ROOT/'artifacts/environment_smoke.json').write_text(json.dumps(report,indent=2))
 print('ENVIRONMENT_SMOKE_PASS',report,flush=True)
 env.close()
except BaseException:
 import os,traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
app.close()
