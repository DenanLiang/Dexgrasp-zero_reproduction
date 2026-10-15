"""Shared GNN/PPO training for Shadow, Allegro and Barrett on filtered-CMAP train objects."""
import sys,argparse,json,hashlib,time,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/dexgrasp_lab'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs',type=int,default=144)
parser.add_argument('--iterations',type=int,default=1000)
parser.add_argument('--rollout',type=int,default=130)
parser.add_argument('--epochs',type=int,default=4)
parser.add_argument('--minibatches',type=int,default=4)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--output',type=Path,default=ROOT/'logs'/time.strftime('%Y%m%d_%H%M%S'))
parser.add_argument('--resume',type=Path)
parser.add_argument('--stage',choices=['floating','fixed'],default='floating')
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
if min(args.iterations,args.rollout,args.epochs,args.minibatches,args.num_envs)<1:parser.error('positive counts required')
if args.output.exists() and any(args.output.iterdir()):parser.error('Output directory is not empty; use a new directory')
launcher=AppLauncher(args);app=launcher.app
try:
 import torch,numpy as np,gymnasium as gym,importlib.metadata
 import dexgrasp_lab.tasks
 from dexgrasp_lab.tasks.direct.dexgrasp_lab.dexgrasp_lab_env_cfg import DexgraspLabEnvCfg
 from dexgrasp_lab.policy import ActorCritic
 from dexgrasp_lab.ppo import PPO,PPOCfg
 from dexgrasp_lab.dataset import manifest,split_objects,object_mesh_path
 torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
 cfg=DexgraspLabEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed;cfg.stage=args.stage
 env=gym.make('DexGraspZero-CMAP-Direct-v0',cfg=cfg).unwrapped
 model=ActorCritic(cfg.hands).to(env.device);pcfg=PPOCfg(epochs=args.epochs,minibatches=args.minibatches)
 algorithm=PPO(model,pcfg)
 protocol={'dataset':manifest('train'),'hands':list(cfg.hands),'stage':cfg.stage,'obs_dim':model.obs_dim,
           'action_dim':model.action_dim,'physics_dt':cfg.sim.dt,'decimation':cfg.decimation,'seed':args.seed,
           'model_config':model.config,
           'hand_urdf_sha256':{h:hashlib.sha256(s['path'].read_bytes()).hexdigest() for h,s in env.specs.items()},
           'object_mesh_sha256':{n:hashlib.sha256(object_mesh_path(n).read_bytes()).hexdigest() for n in split_objects('train')},
           'source_code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'source').rglob('*.py'))}}
 start=0
 if args.resume:
  checkpoint=torch.load(args.resume,map_location=env.device,weights_only=False)
  if checkpoint['protocol']!=protocol:raise ValueError('Checkpoint protocol/code differs')
  model.load_state_dict(checkpoint['model'],strict=True);algorithm.optimizer.load_state_dict(checkpoint['optimizer'])
  start=checkpoint['iteration'];torch.set_rng_state(checkpoint['torch_rng'].cpu())
  if torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
 args.output.mkdir(parents=True,exist_ok=True)
 record={'protocol':protocol,'ppo':dataclasses.asdict(pcfg),'num_envs':env.num_envs,'rollout':args.rollout,
         'versions':{n:importlib.metadata.version(n) for n in ['isaaclab','isaacsim','torch','torch-geometric']},
         'assigned_objects':env.object_names,'model_config':model.config,'resume_from':str(args.resume) if args.resume else None,
         'initial_iteration':start,'requested_updates':args.iterations}
 (args.output/'manifest.json').write_text(json.dumps(record,indent=2))
 observation,_=env.reset();observation=observation['policy']
 logs=[];began=time.monotonic()
 for iteration in range(start,start+args.iterations):
  rollout,observation=algorithm.collect(env,observation,args.rollout)
  stats=algorithm.update(rollout)
  stats.update(iteration=iteration+1,transitions=int(rollout['reward'].numel()),numerical_failures=env.numerical_failures,
               workspace_failures=env.workspace_failures,
               reward_mean=float(rollout['reward'].mean()),elapsed_seconds=time.monotonic()-began)
  stats['per_hand_reward']={name:float(rollout['reward'][:,ids].mean()) for name,ids in env.global_ids.items()}
  stats['per_hand_contact_fraction']={name:float(rollout['obs'][:,ids,2*env.specs[name]['act_dim']+6:2*env.specs[name]['act_dim']+6+len(env.specs[name]['contact_ids'])].mean()) for name,ids in env.global_ids.items()}
  logs.append(stats)
  with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
  state={'model':model.state_dict(),'optimizer':algorithm.optimizer.state_dict(),'iteration':iteration+1,
         'protocol':protocol,'model_config':model.config,'torch_rng':torch.get_rng_state(),
         'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'ppo':dataclasses.asdict(pcfg)}
  tmp=args.output/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(args.output/'checkpoint.pt')
  print('PPO_UPDATE',json.dumps(stats),flush=True)
 summary={'complete':True,'updates':len(logs),'first_iteration':start+1,'last_iteration':start+len(logs),
          'total_transitions':sum(r['transitions'] for r in logs),'all_losses_finite':True,
          'all_actor_updates_nonzero':all(r['actor_parameter_delta']>0 for r in logs),'numerical_failures':env.numerical_failures,
          'workspace_failures':env.workspace_failures,
          'hands':list(cfg.hands),'unique_training_objects':len(set(env.object_names)),'elapsed_seconds':time.monotonic()-began}
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
 print('TRAINING_COMPLETE',json.dumps(summary),flush=True)
 env.close()
except BaseException:
 import os,traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
app.close()
