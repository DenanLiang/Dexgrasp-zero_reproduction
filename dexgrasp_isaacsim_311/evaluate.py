#!/usr/bin/env python3
"""Reproducible YCB evaluation. Run with the dexgrasp_isaacsim Python interpreter."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hand', choices=['allegro','shadow','leap','inspire','ability','schunk','brainco','barrett'], default='leap')
    p.add_argument('--updates',type=int,default=25)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--objects',nargs='+',help='Exact YCB folder names; default is every object')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--visualize',action='store_true')
    p.add_argument('--no-root-guide',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Validate inputs and checkpoint without starting Isaac Sim')
    p.add_argument('--trace',action='store_true',help='Save per-step observations, actions and joint states')
    p.add_argument('--dump-reset',action='store_true',help='Save initial observation and FK for validation')
    args = p.parse_args()
    if args.output is None:
        args.output = Path(__file__).resolve().parent/'results'/f'{args.hand}_ycb'
    if args.updates<1 or args.updates>25 or args.steps<1 or args.batch_size<1:
        p.error('Require 1 <= updates <= 25, steps > 0 and batch-size > 0')
    # SimulationApp must precede torch imports (Isaac Sim plugin ABI requirement).
    app=None
    if not args.preflight:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':not args.visualize,'active_gpu':args.gpu,'physics_gpu':args.gpu,
                           'multi_gpu':False,'sync_loads':False,'fast_shutdown':True})
    try:
        import numpy as np
        import torch
        from dexgrasp_isaac.task import (YCB,HAND,WEIGHT,CONFIG,DEX,ObjectData,load_policy,digest,summarize,hand_spec)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        names=sorted(d.name for d in YCB.iterdir() if d.is_dir())
        if args.objects:
            unknown=set(args.objects)-set(names)
            if unknown: p.error(f'Unknown YCB objects: {unknown}')
            names=sorted(set(args.objects))
        if not names: raise RuntimeError('No YCB objects found')
        device=f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
        if not args.preflight and device=='cpu': raise RuntimeError('CUDA unavailable; cannot run Isaac Sim evaluation')
        objects=[ObjectData(n,args.seed) for n in names]
        for obj in objects:
            for u in range(args.updates):
                if f'{obj.name}_{u}' not in obj.labels: raise KeyError(f'Missing label {obj.name}_{u}')
        spec=hand_spec(args.hand)
        policy=load_policy(device,spec)
        with torch.inference_mode():
            probe=policy(torch.zeros((2,spec['obs_dim']),device=device),spec['hand_type'])
        if probe.shape!=(2,spec['act_dim']) or not torch.isfinite(probe).all(): raise RuntimeError('Invalid checkpoint output')
        print(f'PREFLIGHT_OK objects={len(names)} trials={len(names)*args.updates} policy_output={tuple(probe.shape)} device={device}',flush=True)
        if args.preflight: return
        from dexgrasp_isaac.simulation import CrossHandYCB
        manifest={'backend':'Isaac Sim 5.1 / PhysX CPU TGS', 'state_transport':'Fabric; direct physics tensors',
                  'versions':{n:importlib.metadata.version(n) for n in ['isaacsim','torch','numpy','trimesh','scipy','scikit-learn','torch-geometric']},
                  'entrypoint_sha256':digest(__file__),'hand':args.hand,'seed':args.seed,
                  'objects':names,'updates':args.updates,'steps':args.steps,'batch_size':args.batch_size,
                  'root_guided':not args.no_root_guide,'lift_step':130,'physics_dt':.0025,'control_dt':.01,
                  'success':'height_delta > 0.1 m AND center_error < 0.1 m',
                  'checkpoint':str(WEIGHT),'checkpoint_sha256':digest(WEIGHT),'config_sha256':digest(CONFIG),
                  'hand_urdf_sha256':digest(spec['path']),'label_sha256':{n:digest(YCB/f'{n}_label.pkl') for n in names},
                  'implementation_sha256':{x.name:digest(x) for x in sorted((Path(__file__).parent/'dexgrasp_isaac').glob('*')) if x.suffix in ['.py','.json']},
                  'point_sampling':'200 points per part, deterministic per object and seed',
                  'surface_points_sha256':hashlib.sha256(np.stack([o.points for o in objects]).tobytes()).hexdigest(),
                  'object_urdf_sha256':{o.name:digest(o.path/f'{o.name}.urdf') for o in objects},
                  'table':'static box, dimensions 2 x 1 x 0.5 m, top z=0.5 m',
                  'drive':'force PD with original gains + URDF passive damping; URDF effort limits',
                  'object_drive':'zero stiffness; URDF passive damping; damping force ceiling 1e6',
                  'numerical_failures':'terminate and park affected trial; count as failure without retry; preserve failure reason and step',
                  'initial_observation':'fresh goals and zero previous PD target buffer on each reset; source caches stale goals/previous targets',
                  'reference_poses_sha256':digest(Path(__file__).parent/'reference/initial_poses.npz'),
                  'reference_metadata_sha256':digest(Path(__file__).parent/'reference/initial_poses.json'),
                  'barrett_initialization':'source fixed wrist override (z+0.3, intrinsic XYZ [3.14,0,0]); label target center retained',
                  'contact_impulse':'PhysX net normal impulse per link/part (differs from RaiSim last contact impulse)',
                  'dummy_link_mass_kg':1e-5,'dummy_link_inertia_kg_m2':1e-8}
        out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
        manifest_path=out/'manifest.json'; rows_path=out/'trials.jsonl'
        rows=[]
        if manifest_path.exists():
            if not args.resume: raise FileExistsError(f'{out} exists; choose another output or pass --resume')
            if json.loads(manifest_path.read_text()) != manifest: raise ValueError('Resume manifest differs; use a new output directory')
            if rows_path.exists():
                rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        else:
            if rows_path.exists(): raise RuntimeError('Trials exist without manifest')
            atomic_json(manifest_path,manifest)
            np.savez_compressed(out/'surface_points.npz',names=names,points=np.stack([o.points for o in objects]))
        done={(r['object'],r['update']) for r in rows}
        if len(done)!=len(rows): raise RuntimeError('Duplicate trials in saved results')
        expected=len(names)*args.updates
        start=time.monotonic()
        for offset in range(0,len(objects),args.batch_size):
            batch=objects[offset:offset+args.batch_size]
            pending=[u for u in range(args.updates) if any((o.name,u) not in done for o in batch)]
            if not pending: continue
            env=CrossHandYCB(batch,device,args.visualize,not args.no_root_guide,args.hand)
            try:
                for update in pending:
                    obs=env.reset(update)
                    if args.dump_reset:
                        np.savez(out/f'reset_{offset}_{update}.npz',obs=obs,q=env.last_q,base_pos=env.base_pos,
                                 base_rot=env.base_rot,hand_center=env.hand_center,aff_center=env.aff_center,
                                 points=env.points.cpu().numpy())
                    valid=np.isfinite(obs).all(axis=1)
                    trace_obs=[]; trace_actions=[]; trace_q=[]
                    with torch.inference_mode():
                        for step in range(args.steps):
                            actions=policy(torch.from_numpy(np.nan_to_num(obs)).to(device),spec['hand_type']).cpu().numpy()
                            if args.trace:
                                trace_obs.append(obs.copy()); trace_actions.append(actions.copy()); trace_q.append(env.last_q.copy())
                            valid &= np.isfinite(actions).all(axis=1)
                            obs=env.step(actions,step)
                            valid &= np.isfinite(obs).all(axis=1) & ~env.failed
                    if args.trace:
                        np.savez_compressed(out/f'trace_{offset}_{update}.npz',obs=trace_obs,actions=trace_actions,q=trace_q)
                    current=[r for r in env.results(update,valid) if (r['object'],r['update']) not in done]
                    with rows_path.open('a') as f:
                        for row in current: f.write(json.dumps(row,allow_nan=False)+'\n')
                        f.flush(); os.fsync(f.fileno())
                    rows.extend(current); done.update((r['object'],r['update']) for r in current)
                    report=summarize(rows,expected)
                    report['elapsed_seconds_this_invocation']=time.monotonic()-start
                    atomic_json(out/'summary.json',report)
                    with (out/'trials.csv').open('w') as f:
                        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    print(f"PROGRESS {len(rows)}/{expected} success={report['successes']} rate={report['success_rate']:.4f} invalid={report['invalid_trials']} elapsed={time.monotonic()-start:.1f}s",flush=True)
            finally: env.close()
        print(f'FINISHED {out}/summary.json',flush=True)
    except BaseException as error:
        import sys
        import traceback
        traceback.print_exc()
        if app is not None:
            # Kit's fast shutdown exits with code 0 even after Python errors;
            # full plugin shutdown can hang in 5.1. This standalone process
            # must preserve failure status for callers. The OS releases its
            # GPU context and file handles on exit. Trial files are fsynced.
            sys.stdout.flush(); sys.stderr.flush()
            code = error.code if isinstance(error,SystemExit) and isinstance(error.code,int) else (130 if isinstance(error,KeyboardInterrupt) else 1)
            os._exit(code)
        raise
    finally:
        if app is not None: app.close()


if __name__=='__main__': main()
