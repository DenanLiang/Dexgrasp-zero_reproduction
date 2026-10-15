#!/usr/bin/env python3
"""Launch isolated Isaac Sim processes and aggregate all eight hands."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
HANDS=['allegro','shadow','leap','inspire','ability','schunk','brainco','barrett']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hands',nargs='+',choices=HANDS,default=HANDS)
    p.add_argument('--gpus',nargs='+',type=int,default=[0])
    p.add_argument('--updates',type=int,default=25)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=78)
    p.add_argument('--objects',nargs='+')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output',type=Path,default=ROOT/'results/all_hands_ycb')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if len(set(args.hands)) != len(args.hands) or len(set(args.gpus)) != len(args.gpus): p.error('Duplicate hands or GPUs')
    def worker(gpu,hands):
        results=[]
        for hand in hands:
            dest=args.output/hand
            cmd=[sys.executable,'-u',str(ROOT/'evaluate.py'),'--hand',hand,'--gpu',str(gpu),
                 '--updates',str(args.updates),'--steps',str(args.steps),'--batch-size',str(args.batch_size),
                 '--seed',str(args.seed),'--output',str(dest)]
            if args.objects: cmd+=['--objects',*args.objects]
            if args.resume: cmd+=['--resume']
            log=args.output/f'{hand}.log'
            print(f'START hand={hand} gpu={gpu} log={log}',flush=True)
            with log.open('a' if args.resume else 'w') as f:
                process=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            summary=dest/'summary.json'
            if process.returncode or not summary.exists() or not json.loads(summary.read_text())['complete']:
                results.append({'hand':hand,'status':'failed','returncode':process.returncode,'log':str(log)})
                print(f'FAILED {hand}: {log}',flush=True)
            else:
                report=json.loads(summary.read_text())
                results.append({'hand':hand,'status':'complete',**report})
                print(f"DONE {hand} {report['successes']}/{report['completed_trials']} rate={report['success_rate']:.4f}",flush=True)
        return results
    reports=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,g,args.hands[i::len(args.gpus)]) for i,g in enumerate(args.gpus)]
        for future in concurrent.futures.as_completed(futures):
            reports.extend(future.result())
    reports.sort(key=lambda r:args.hands.index(r['hand']))
    complete=all(r['status']=='complete' for r in reports)
    report={'complete':complete,'hands':reports,'protocol':{'objects':args.objects or 'all 78 YCB objects',
            'updates':args.updates,'steps':args.steps,'seed':args.seed}}
    (args.output/'all_hands_summary.json').write_text(json.dumps(report,indent=2))
    if not complete: sys.exit(1)


if __name__=='__main__': main()
