"""Audit a completed eight-hand run, then export combined CSVs and a success-rate plot."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from check_results import audit


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    expected=['allegro','shadow','leap','inspire','ability','schunk','brainco','barrett']
    rates=[];trials=[];per_object=[];reports=[];protocol=None;implementation=None
    for hand in expected:
        directory=args.directory/hand
        manifest,summary=audit(directory)
        assert manifest['hand']==hand
        assert len(manifest['objects'])==78 and manifest['updates']==25 and manifest['steps']==200
        current_protocol={k:manifest[k] for k in ['objects','updates','steps','seed']}
        if protocol is None: protocol=current_protocol
        assert protocol==current_protocol
        common={k:manifest[k] for k in ['entrypoint_sha256','implementation_sha256','checkpoint_sha256',
                'config_sha256','reference_poses_sha256','reference_metadata_sha256','label_sha256',
                'object_urdf_sha256','surface_points_sha256','physics_dt','control_dt','lift_step',
                'root_guided','batch_size','versions']}
        if implementation is None: implementation=common
        assert implementation==common,'Mixed evaluator code, data, versions or simulation parameters'
        reports.append({'hand':hand,'status':'complete',**summary})
        rates.append({'hand':hand,'trials':summary['completed_trials'],'successes':summary['successes'],
                      'success_rate':summary['success_rate'],'invalid_trials':summary['invalid_trials']})
        for line in (directory/'trials.jsonl').read_text().splitlines():
            trials.append({'hand':hand,**json.loads(line)})
        for name,r in summary['per_object'].items():per_object.append({'hand':hand,'object':name,**r})
    assert len(trials)==15600
    # A hand can be resumed separately after a launcher reports its first failure.
    # Rebuild the aggregate only after every saved trial has passed the audit.
    aggregate=args.directory/'all_hands_summary.json'
    report={'complete':True,'hands':reports,'protocol':protocol,'verification':'all saved trials audited'}
    if aggregate.exists():
        previous=json.loads(aggregate.read_text())
        if not previous.get('complete'):
            backup=args.directory/'runner_summary_before_audit.json'
            if not backup.exists(): backup.write_text(json.dumps(previous,indent=2))
    temporary=aggregate.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2));temporary.replace(aggregate)
    write_csv(args.directory/'success_rates.csv',rates)
    write_csv(args.directory/'all_trials.csv',trials)
    write_csv(args.directory/'per_object.csv',per_object)
    failures=[r for r in trials if not r['valid']]
    if failures: write_csv(args.directory/'numerical_failures.csv',failures)
    failure_counts={h:dict(Counter(r.get('failure_reason','unspecified') for r in failures if r['hand']==h)) for h in expected}
    (args.directory/'numerical_failure_counts.json').write_text(json.dumps(failure_counts,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    bars=ax.bar([r['hand'].capitalize() for r in rates],[r['success_rate']*100 for r in rates],color='#34699a')
    ax.bar_label(bars,labels=[f"{r['success_rate']:.1%}\n({r['successes']}/1950)" for r in rates],padding=4,fontsize=9)
    ax.set_ylim(0,105);ax.set_ylabel('Grasp success rate (%)')
    ax.set_title('DexGrasp-Zero checkpoint transfer to Isaac Sim / PhysX')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'78 YCB objects × 25 source poses per hand • 200 control steps • numerical failures included',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(args.directory/'success_rates.png',dpi=180);plt.close(fig)
    print('EXPORTED 15,600 trials, 624 object/hand rows and success_rates.png')


if __name__=='__main__':main()
