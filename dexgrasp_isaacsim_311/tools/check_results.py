#!/usr/bin/env python3
"""Audit saved evaluation coverage, unique trial IDs and recomputed success counts."""
import argparse
import json
from pathlib import Path
import math


def audit(directory, allow_partial=False):
    manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    rows=[json.loads(line) for line in (directory/'trials.jsonl').read_text().splitlines()]
    expected={(name,u) for name in manifest['objects'] for u in range(manifest['updates'])}
    keys=[(r['object'],r['update']) for r in rows]
    assert len(keys)==len(set(keys)),f'{directory}: duplicate trials'
    assert set(keys)<=expected,f'{directory}: unexpected trials'
    if not allow_partial: assert set(keys)==expected,f'{directory}: missing {len(expected-set(keys))} trials'
    for r in rows:
        if r['valid']:
            for k in ['lift_height_m','center_error_m','direction_error','angle_error_rad','contact_ratio']:
                assert math.isfinite(r[k]),(directory,r)
            success=r['lift_height_m']>.1 and r['center_error_m']<.1
        else: success=False
        assert r['success']==success,(directory,r)
    successes=sum(r['success'] for r in rows)
    assert summary['successes']==successes
    assert summary['completed_trials']==len(rows)
    assert summary['expected_trials']==len(expected)
    assert summary['complete']==(set(keys)==expected)
    assert math.isclose(summary['success_rate'],successes/len(rows))
    assert summary['invalid_trials']==sum(not r['valid'] for r in rows)
    by_object={name:[r for r in rows if r['object']==name] for name in {r['object'] for r in rows}}
    assert set(summary['per_object'])==set(by_object)
    for name,object_rows in by_object.items():
        saved=summary['per_object'][name]
        count=sum(r['success'] for r in object_rows)
        assert saved['trials']==len(object_rows)
        assert saved['successes']==count
        assert math.isclose(saved['success_rate'],count/len(object_rows))
    assert math.isclose(summary['macro_object_success_rate'],sum(v['success_rate'] for v in summary['per_object'].values())/len(by_object))
    print(f"PASS {manifest['hand']}: {len(rows)}/{len(expected)} trials, {successes} successes ({successes/len(rows):.4%})")
    return manifest,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    directories=[args.directory] if (args.directory/'manifest.json').exists() else sorted(p.parent for p in args.directory.glob('*/manifest.json'))
    if not directories: raise FileNotFoundError('No manifests found')
    results=[audit(d,args.allow_partial) for d in directories]
    aggregate=args.directory/'all_hands_summary.json'
    if aggregate.exists():
        report=json.loads(aggregate.read_text())
        assert report['complete']==all(s['complete'] for _,s in results)
        assert {r['hand'] for r in report['hands']}=={m['hand'] for m,_ in results}
        for row in report['hands']:
            _,summary=next((m,s) for m,s in results if m['hand']==row['hand'])
            assert row['successes']==summary['successes']
    print('RESULT_AUDIT_PASS')


if __name__=='__main__':main()
