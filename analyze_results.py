"""Summarize measured runs without generating model responses or opening hidden data."""
import argparse
from pathlib import Path
from run_trackc import HERE, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE/'runs'/'main')
    root = parser.parse_args().run_dir.resolve()
    names = ['R0','R1','R2','R3','R4','human_instruction_only','human_demos_only',
             'ablation_fixed3','ablation_fixed5']
    measured = {name:read(root/f'{name}.json') for name in names if (root/f'{name}.json').exists()}
    summary = {}
    for name, run in measured.items():
        records = run['records']
        failures = [r for r in records if not r['correct']]
        summary[name] = {**run['metrics'],
            'correct_count':sum(r['correct'] for r in records),
            'failure_count':len(failures),
            'failures_with_any_truncated_response':sum(any(r['truncated']) for r in failures),
            'unparseable_predictions':sum(r['prediction'] is None for r in records),
            'expanded_to_five':sum(len(r['responses'])==5 for r in records),
            'config':run['config']}
    transitions = {}
    pairs = [('R0','R1'),('R1','R2'),('R2','R3'),('R3','R4')]
    if 'R4' in measured and (root/'adaptive_plan.json').exists():
        base=read(root/'adaptive_plan.json').get('base_round','R3')
        if (base,'R4') not in pairs:
            pairs.append((base,'R4'))
    for before,after in pairs:
        if before not in measured or after not in measured:
            continue
        a={r['id']:r for r in measured[before]['records']}
        b={r['id']:r for r in measured[after]['records']}
        assert a.keys()==b.keys(), 'Comparisons must use identical dev IDs'
        transitions[before+' -> '+after] = {
            'recovered':[i for i in a if not a[i]['correct'] and b[i]['correct']],
            'regressed':[i for i in a if a[i]['correct'] and not b[i]['correct']],
            'still_wrong':[i for i in a if not a[i]['correct'] and not b[i]['correct']]}
    usage = {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,
             'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':0}
    proposals = []
    for path in sorted(root.glob('proposal_*.json')):
        proposal=read(path)
        audit=proposal.get('audit',{})
        for key in usage:
            usage[key]+=(audit.get('usage') or {}).get(key,0)
        proposals.append({'file':path.name,'failed':proposal.get('failed',False),
                          'config':proposal.get('config'),'response_model':audit.get('response_model')})
    save(root/'analysis_summary.json',{'metrics':summary,'transitions':transitions,
        'improver_usage':usage,'proposals':proposals,
        'note':'Automatic error categories describe observable output properties, not inferred causes.'})
    for name,item in summary.items():
        print(f"{name}: {item['correct_count']}/{item['n']}; "
              f"tokens={item['generated_tokens']}; generation cap rate={item['truncation_rate']:.1%}")
    for pair,change in transitions.items():
        print(pair, {key:len(value) for key,value in change.items()})


if __name__=='__main__':
    main()
