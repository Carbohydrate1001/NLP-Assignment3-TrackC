"""Replay saved dev configurations without an improver API or hidden-test access."""
import argparse
from datetime import datetime
from pathlib import Path
from run_trackc import HERE, Evaluator, cache_eval, read, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'runs'/'main')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rounds',nargs='+',default=['R0','R1','R2','R3','R4'])
    args=parser.parse_args()
    source=args.source.resolve()
    output=(args.output or HERE/'runs'/('reproduction_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if output.exists():
        raise RuntimeError('Choose a new output directory; original results must stay unchanged.')
    allowed={'R0','R1','R2','R3','R4','human_instruction_only','human_demos_only','ablation_fixed3','ablation_fixed5'}
    if not set(args.rounds)<=allowed:
        raise ValueError('Only declared dev artifacts may be replayed.')
    manifest,visible=read(source/'manifest.json'),read(source/'visible.json')
    original={label:read(source/f'{label}.json') for label in args.rounds}
    output.mkdir(parents=True)
    save(output/'manifest.json',manifest)
    save(output/'visible.json',visible)
    save(output/'purpose.json',{'purpose':'Replay saved configurations; no test data or model selection',
                               'source':str(source),'rounds':args.rounds})
    evaluator=Evaluator(manifest,visible['demos'],output)
    for label,result in original.items():
        replay=cache_eval(output,label,evaluator,visible['dev'],result['config'],
                          seed=result['metrics']['seed'],max_tokens=result['metrics']['max_new_tokens'])
        print(f"{label}: saved={result['metrics']['accuracy']:.1%}, replay={replay['metrics']['accuracy']:.1%}")


if __name__=='__main__':
    main()
