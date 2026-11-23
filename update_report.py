"""Generate report tables from completed, sealed measurements; never run inference."""
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs' / 'main'
REPORT = ROOT / 'report'


def read(name):
    return json.loads((RUN / (name + '.json')).read_text(encoding='utf-8'))


def table(header, rows, caption, alignment):
    return '\n'.join([r'\begin{table}[H]\centering\small',
        r'\begin{tabular}{' + alignment + '}', r'\toprule ' + header + r' \\',
        r'\midrule', *[row + r' \\' for row in rows], r'\bottomrule\end{tabular}',
        r'\caption{' + caption + '}', r'\end{table}'])


def main():
    names = ['R0','R1','R2','R3','R4','human_instruction_only','human_demos_only',
             'ablation_fixed3','ablation_fixed5']
    runs = {name:read(name) for name in names}
    final, frozen = read('final_test'), read('frozen')
    if final['selected_round'] != frozen['selected_round']:
        raise ValueError('Selected artifact mismatch')
    labels = {'ablation_fixed3':'fixed-three', 'ablation_fixed5':'fixed-five',
              'human_instruction_only':'instruction-only', 'human_demos_only':'demo-only'}
    selected = labels.get(final['selected_round'], final['selected_round'])
    evaluations = {}
    for path in RUN.glob('*.json'):
        obj = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj,dict) and 'metrics' in obj and 'records' in obj:
            evaluations[path.stem] = obj['metrics']
    total_seconds = sum(m['elapsed_seconds'] for m in evaluations.values())
    macros = {'SelectedName':selected, 'FinalDev':f"{100*final['dev_accuracy']:.0f}\\%",
              'FinalTest':f"{100*final['metrics']['accuracy']:.0f}\\%",
              'FinalGap':f"{final['gap_percentage_points']:.1f}",
              'LocalMinutes':f'{total_seconds/60:.1f}'}
    (REPORT/'metrics.tex').write_text('\n'.join(
        '\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items()),encoding='utf-8')
    rows=[]
    for name in names[:5]:
        m=runs[name]['metrics']
        rows.append(f"{name} & {100*m['accuracy']:.0f}\\% & {m['generations']} & "
                    f"{m['generated_tokens']:,} & {m['elapsed_seconds']:.1f} & "
                    f"{100*m['truncation_rate']:.1f}\\%")
    result_text=table('Round & Dev & Generations & Output tokens & Seconds & Cap rate',rows,
        'Results on 100 dev questions with a 256-token limit per response. '
        'Cap rate is the fraction of responses reaching the limit; time covers inference.', 'lrrrrr')
    fm=final['metrics']
    result_text += (f"\nThe selected \\textbf{{{selected}}} configuration scored "
                   f"{100*final['dev_accuracy']:.0f}\\% dev and "
                   f"\\textbf{{{100*fm['accuracy']:.0f}\\% test}} "
                   f"({round(fm['n']*fm['accuracy'])}/{fm['n']}), with a dev-minus-test gap of "
                   f"{final['gap_percentage_points']:.1f} percentage points. "
                   f"Test evaluation used {fm['generations']} generations, "
                   f"{fm['generated_tokens']:,} output tokens, and {fm['elapsed_seconds']:.1f} seconds.\n")
    shutil.copy2(RUN/'rsi_curve.pdf', REPORT/'rsi_curve.pdf')
    result_text += '\n'.join([r'\begin{figure}[H]\centering',
        r'\includegraphics[width=.72\linewidth]{rsi_curve.pdf}',
        r'\caption{Dev accuracy across rounds. The dashed line marks the test score '
        r'of the final five-response configuration. '
        r'R4 uses the R2 prompt.}',r'\end{figure}'])
    (REPORT/'results.tex').write_text(result_text,encoding='utf-8')
    human=[]
    for label,name in [('Neither (R0)','R0'),('Instruction only','human_instruction_only'),
                       ('Demonstrations only','human_demos_only'),('Both (R1)','R1')]:
        m=runs[name]['metrics']
        human.append(f"{label} & {100*m['accuracy']:.0f}\\% & "
                     f"{100*m['truncation_rate']:.0f}\\% & {m['generated_tokens']:,}")
    text=table('R1 component & Dev & Cap rate & Output tokens',human,
               'Instruction and demonstration ablation; all other settings are fixed.', 'lrrr')
    sampling=[]
    for label,name in [('Greedy (R2)','R2'),('Fixed three','ablation_fixed3'),
                       ('Adaptive (R4)','R4'),('Fixed five','ablation_fixed5')]:
        m=runs[name]['metrics']
        sampling.append(f"{label} & {100*m['accuracy']:.0f}\\% & {m['generations']} & "
                        f"{m['generated_tokens']:,} & {m['elapsed_seconds']:.1f}")
    text+='\n'+table('Decoding & Dev & Generations & Output tokens & Seconds',sampling,
        'Sampling comparison with the same R2 instruction and static demonstrations. '
        'All sampled variants include one greedy answer.', 'lrrrr')
    (REPORT/'ablations.tex').write_text(text,encoding='utf-8')
    a,b,c=(runs[x]['metrics'] for x in ['R4','ablation_fixed3','ablation_fixed5'])
    paragraph=(f"Three responses per question yielded {b['accuracy']:.0%} dev accuracy, "
               f"adaptive sampling {a['accuracy']:.0%}, and five responses {c['accuracy']:.0%}. "
               f"Compared with five responses, adaptive sampling used "
               f"{100*(1-a['generations']/c['generations']):.1f}% fewer generations and "
               f"{100*(1-a['generated_tokens']/c['generated_tokens']):.1f}% fewer output tokens, "
               f"at {100*(c['accuracy']-a['accuracy']):.0f} percentage points lower accuracy. "
               f"The five-response configuration was selected for its higher dev accuracy. ")
    (REPORT/'sampling_analysis.tex').write_text(paragraph.replace('%',r'\%'),encoding='utf-8')
    with (RUN/'complete_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['artifact','split',*runs['R0']['metrics'].keys()])
        writer.writeheader()
        for name,obj in runs.items():
            writer.writerow({'artifact':name,'split':'dev',**obj['metrics']})
        writer.writerow({'artifact':final['selected_round'],'split':'test',**fm})
    (RUN/'resource_totals.json').write_text(json.dumps({
        'measured_target_evaluations':len(evaluations),
        'target_inference_seconds':total_seconds,
        'target_generated_tokens':sum(m['generated_tokens'] for m in evaluations.values()),
        'target_generations':sum(m['generations'] for m in evaluations.values()),
        'evaluation_names':list(evaluations),
        'excludes':'downloads, model loading, API waiting, reporting'},indent=2),encoding='utf-8')
    print(f'Generated final measured tables: {selected}, test={fm["accuracy"]:.0%}.')


if __name__=='__main__':
    main()
