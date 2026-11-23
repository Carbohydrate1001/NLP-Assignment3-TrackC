"""A dev-only 2x2 ablation isolating R1's instruction and demonstrations."""
import argparse
import csv
from pathlib import Path
from run_trackc import HERE, Evaluator, cache_eval, check_open, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE/'runs'/'main')
    root = parser.parse_args().run_dir.resolve()
    check_open(root)
    manifest, visible = read(root/'manifest.json'), read(root/'visible.json')
    r0, r1 = read(root/'R0.json'), read(root/'R1.json')
    evaluator = Evaluator(manifest, visible['demos'], root)
    instruction_only = {**r1['config'], 'demo_ids': []}
    demos_only = {**r0['config'], 'demo_ids': r1['config']['demo_ids']}
    measured = {
        'baseline': r0,
        'instruction_only': cache_eval(root, 'human_instruction_only', evaluator,
                                      visible['dev'], instruction_only),
        'demos_only': cache_eval(root, 'human_demos_only', evaluator, visible['dev'], demos_only),
        'instruction_and_demos': r1,
    }
    rows = [{'variant':name, **result['metrics']} for name,result in measured.items()]
    with (root/'human_ablation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['variant']}: accuracy={row['accuracy']:.1%}, "
              f"tokens={row['generated_tokens']}, truncated={row['truncation_rate']:.1%}")


if __name__ == '__main__':
    main()
