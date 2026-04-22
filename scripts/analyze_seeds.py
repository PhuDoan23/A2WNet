"""
Statistical analysis of multi-seed results (pretrained + ablation).

Loads results/runs/ and results/ablation_runs/ for all runs, then:
  - Prints mean ± std test accuracy per model
  - Runs paired t-test between every model pair (paired by seed)
  - Flags pairs with p < 0.05 as significant

Usage:
    python analyze_seeds.py            # pretrained runs only
    python analyze_seeds.py --all      # pretrained + ablation combined
    python analyze_seeds.py --ablation # ablation only
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEEDS = [42, 123, 256, 789, 1024]

PRETRAINED_MODELS = ['vgg16', 'vit', 'hybrid_gated', 'a2wnet']
ABLATION_MODELS   = ['vgg16_scratch', 'a2wnet_scratch']

LABELS = {
    'vgg16':          'VGG16 (ImageNet)',
    'vit':            'ViT-B/16 (ImageNet-21k)',
    'hybrid_gated':   'HybridGated (ImageNet)',
    'a2wnet':         'A2WNet (ImageNet)',
    'vgg16_scratch':  'VGG16 (scratch)',
    'a2wnet_scratch': 'A2WNet (scratch)',
}


def load_results(models, subdir):
    runs_dir = os.path.join(PROJECT_ROOT, 'results', subdir)
    data, missing = {}, []
    for model in models:
        accs = []
        for seed in SEEDS:
            path = os.path.join(runs_dir, f'{model}_seed{seed}.json')
            if not os.path.exists(path):
                missing.append(f'{model}_seed{seed}')
                continue
            with open(path) as f:
                accs.append(json.load(f)['test_accuracy'])
        data[model] = accs
    return data, missing


def print_summary(data, title='Model Performance Summary (Test Accuracy)'):
    print('=' * 68)
    print(title)
    print(f"  {'Model':<30}  {'N':>2}  {'Mean':>9}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print('=' * 68)
    rows = []
    for model, accs in data.items():
        if not accs:
            print(f"  {LABELS.get(model, model):<30}  no data")
            continue
        n    = len(accs)
        mean = np.mean(accs) * 100
        std  = np.std(accs, ddof=1) * 100
        mn   = np.min(accs) * 100
        mx   = np.max(accs) * 100
        label = LABELS.get(model, model)
        print(f"  {label:<30}  {n:>2}  {mean:>8.2f}%  {std:>6.2f}%  {mn:>6.2f}%  {mx:>6.2f}%")
        rows.append({'Model': label, 'N': n,
                     'Mean (%)': round(mean, 4), 'Std (%)': round(std, 4),
                     'Min (%)': round(mn, 4),    'Max (%)': round(mx, 4)})
    print('=' * 68)
    return rows


def print_ttests(data, title='Paired t-test'):
    complete = [m for m in data if len(data[m]) == len(SEEDS)]
    if len(complete) < 2:
        print("Need ≥2 fully-complete models for t-tests.")
        return
    print()
    print('=' * 78)
    print(f'{title} (paired by seed, two-tailed, α=0.05, df={len(SEEDS)-1})')
    print('=' * 78)
    print(f"  {'Pair':<44}  {'t-stat':>8}  {'p-value':>10}  {'sig?':>6}")
    print('-' * 78)
    for m1, m2 in combinations(complete, 2):
        a1, a2   = np.array(data[m1]), np.array(data[m2])
        t, p     = stats.ttest_rel(a1, a2)
        sig      = 'YES *' if p < 0.05 else 'no'
        pair     = f"{LABELS.get(m1,m1)} vs {LABELS.get(m2,m2)}"
        print(f"  {pair:<44}  {t:>8.3f}  {p:>10.4f}  {sig:>6}")
    print('=' * 78)
    print()
    print("Note: n=5 seeds → low power (df=4). p<0.05 requires a large, consistent gap.")


def save_csv(rows, filename):
    if not rows:
        return
    out = os.path.join(PROJECT_ROOT, 'results', filename)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all',      action='store_true',
                       help='Show pretrained + ablation together')
    group.add_argument('--ablation', action='store_true',
                       help='Show ablation results only')
    args = parser.parse_args()

    if args.ablation:
        data, missing = load_results(ABLATION_MODELS, 'ablation_runs')
        if missing:
            print(f"WARNING: {len(missing)} missing: {missing}\n")
        rows = print_summary(data, 'Ablation Results (no ImageNet weights)')
        print_ttests(data, 'Paired t-test — Ablation models')
        save_csv(rows, 'ablation_summary.csv')

    elif args.all:
        pt_data, pt_miss = load_results(PRETRAINED_MODELS, 'runs')
        ab_data, ab_miss = load_results(ABLATION_MODELS,   'ablation_runs')
        missing = pt_miss + ab_miss
        if missing:
            print(f"WARNING: {len(missing)} missing: {missing}\n")
        combined = {**pt_data, **ab_data}
        rows = print_summary(combined, 'All Results — Pretrained vs Scratch')

        # Key ablation comparisons
        ablation_pairs = {
            'vgg16 vs vgg16_scratch':   ('vgg16',  'vgg16_scratch'),
            'a2wnet vs a2wnet_scratch': ('a2wnet', 'a2wnet_scratch'),
        }
        print()
        print('=' * 68)
        print('ImageNet Ablation — pretrained vs scratch (paired t-test)')
        print(f"  α=0.05, df={len(SEEDS)-1}")
        print('=' * 68)
        print(f"  {'Comparison':<40}  {'Δ mean':>8}  {'t-stat':>8}  {'p':>8}  {'sig?':>6}")
        print('-' * 68)
        for label, (m_pre, m_scratch) in ablation_pairs.items():
            a_pre     = np.array(combined.get(m_pre, []))
            a_scratch = np.array(combined.get(m_scratch, []))
            if len(a_pre) == len(SEEDS) and len(a_scratch) == len(SEEDS):
                delta = (np.mean(a_pre) - np.mean(a_scratch)) * 100
                t, p  = stats.ttest_rel(a_pre, a_scratch)
                sig   = 'YES *' if p < 0.05 else 'no'
                print(f"  {label:<40}  {delta:>+7.2f}%  {t:>8.3f}  {p:>8.4f}  {sig:>6}")
            else:
                print(f"  {label:<40}  incomplete data")
        print('=' * 68)
        save_csv(rows, 'all_summary.csv')

    else:
        data, missing = load_results(PRETRAINED_MODELS, 'runs')
        if missing:
            print(f"WARNING: {len(missing)} missing: {missing}\n")
        rows = print_summary(data, 'Pretrained Results (Test Accuracy)')
        print_ttests(data, 'Paired t-test — Pretrained models')
        save_csv(rows, 'seed_summary.csv')
