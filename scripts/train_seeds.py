"""
Orchestrates 20 training runs: 4 models × 5 seeds.

Each run is a subprocess with PYTHONHASHSEED set in the environment so that
hash randomness is isolated before Python starts. Results are written to
results/runs/{model}_seed{seed}.json. Completed runs are skipped on restart
(crash-resume safe).

Usage:
    python train_seeds.py               # run all 20
    python train_seeds.py --model vit   # run all 5 seeds for one model only
"""
import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ONE    = os.path.join(PROJECT_ROOT, 'scripts', 'train_one.py')

SEEDS  = [42, 123, 256, 789, 1024]
MODELS = ['vgg16', 'vit', 'hybrid_gated', 'a2wnet']


def result_exists(model, seed):
    path = os.path.join(PROJECT_ROOT, 'results', 'runs',
                        f'{model}_seed{seed}.json')
    return os.path.exists(path)


def run_all(models):
    total  = len(models) * len(SEEDS)
    done   = 0
    failed = []

    for model in models:
        for seed in SEEDS:
            done += 1
            tag = f"{model} seed={seed}"

            if result_exists(model, seed):
                print(f"[{done}/{total}] SKIP  {tag}  (already done)")
                continue

            print(f"\n[{done}/{total}] START {tag}")
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = str(seed)

            proc = subprocess.run(
                [sys.executable, TRAIN_ONE, '--model', model, '--seed', str(seed)],
                env=env,
                cwd=PROJECT_ROOT
            )

            if proc.returncode != 0:
                print(f"  FAILED: {tag}  (exit code {proc.returncode})")
                failed.append((model, seed))
            else:
                print(f"  DONE:  {tag}")

    print(f"\n{'='*60}")
    print(f"Finished. {total - len(failed)} / {total} runs succeeded.")
    if failed:
        print("Failed runs:", failed)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, default=None,
                        help='Run only this model (all 5 seeds)')
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS
    run_all(models_to_run)
