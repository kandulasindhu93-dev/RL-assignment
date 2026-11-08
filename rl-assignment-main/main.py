"""Entry point: run both methods on all seeds and save raw results.

Usage:  python main.py [--seeds 0 1 2] [--steps 50000] [--out results]
Each (method, seed) run is saved to <out>/raw/ as soon as it finishes, so an
interrupted experiment can simply be restarted (finished runs are skipped).
"""
import argparse, os, pickle, time
import pandas as pd
from config import Config
from train import run_experiment

METHODS = ["sequential", "replay"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=None, help="env steps per task")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    cfg = Config(seeds=args.seeds, out_dir=args.out)
    if args.steps:
        cfg.steps_per_task = args.steps
    os.makedirs(f"{cfg.out_dir}/raw", exist_ok=True)

    t0 = time.time()
    for seed in cfg.seeds:
        for method in METHODS:
            path = f"{cfg.out_dir}/raw/{method}_seed{seed}.pkl"
            if os.path.exists(path):
                print("skip (done):", path); continue
            logs = run_experiment(method, seed, cfg)
            with open(path, "wb") as f:
                pickle.dump(logs, f)
    print(f"all runs finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
