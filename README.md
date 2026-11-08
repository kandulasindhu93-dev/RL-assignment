# Lifelong Learning Without Forgetting (MiniGrid, Double DQN)

## Team Members

| S.No | Name | Roll Number |
|---|---|---|
| 1 | C. Sanya | 100523729011 |
| 2 | K. Sindhu | 100523729029 |
| 3 | Nissi | 100523729001 |
| 4 | Ch. Vaishnavi | 100523729012 |

A single Double-DQN agent learns three MiniGrid tasks **sequentially** (Empty-Random-5x5, LavaGapS5, DoorKey-5x5).
Two methods are compared: **Sequential** (replay buffer cleared at each task switch) and **Sequential + Experience Replay**
(one buffer per task kept; each batch is 50% old-task / 50% current-task transitions). After every stage all three tasks are evaluated.

## Setup
```bash
pip install -r requirements.txt
python main.py --seeds 0 1 2        # ~30 min on 1 CPU core; saves results/raw/*.pkl (restartable)
python baseline.py                  # random-policy baseline -> results/baseline_random.csv
python analysis.py                  # writes CSVs, Learning_Curves.png and results/figures/*
```

## Files
| File | Purpose |
|---|---|
| config.py | tasks and hyper-parameters |
| envs.py | MiniGrid env + ShapedReward + one-hot observation wrapper |
| agent.py | Double DQN (MLP 588-128-128-7) |
| replay.py | FIFO buffer and TaskReplay (sequential / replay modes) |
| train.py | sequential training and evaluation after every stage |
| evaluate.py | greedy evaluation with the native reward |
| baseline.py | random-policy baseline (uniform random actions) |
| main.py / analysis.py | run experiments / metrics, CSVs, figures |

## Results (from the executed runs; success rate, mean ± std over 3 seeds)
| Metric | Sequential | Seq. + Replay |
|---|---|---|
| Average final performance | 0.120 ± 0.105 | 0.464 ± 0.034 |
| Average forgetting (Empty, LavaGap) | 0.992 ± 0.014 | 0.402 ± 0.137 |
| Backward transfer | -0.992 ± 0.014 | -0.393 ± 0.142 |
| Final success: Empty | 0.000 ± 0.000 | 0.963 ± 0.064 |
| Final success: LavaGap | 0.000 ± 0.000 | 0.233 ± 0.214 |
| Final success: DoorKey | 0.360 ± 0.316 | 0.197 ± 0.341 |

**Random-policy baseline** (same evaluation protocol): average success 0.228 ± 0.011 (Empty 0.527, LavaGap 0.077, DoorKey 0.080). Replay ends above it; the no-replay agent ends below it.

Output files: `results/Experiment_Results.csv` (end-of-stage evaluation of every task, every run), `metrics_per_run.csv`,
`metrics_summary_*.csv`, `matrix_*.csv`, `training_episodes.csv`, `periodic_eval.csv`, `Learning_Curves.png`, `figures/`.

## Metrics
R[i,j] = success rate on task j after stage i. Forgetting(j) = max earlier R[i,j] - final R[K,j]. Backward transfer = mean(final R[K,j] - R[j,j]) for tasks 1-2. Average final performance = mean final success over all tasks.

## Simplifications / caveats (please read)
* **Reward shaping during training only**: -0.01/step, -1 on lava, +0.25 key pick-up, +0.25 door opening. Plain DQN on the sparse native reward did not learn LavaGap/DoorKey in our budget. All evaluation uses the native reward.
* Learning rate, soft target updates (tau=0.005) and buffer size were chosen in short single-task pilot runs (seed 0) and used unchanged for both methods.
* Only 3 seeds; DoorKey results are very seed-dependent (0.00-0.59), so differences on DoorKey are not statistically reliable.
* Replay reduces forgetting of Empty but LavaGap is still mostly forgotten; replay did not improve learning of the new task (DoorKey).
