"""Central configuration for the lifelong-learning (continual RL) experiment."""
from dataclasses import dataclass, field
from typing import List

# Three MiniGrid tasks learned sequentially. All share the same 7-action
# discrete action space and the same 7x7 egocentric observation format, so one
# single-head network can be trained on all of them without a task identifier.
TASKS: List[str] = [
    "MiniGrid-Empty-Random-5x5-v0",   # Task 1: navigate to the goal square
    "MiniGrid-LavaGapS5-v0",          # Task 2: reach goal through the gap in a lava wall
    "MiniGrid-DoorKey-5x5-v0",        # Task 3: pick up key, open door, reach goal
]
TASK_SHORT = ["Empty", "LavaGap", "DoorKey"]


@dataclass
class Config:
    # ---- experiment layout ----
    tasks: List[str] = field(default_factory=lambda: list(TASKS))
    steps_per_task: int = 50_000       # environment steps spent on each task
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    # ---- DQN hyper-parameters ----
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    hidden: int = 128
    train_freq: int = 2                # one gradient step every N env steps
    target_update: int = 1000          # gradient steps between hard target syncs (if tau == 0)
    tau: float = 0.005                  # >0 -> soft (Polyak) target updates every gradient step
    learning_starts: int = 1_000
    eps_start_first: float = 1.0       # epsilon at start of task 1
    eps_start_later: float = 0.5       # epsilon at start of tasks 2,3 (same for both methods)
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4        # fraction of the task's steps over which eps decays
    # ---- experience replay ----
    buffer_capacity: int = 30_000      # per task
    replay_ratio: float = 0.5          # fraction of each batch drawn from OLD tasks
    # ---- evaluation ----
    eval_episodes_final: int = 100     # at end of every stage (performance matrix)
    eval_episodes_periodic: int = 20   # during training (fine-grained curves)
    eval_every: int = 5_000
    out_dir: str = "results"
