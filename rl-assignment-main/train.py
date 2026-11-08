"""Training pipeline: sequential multi-task DQN with or without experience replay."""
import time
import numpy as np
import torch

from agent import DQNAgent
from config import Config, TASK_SHORT
from envs import make_env, obs_dim_and_actions
from evaluate import evaluate
from replay import TaskReplay


def epsilon(step: int, total: int, start: float, cfg: Config) -> float:
    """Linear decay from `start` to cfg.eps_end over eps_decay_frac of the task."""
    frac = min(1.0, step / (cfg.eps_decay_frac * total))
    return start + frac * (cfg.eps_end - start)


def train_task(agent, memory, task_idx, cfg, seed, logs, method, global_step, verbose=True):
    """Train the (already partly trained) agent on one task for cfg.steps_per_task steps."""
    task_id = cfg.tasks[task_idx]
    env = make_env(task_id)
    memory.start_task()
    agent.reset_optimizer()
    eps0 = cfg.eps_start_first if task_idx == 0 else cfg.eps_start_later
    obs, _ = env.reset(seed=seed * 1000 + task_idx)
    ep_ret, ep_native, ep_len, ep_count = 0.0, 0.0, 0, 0

    for t in range(1, cfg.steps_per_task + 1):
        eps = epsilon(t, cfg.steps_per_task, eps0, cfg)
        a = agent.act(obs, eps)
        obs2, r, term, trunc, info = env.step(a)
        # only true termination stops bootstrapping (time-out is not terminal)
        memory.add(obs, a, r, obs2, float(term))
        obs, ep_ret, ep_len = obs2, ep_ret + r, ep_len + 1
        ep_native += info["native_reward"]
        global_step += 1

        if len(memory.current) >= cfg.learning_starts and t % cfg.train_freq == 0:
            agent.learn(memory.sample(cfg.batch_size))

        if term or trunc:                                   # episode finished
            ep_count += 1
            logs["episodes"].append(dict(
                method=method, seed=seed, stage=task_idx + 1, task=TASK_SHORT[task_idx],
                episode_in_stage=ep_count, global_step=global_step, ret=ep_native,
                shaped_ret=ep_ret, success=int(ep_native > 0), length=ep_len, epsilon=eps))
            obs, _ = env.reset()
            ep_ret, ep_native, ep_len = 0.0, 0.0, 0

        # fine-grained evaluation on ALL tasks during training
        if t % cfg.eval_every == 0 and t < cfg.steps_per_task:
            for j, tid in enumerate(cfg.tasks):
                m = evaluate(agent, tid, cfg.eval_episodes_periodic)
                logs["periodic"].append(dict(method=method, seed=seed, stage=task_idx + 1,
                                             global_step=global_step, eval_task=TASK_SHORT[j],
                                             eval_task_idx=j + 1, **m))
    env.close()
    return global_step


def run_experiment(method: str, seed: int, cfg: Config, verbose: bool = True):
    """Sequentially train on all tasks; evaluate every task after every stage.

    Returns a dict of logs: per-episode training logs, periodic evals, and the
    end-of-stage performance matrix entries.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim, n_act = obs_dim_and_actions(cfg.tasks[0])
    agent = DQNAgent(obs_dim, n_act, cfg, rng)
    memory = TaskReplay(method, cfg.buffer_capacity, obs_dim, cfg.replay_ratio, rng)
    logs = {"episodes": [], "periodic": [], "matrix": []}
    gstep, t0 = 0, time.time()

    for k in range(len(cfg.tasks)):
        gstep = train_task(agent, memory, k, cfg, seed, logs, method, gstep, verbose)
        for j, tid in enumerate(cfg.tasks):                 # evaluate ALL tasks after stage k
            m = evaluate(agent, tid, cfg.eval_episodes_final)
            logs["matrix"].append(dict(method=method, seed=seed, after_stage=k + 1,
                                       eval_task=TASK_SHORT[j], eval_task_idx=j + 1, **m))
        if verbose:
            row = [f"{TASK_SHORT[j]}={logs['matrix'][-3 + j]['success_rate']:.2f}" for j in range(3)]
            print(f"[{method:10s} seed={seed}] after stage {k+1} ({TASK_SHORT[k]}): "
                  + "  ".join(row) + f"   ({time.time()-t0:.0f}s)", flush=True)
    return logs
