"""Greedy evaluation of an agent on a task."""
import numpy as np
from envs import make_env


def evaluate(agent, task_id: str, episodes: int, seed_base: int = 10_000):
    """Run `episodes` greedy episodes with fixed seeds.

    Returns dict with mean return and success rate (success = reward > 0,
    i.e. the goal was reached; lava / timeout give reward 0).
    """
    env = make_env(task_id, shaped=False)   # evaluation uses the native reward
    rets = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs, eps=0.0))
            total += r
            done = term or trunc
        rets.append(total)
    env.close()
    rets = np.array(rets)
    return {"mean_return": float(rets.mean()), "success_rate": float((rets > 0).mean())}
