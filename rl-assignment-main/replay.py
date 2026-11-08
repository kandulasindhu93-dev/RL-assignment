"""Replay buffers: a plain FIFO buffer and a multi-task rehearsal buffer."""
import numpy as np


class ReplayBuffer:
    """Fixed-size FIFO buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, obs_dim: int):
        self.cap, self.n, self.ptr = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k: int, rng):
        idx = rng.integers(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


class TaskReplay:
    """Holds one buffer per task.

    mode == "sequential": only the current task's buffer is used and it is
        emptied when a new task starts (naive fine-tuning, no memory).
    mode == "replay": buffers of earlier tasks are kept; each training batch
        is a mix of `replay_ratio` old-task samples (split evenly across old
        tasks) and the rest from the current task (experience replay / rehearsal).
    """

    def __init__(self, mode: str, capacity: int, obs_dim: int, replay_ratio: float, rng):
        assert mode in ("sequential", "replay")
        self.mode, self.cap, self.dim = mode, capacity, obs_dim
        self.ratio, self.rng = replay_ratio, rng
        self.buffers = []          # one ReplayBuffer per task seen so far

    def start_task(self):
        if self.mode == "sequential":
            self.buffers = []      # forget all previous experience
        self.buffers.append(ReplayBuffer(self.cap, self.dim))

    @property
    def current(self):
        return self.buffers[-1]

    def add(self, *transition):
        self.current.add(*transition)

    def sample(self, batch_size: int):
        old = [b for b in self.buffers[:-1] if len(b) > 0]
        if self.mode == "sequential" or not old:
            return self.current.sample(batch_size, self.rng)
        n_old = int(round(batch_size * self.ratio))
        parts = [self.current.sample(batch_size - n_old, self.rng)]
        per = np.full(len(old), n_old // len(old))
        per[: n_old - per.sum()] += 1                      # distribute remainder
        for b, k in zip(old, per):
            if k > 0:
                parts.append(b.sample(int(k), self.rng))
        return tuple(np.concatenate(x) for x in zip(*parts))
