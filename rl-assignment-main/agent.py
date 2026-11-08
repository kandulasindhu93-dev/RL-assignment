"""Double-DQN agent (PyTorch)."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """One network is shared by ALL tasks (single head, no task id)."""

    def __init__(self, obs_dim, n_actions, cfg, rng):
        self.n_actions, self.cfg, self.rng = n_actions, cfg, rng
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden)
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q = self.q(torch.from_numpy(obs).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def learn(self, batch):
        s, a, r, s2, d = (torch.from_numpy(x) for x in batch)
        with torch.no_grad():                               # Double DQN target
            a2 = self.q(s2).argmax(1, keepdim=True)
            y = r + self.cfg.gamma * (1 - d) * self.target(s2).gather(1, a2).squeeze(1)
        qsa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.cfg.tau > 0:                                # soft target update
            with torch.no_grad():
                for tp, p in zip(self.target.parameters(), self.q.parameters()):
                    tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)
        elif self.updates % self.cfg.target_update == 0:    # hard target update
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def reset_optimizer(self):
        """Fresh Adam moments at each new task (same for both methods)."""
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
