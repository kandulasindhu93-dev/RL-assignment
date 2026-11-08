"""Environment setup: MiniGrid tasks wrapped to a common flat observation."""
import gymnasium as gym
import numpy as np
import minigrid  # noqa: F401  (registers the MiniGrid-* environments)
from gymnasium import spaces

N_OBJECTS = 11   # MiniGrid object types (wall, floor, door, key, goal, lava, ...)
N_STATES = 3     # door state (open / closed / locked)


class OneHotObs(gym.ObservationWrapper):
    """7x7x3 egocentric grid -> flat vector.

    For each of the 49 visible cells we use a one-hot of the object type (11)
    plus the door-state channel scaled to [0,1]  ->  49 * 12 = 588 features.
    The agent direction is implicit because the view is egocentric.
    """

    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space["image"].shape
        self.dim = h * w * (N_OBJECTS + 1)
        self.observation_space = spaces.Box(0.0, 1.0, (self.dim,), np.float32)

    def observation(self, obs):
        img = obs["image"]
        obj = img[..., 0].astype(np.int64)
        onehot = np.eye(N_OBJECTS, dtype=np.float32)[obj]          # (h,w,11)
        state = (img[..., 2].astype(np.float32) / (N_STATES - 1))[..., None]
        return np.concatenate([onehot, state], axis=-1).reshape(-1)


class ShapedReward(gym.Wrapper):
    """Dense reward shaping (a documented simplification of the native sparse reward).

    Native MiniGrid reward is only given when the goal is reached
    (1 - 0.9 * t / T). With plain DQN this was too sparse / unstable for
    LavaGap and DoorKey in our small compute budget, so for TRAINING we add:
        * -0.01 per step              (discourages stalling until time-out)
        * -1.0 when the episode terminates without the goal (= lava)
        * +0.25 the first time the key is picked up      (DoorKey)
        * +0.25 the first time a door is opened          (DoorKey)
    The untouched native reward is returned in info["native_reward"]; ALL
    evaluation metrics (return, success rate) use the native reward only.
    """

    def __init__(self, env, step_penalty=0.01, lava_penalty=1.0, subgoal_bonus=0.25):
        super().__init__(env)
        self.sp, self.lp, self.sb = step_penalty, lava_penalty, subgoal_bonus

    def reset(self, **kw):
        self.got_key = self.opened = False
        return self.env.reset(**kw)

    def _any_door_open(self):
        g = self.env.unwrapped.grid.grid
        return any(o is not None and o.type == "door" and o.is_open for o in g)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        shaped = r - self.sp
        if term and r == 0:
            shaped -= self.lp
        u = self.env.unwrapped
        if not self.got_key and u.carrying is not None and u.carrying.type == "key":
            self.got_key, shaped = True, shaped + self.sb
        if not self.opened and self._any_door_open():
            self.opened, shaped = True, shaped + self.sb
        info["native_reward"] = r
        return obs, shaped, term, trunc, info


def make_env(task_id: str, shaped: bool = True) -> gym.Env:
    env = gym.make(task_id)
    if shaped:
        env = ShapedReward(env)
    return OneHotObs(env)


def obs_dim_and_actions(task_id: str):
    env = make_env(task_id)
    d, a = env.observation_space.shape[0], env.action_space.n
    env.close()
    return d, a
