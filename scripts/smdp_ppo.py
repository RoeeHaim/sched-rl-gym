"""SMDP (time-aware) discounting for MaskablePPO.

The environment reports, for every decision i, the raw rewards of each elapsed
time unit ``info["step_rewards"] = [r_1, ..., r_tau]`` and the sojourn time
``info["tau"]`` (0 when an action schedules a job without advancing time).
Discounting is done here, on the agent side, in two stages:

Stage 1 (inside a transition), in ``VecSmdpReward``:
    R_i = sum_{k=0}^{tau_i - 1} gamma^k * r_{i,k+1}

Stage 2 (between decisions), in ``SmdpMaskableRolloutBuffer``:
    delta_i = R_i + gamma^{tau_i} V(s_{i+1}) (1 - done_i) - V(s_i)
    A_i     = delta_i + (gamma * lambda)^{tau_i} (1 - done_i) A_{i+1}
    Return_i = A_i + V(s_i)

``gamma`` is therefore a discount *per time unit*, not per decision: two
decisions taken at the same instant are not discounted relative to each other.
"""

from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


def intra_option_reward(step_rewards, gamma: float) -> float:
    """Stage 1: discounted sum of the per-time-unit rewards of one transition."""
    if not step_rewards:
        return 0.0
    r = np.asarray(step_rewards, dtype=np.float64)
    return float(np.dot(gamma ** np.arange(len(r)), r))


class VecSmdpReward(VecEnvWrapper):
    """Replaces each reward with R_i (stage 1) computed from ``info["step_rewards"]``.

    The raw undiscounted reward is kept in ``info["raw_reward"]``. Put a
    ``VecMonitor`` *below* this wrapper so logged episode returns stay raw.
    """

    def __init__(self, venv: VecEnv, gamma: float):
        super().__init__(venv)
        self.gamma = gamma

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.array(rewards, dtype=np.float32, copy=True)
        for i, info in enumerate(infos):
            if "step_rewards" not in info:
                raise KeyError("the environment must report info['step_rewards'] and info['tau']")
            info["raw_reward"] = float(rewards[i])
            rewards[i] = intra_option_reward(info["step_rewards"], self.gamma)
        return obs, rewards, dones, infos


class _TauRecorder(VecEnvWrapper):
    """Thin pass-through that hands each step's taus to the rollout buffer.

    ``MaskablePPO.collect_rollouts`` calls ``env.step`` and then
    ``rollout_buffer.add`` for the same transition, so the buffer can pick the
    taus up in ``add``.
    """

    def __init__(self, venv: VecEnv, buffer: "SmdpMaskableRolloutBuffer"):
        super().__init__(venv)
        self.buffer = buffer

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.buffer.pending_taus = np.array([info["tau"] for info in infos], dtype=np.float32)
        return obs, rewards, dones, infos


class SmdpMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Rollout buffer that stores tau_i and uses it in GAE (stage 2)."""

    def reset(self) -> None:
        self.taus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pending_taus = None
        super().reset()

    def add(self, *args, **kwargs) -> None:
        if self.pending_taus is None:
            raise RuntimeError("tau missing: the buffer must be filled by SmdpMaskablePPO")
        self.taus[self.pos] = self.pending_taus
        self.pending_taus = None
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values, dones) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            discount = self.gamma ** self.taus[step]
            trace = (self.gamma * self.gae_lambda) ** self.taus[step]
            delta = self.rewards[step] + discount * next_values * next_non_terminal - self.values[step]
            last_gae = delta + trace * next_non_terminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values


class SmdpMaskablePPO(MaskablePPO):
    """MaskablePPO with SMDP returns. Train on an env wrapped in ``VecSmdpReward``."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", SmdpMaskableRolloutBuffer)
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        return super().collect_rollouts(
            _TauRecorder(env, rollout_buffer), callback, rollout_buffer, n_rollout_steps, use_masking
        )
