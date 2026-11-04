"""SMDP training for MaskablePPO with a pluggable optimality criterion.

The environment reports, for every decision i, the raw reward of each elapsed
time unit ``info["step_rewards"] = [r_1, ..., r_d]`` and the sojourn time
``info["duration"] = d`` (0 when an action schedules a job without advancing
time). The environment never discounts. The learner turns those facts into
returns, and a ``Criterion`` decides how, through three answers:

    R_i      = criterion.intra(step_rewards_i)         reward credited to decision i
    d(D_i)   = criterion.discount(D_i)                 multiplier on V(s_{i+1})
    c(D_i)   = criterion.offset(D_i)                   additive term (e.g. -rho * D_i)

Stage 1 (inside a transition) happens in ``VecSmdpReward``; stage 2 (between
decisions) in ``SmdpMaskableRolloutBuffer``:

    delta_i  = R_i + c(D_i) + d(D_i) V(s_{i+1}) (1 - done_i) - V(s_i)
    A_i      = delta_i + trace(D_i) (1 - done_i) A_{i+1}
    Return_i = A_i + V(s_i)

Criteria:
    discounted  gamma per time unit: R = sum_k gamma^k r_k, d = gamma^D,
                trace = (gamma * lambda)^D                  (the default)
    paper       Cunha & Chaimowicz: discounted inside a transition only,
                d = 1, trace = lambda (PPO gamma = 1)
    average     long-run reward per time unit: R = sum_k r_k, c = -rho * D,
                d = 1, trace = lambda; rho is estimated every rollout and the
                episode boundary does not cut the bootstrap (continuing task)
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


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
class Criterion:
    """How rewards and elapsed time become returns. Subclasses override what differs."""

    name = "base"
    continuing = False  # True: episode boundaries do not cut the bootstrap

    def intra(self, step_rewards) -> float:
        raise NotImplementedError

    def discount(self, durations: np.ndarray) -> np.ndarray:
        return np.ones_like(durations, dtype=np.float32)

    def trace(self, durations: np.ndarray, gae_lambda: float) -> np.ndarray:
        return np.full_like(durations, gae_lambda, dtype=np.float32)

    def offset(self, durations: np.ndarray) -> np.ndarray:
        return np.zeros_like(durations, dtype=np.float32)

    def update(self, rewards, durations, values, next_values, next_non_terminal) -> None:
        """Per-rollout statistics (e.g. rho). Called before advantages are computed."""

    def stats(self) -> dict:
        return {}


class Discounted(Criterion):
    """Textbook discounted SMDP, gamma per time unit."""

    name = "discounted"

    def __init__(self, gamma: float):
        self.gamma = gamma

    def intra(self, step_rewards) -> float:
        return intra_option_reward(step_rewards, self.gamma)

    def discount(self, durations):
        return np.power(self.gamma, durations).astype(np.float32)

    def trace(self, durations, gae_lambda):
        return np.power(self.gamma * gae_lambda, durations).astype(np.float32)


class Paper(Discounted):
    """Cunha & Chaimowicz: discount inside a transition, PPO gamma = 1 between decisions."""

    name = "paper"

    def discount(self, durations):
        return np.ones_like(durations, dtype=np.float32)

    def trace(self, durations, gae_lambda):
        return np.full_like(durations, gae_lambda, dtype=np.float32)


class AverageReward(Criterion):
    """Average reward per time unit: delta = R - rho * D + V(s') - V(s)."""

    name = "average"
    continuing = True

    def __init__(self, rho_lr: float = 0.1, rho_method: str = "ratio", rho_init: float = 0.0):
        if rho_method not in ("ratio", "td"):
            raise ValueError("rho_method must be 'ratio' or 'td'")
        self.rho = float(rho_init)
        self.rho_lr = rho_lr
        self.rho_method = rho_method

    def intra(self, step_rewards) -> float:
        return float(np.sum(step_rewards)) if step_rewards else 0.0

    def offset(self, durations):
        return (-self.rho * durations).astype(np.float32)

    def update(self, rewards, durations, values, next_values, next_non_terminal):
        total_time = float(durations.sum())
        if total_time <= 0:
            return  # only zero-duration decisions in this rollout: nothing to learn about rho
        if self.rho_method == "ratio":
            target = float(rewards.sum()) / total_time
            self.rho += self.rho_lr * (target - self.rho)
        else:  # differential TD update, divided by elapsed time
            delta = rewards - self.rho * durations + next_values * next_non_terminal - values
            self.rho += self.rho_lr * float(delta.sum()) / total_time

    def stats(self):
        return {"rho": self.rho}


def make_criterion(name: str, gamma: float, rho_lr: float = 0.1, rho_method: str = "ratio") -> Criterion:
    if name == "discounted":
        return Discounted(gamma)
    if name == "paper":
        return Paper(gamma)
    if name == "average":
        return AverageReward(rho_lr=rho_lr, rho_method=rho_method)
    raise ValueError(f"unknown criterion {name!r}")


# ---------------------------------------------------------------------------
# Stage 1: reward of one transition
# ---------------------------------------------------------------------------
class VecSmdpReward(VecEnvWrapper):
    """Replaces each reward with R_i = criterion.intra(info["step_rewards"]).

    The raw undiscounted reward is kept in ``info["raw_reward"]``. Put a
    ``VecMonitor`` *below* this wrapper so logged episode returns stay raw.
    """

    def __init__(self, venv: VecEnv, criterion: Criterion):
        super().__init__(venv)
        self.criterion = criterion

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.array(rewards, dtype=np.float32, copy=True)
        for i, info in enumerate(infos):
            if "step_rewards" not in info:
                raise KeyError("the environment must report info['step_rewards'] and info['duration']")
            info["raw_reward"] = float(rewards[i])
            rewards[i] = self.criterion.intra(info["step_rewards"])
        return obs, rewards, dones, infos


class _DurationRecorder(VecEnvWrapper):
    """Thin pass-through that hands each step's durations to the rollout buffer.

    ``MaskablePPO.collect_rollouts`` calls ``env.step`` and then
    ``rollout_buffer.add`` for the same transition, so the buffer can pick the
    durations up in ``add``.
    """

    def __init__(self, venv: VecEnv, buffer: "SmdpMaskableRolloutBuffer"):
        super().__init__(venv)
        self.buffer = buffer

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.buffer.pending_taus = np.array([info["duration"] for info in infos], dtype=np.float32)
        return obs, rewards, dones, infos


# ---------------------------------------------------------------------------
# Stage 2: returns and advantages between decisions
# ---------------------------------------------------------------------------
class SmdpMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Rollout buffer that stores each transition's duration and uses the criterion in GAE.

    Without an explicit criterion it uses ``Discounted(self.gamma)``.
    """

    criterion: Criterion | None = None

    def reset(self) -> None:
        self.taus = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pending_taus = None
        super().reset()

    def add(self, *args, **kwargs) -> None:
        if self.pending_taus is None:
            raise RuntimeError("duration missing: the buffer must be filled by SmdpMaskablePPO")
        self.taus[self.pos] = self.pending_taus
        self.pending_taus = None
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values, dones) -> None:
        criterion = self.criterion if self.criterion is not None else Discounted(self.gamma)
        last_values = last_values.clone().cpu().numpy().flatten()

        next_values = np.empty_like(self.values)
        next_non_terminal = np.empty_like(self.values)
        next_values[:-1] = self.values[1:]
        next_values[-1] = last_values
        next_non_terminal[:-1] = 1.0 - self.episode_starts[1:]
        next_non_terminal[-1] = 1.0 - dones.astype(np.float32)
        if criterion.continuing:
            # one long chain: the state after a reset is simply the next state
            next_non_terminal[:] = 1.0

        criterion.update(self.rewards, self.taus, self.values, next_values, next_non_terminal)
        discount = criterion.discount(self.taus)
        trace = criterion.trace(self.taus, self.gae_lambda)
        offset = criterion.offset(self.taus)

        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            delta = (
                self.rewards[step]
                + offset[step]
                + discount[step] * next_values[step] * next_non_terminal[step]
                - self.values[step]
            )
            last_gae = delta + trace[step] * next_non_terminal[step] * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values


class SmdpMaskablePPO(MaskablePPO):
    """MaskablePPO with SMDP returns.

    Train on an env wrapped in ``VecSmdpReward`` with the *same* criterion.
    Without ``criterion`` it uses ``Discounted(gamma)``.
    """

    def __init__(self, *args, criterion: Criterion | None = None, **kwargs):
        kwargs.setdefault("rollout_buffer_class", SmdpMaskableRolloutBuffer)
        self.criterion = criterion
        super().__init__(*args, **kwargs)
        if self.criterion is None:
            self.criterion = Discounted(self.gamma)
        self.rollout_buffer.criterion = self.criterion

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        result = super().collect_rollouts(
            _DurationRecorder(env, rollout_buffer), callback, rollout_buffer, n_rollout_steps, use_masking
        )
        for key, value in self.criterion.stats().items():
            self.logger.record(f"criterion/{key}", value)
        return result
