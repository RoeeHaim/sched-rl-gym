"""Shared helpers for training and evaluating PPO agents on CompactRM-v0.

Both scripts/train_ppo.py and scripts/evaluate.py build the environment through
make_env(), so the agent is always evaluated in exactly the environment it was
trained in.
"""

import random

import gymnasium
import numpy as np

import schedgym.envs  # noqa: F401  (registers the environments)

DEFAULT_CONFIG = {
    "sim": "event_based",  # SMDP stepping, as in Cunha & Chaimowicz
    "slots": 10,           # job slots visible to the agent (actions = slots + 1)
    "time_limit": 200,
    "processors": 10,
    "memory": False,       # memory modelling off by default (bugs 11-13)
    "rate": 0.7,           # DeepRM job arrival rate
    "estimates": None,     # None: requested_time == execution_time (exact estimates)
    "estimate_noise": 0.5, # std of the estimate error, relative to the run time
}


def env_kwargs(cfg: dict) -> dict:
    """gymnasium.make() keyword arguments for CompactRM-v0 from a config dict."""
    return {
        "simulation_type": cfg["sim"],
        "job_slots": cfg["slots"],
        "time_limit": cfg["time_limit"],
        "processors": cfg["processors"],
        "memory": cfg["processors"],
        "ignore_memory": not cfg["memory"],
        "backlog_size": max(60, cfg["slots"] * 20),
        "workload": {
            "type": "deeprm",
            "new_job_rate": cfg["rate"],
            "max_job_size": cfg["processors"],
            "max_job_len": 15,
            "small_job_chance": 0.8,
            "ignore_memory": not cfg["memory"],
            **estimate_kwargs(cfg),
        },
        "disable_env_checker": True,
    }


def estimate_kwargs(cfg: dict) -> dict:
    """DeepRM workload arguments for inaccurate run-time estimates.

    With estimates the job's requested_time differs from its execution_time,
    which is what real users submit. Older configs have no 'estimates' key and
    keep exact estimates.
    """
    if not cfg.get("estimates"):
        return {}
    return {
        "runtime_estimates": cfg["estimates"],
        "estimate_parameters": cfg.get("estimate_noise", DEFAULT_CONFIG["estimate_noise"]),
    }


class SeedWorkload(gymnasium.Wrapper):
    """Workaround for bug 8.

    env.reset(seed=...) does not seed the job generator, which draws from
    Python's global `random`. This wrapper seeds `random` on every reset:
    with the given seed, or with a fresh one from its own generator, so that
    every episode gets a different but reproducible job sequence.
    """

    def __init__(self, env, base_seed: int = 0):
        super().__init__(env)
        self._rng = np.random.default_rng(base_seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = int(self._rng.integers(2**31 - 1))
        random.seed(seed)
        return self.env.reset(seed=seed, options=options)


def action_mask(env) -> np.ndarray:
    """Which actions are valid right now.

    Action i (0 <= i < slots) is valid only if the i-th job in the queue exists
    and can start now. The last action ("wait") is always valid.
    """
    u = env.unwrapped
    s = u.scheduler
    mask = np.zeros(u.job_slots + 1, dtype=bool)
    for i, job in enumerate(s.queue_admission[: u.job_slots]):
        mask[i] = bool(s.can_schedule_now(job))
    mask[-1] = True
    return mask


def make_env(cfg: dict, base_seed: int = 0):
    """CompactRM-v0 wrapped with the bug-8 seeding fix and the action mask."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = gymnasium.make("CompactRM-v0", **env_kwargs(cfg))
    env = SeedWorkload(env, base_seed)
    return ActionMasker(env, action_mask)
