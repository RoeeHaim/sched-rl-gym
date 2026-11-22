"""Continuing-regime training and evaluation of DSMART agents on sched-rl-gym.

``run(config) -> dict`` is the single entry point used by the CLI
(``scripts/train_dsmart.py``) and by sweep drivers. It trains one agent for a
fixed budget of SIMULATED TIME (summed over the parallel environments), and
writes a JSONL log with one record per line:

* ``{"kind": "config", ...}``      the resolved configuration
* ``{"kind": "train", ...}``       one per PPO update (rho, empirical rate,
                                   advantage/critic/entropy diagnostics, tau stats)
* ``{"kind": "eval", ...}``        periodic held-out evaluation
* ``{"kind": "final", ...}``       the returned summary

The headline metric everywhere is the REWARD RATE: ``sum(r) / sum(tau)``, the
reward per unit of simulated time. In this environment the reward is
``-sum(1/execution_time)`` over the jobs in the system per time unit, so the
rate is minus the time-averaged "slowdown load" and is always <= 0 (higher is
better).

Algorithms: ``DS`` / ``DS_relaxed`` / ``DS_smoothed`` (DSMART with the three rho
estimators), ``APO`` (average reward per DECISION, ignores tau), ``PPO``
(discounted, gamma per time unit). Heuristic references that are only
evaluated: ``SJF`` (shortest fitting job), ``FIRST`` (first fitting job),
``RANDOM`` (uniform over valid actions); none of them waits while a job fits
except ``RANDOM``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time as _time
from typing import Any

import gymnasium
import numpy as np
import torch

import schedgym.envs  # noqa: F401  (registers the environments)

from .deep_smart import ALGOS, make_agent
from .ppo import RolloutBuffer

HEURISTICS = ("SJF", "FIRST", "RANDOM")

DEFAULT_WORKLOAD = {
    "type": "deeprm",
    # 0.7 (the DeepRM default) overloads the 10x10 cluster: the queue, and with
    # it the cost rate, grows without bound under every policy, so a long-run
    # average is not defined. 0.3 is a stable load (~0.55 of capacity).
    "new_job_rate": 0.3,
    "max_job_size": 10,
    "max_job_len": 15,
    "small_job_chance": 0.8,
}

DEFAULT_ENV_KWARGS = {
    "simulation_type": "event_based",
    "wait_mode": "unit",
    "time_limit": None,  # continuing: never truncate
    "job_time_feature": "age",  # clock-free job features
    "include_time_left": False,  # no episodic clock in the observation
    "workload": DEFAULT_WORKLOAD,
}

DEFAULTS: dict[str, Any] = {
    "algo": "DS",
    "env_id": "CompactRM-v0",
    "env_kwargs": DEFAULT_ENV_KWARGS,
    "seed": 0,
    "n_envs": 8,
    "batch_T": 256,  # decisions per env per update
    "time_budget": 200_000,  # simulated time units, summed over envs
    # --- DSMART / PPO hyperparameters (paper Table 1 axes first) ---
    "gae_lambda": 0.9,
    "rho_lr": 0.3,
    "rm_vbias_coeff": 0.0,  # v in Eq. 9-10
    "entropy_pressure": 0.02,  # p in Eq. 25
    "a_centering": True,
    "calibrated_pressure": True,
    "entropy_warmup_iters": 20,
    "pressure_track": True,
    "pressure_boost": True,
    # Horizons of the pressure tracker and of the boost, in TIME UNITS. The
    # reference uses 2e6 / 4e5 environment steps for ~1e7-step runs; None scales
    # them to the same fractions of time_budget (1/5 and 1/25).
    "pressure_track_steps": None,
    "boost_efold_steps": None,
    "boost_frac": 0.30,
    "boost_max": 8.0,
    "learning_rate": 3e-4,
    "epochs": 10,
    "minibatches": 20,
    "hidden": [64, 64],
    "clip_grad_norm": 10.0,
    "ratio_clip": 0.2,
    "value_loss_coeff": 1.0,
    "entropy_loss_coeff": None,  # PPO only (DSMART sets it from the pressure)
    "discount": None,  # None: 1.0 for average-reward agents, 0.99 for PPO
    # Overload guard (training envs only). A continuing queue has no natural
    # reset: an early, poor policy lets the backlog grow so large that the
    # training envs never return to the operating region the evaluation
    # measures. When a training env's queue exceeds max_queue it is treated as a
    # time-limit truncation (bootstrapped, masked out of the loss) and reset.
    # None disables it.
    "max_queue": None,
    # --- evaluation ---
    "eval_every": 20_000,  # simulated time units between evaluations
    "eval_time": 5_000,  # time units measured per evaluation rollout
    "eval_warmup": 500,  # time units discarded first (fill the cluster)
    "eval_seeds": [10_001, 10_002],
    # --- output ---
    "log_path": None,
    "save_path": None,
    "device": "cpu",
    "torch_threads": 1,
    "verbose": True,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Copy of ``base`` with ``override`` merged in, recursing into dicts, so
    ``env_kwargs.workload.new_job_rate`` alone keeps the other workload keys."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(config: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (config or {}).items():
        if k == "env_kwargs":
            cfg[k] = _deep_merge(DEFAULT_ENV_KWARGS, v)
            wl = v.get("workload")
            if wl and wl.get("type", DEFAULT_WORKLOAD["type"]) != DEFAULT_WORKLOAD["type"]:
                # another workload model: its keys replace, not extend, the default
                cfg[k]["workload"] = copy.deepcopy(wl)
        elif k not in DEFAULTS:
            raise KeyError(f"unknown config key {k!r}")
        else:
            cfg[k] = v
    if cfg["pressure_track_steps"] is None:
        cfg["pressure_track_steps"] = cfg["time_budget"] / 5
    if cfg["boost_efold_steps"] is None:
        cfg["boost_efold_steps"] = cfg["time_budget"] / 25
    algo = cfg["algo"]
    if algo not in ALGOS and algo not in HEURISTICS:
        raise ValueError(f"unknown algo {algo!r}")
    return cfg


def make_env(cfg: dict):
    env = gymnasium.make(cfg["env_id"], disable_env_checker=True, **cfg["env_kwargs"])
    return env.unwrapped


def build_agent(cfg: dict, obs_dim: int, n_actions: int):
    kwargs = {
        "hidden": tuple(cfg["hidden"]),
        "learning_rate": cfg["learning_rate"],
        "rho_lr": cfg["rho_lr"],
        "rm_vbias_coeff": cfg["rm_vbias_coeff"],
        "value_loss_coeff": cfg["value_loss_coeff"],
        "clip_grad_norm": cfg["clip_grad_norm"],
        "discount": cfg["discount"],
        "gae_lambda": cfg["gae_lambda"],
        "epochs": cfg["epochs"],
        "minibatches": cfg["minibatches"],
        "ratio_clip": cfg["ratio_clip"],
        "batch_T": cfg["batch_T"],
        "device": cfg["device"],
        "seed": cfg["seed"],
        "a_centering": cfg["a_centering"],
        "calibrated_pressure": cfg["calibrated_pressure"],
        "entropy_pressure": cfg["entropy_pressure"],
        "entropy_warmup_iters": cfg["entropy_warmup_iters"],
        "pressure_track": cfg["pressure_track"],
        "pressure_track_steps": cfg["pressure_track_steps"],
        "pressure_boost": cfg["pressure_boost"],
        "boost_efold_steps": cfg["boost_efold_steps"],
        "boost_frac": cfg["boost_frac"],
        "boost_max": cfg["boost_max"],
    }
    if cfg["entropy_loss_coeff"] is not None:
        kwargs["entropy_loss_coeff"] = cfg["entropy_loss_coeff"]
    return make_agent(cfg["algo"], obs_dim, n_actions, **kwargs)


# ---------------------------------------------------------------- policies ---
def heuristic_action(env, name: str, rng: np.random.Generator) -> int:
    mask = env.action_masks()
    if name == "RANDOM":
        return int(rng.choice(np.flatnonzero(mask)))
    fitting = np.flatnonzero(mask[:-1])
    if len(fitting) == 0:
        return len(mask) - 1
    if name == "FIRST":
        return int(fitting[0])
    queue = env.scheduler.queue_admission
    return int(min(fitting, key=lambda i: queue[i].requested_time))  # SJF


class _GlobalRngState:
    """Run a block under its own seed without disturbing training randomness.

    The workload generators draw from the global ``random`` module."""

    def __init__(self, seed: int):
        self.seed = seed

    def __enter__(self):
        self.py, self.np = random.getstate(), np.random.get_state()
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, *exc):
        random.setstate(self.py)
        np.random.set_state(self.np)


def evaluate(cfg: dict, policy, seeds=None) -> dict:
    """Reward rate of ``policy(env, obs) -> action`` on fresh continuing runs.

    Each run first spends ``eval_warmup`` time units filling the cluster from
    empty (not measured), then measures ``eval_time`` time units.
    """
    seeds = cfg["eval_seeds"] if seeds is None else seeds
    R = T = 0.0
    decisions = zero_tau = 0
    taus = []
    slowdowns = []
    queue_end = []
    for seed in seeds:
        with _GlobalRngState(seed):
            env = make_env(cfg)
            obs, _ = env.reset(seed=seed)
            t = 0
            n_completed_at_warmup = None
            while t < cfg["eval_warmup"] + cfg["eval_time"]:
                a = policy(env, obs)
                obs, r, term, trunc, info = env.step(a)
                d = info["duration"]
                if t >= cfg["eval_warmup"]:
                    if n_completed_at_warmup is None:
                        n_completed_at_warmup = len(env.scheduler.queue_completed)
                    R += r
                    T += d
                    decisions += 1
                    zero_tau += d == 0
                    if d > 0:
                        taus.append(d)
                t += d
                if term or trunc:
                    obs, _ = env.reset()
            done = env.scheduler.queue_completed[n_completed_at_warmup or 0 :]
            slowdowns += [j.slowdown for j in done]
            queue_end.append(len(env.scheduler.queue_admission))
    return {
        "reward_rate": R / T if T > 0 else float("nan"),
        "decisions_per_time": decisions / T if T > 0 else float("nan"),
        "frac_tau0": zero_tau / max(decisions, 1),
        "tau_mean_pos": float(np.mean(taus)) if taus else float("nan"),
        "tau_max": float(np.max(taus)) if taus else float("nan"),
        "mean_slowdown": float(np.mean(slowdowns)) if slowdowns else float("nan"),
        "jobs_completed": len(slowdowns),
        "queue_end_mean": float(np.mean(queue_end)),
        "time": T,
    }


def agent_policy(agent, greedy: bool):
    def policy(env, obs):
        mask = env.action_masks()
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(mask).unsqueeze(0)
        if greedy:
            return int(agent.eval_act(o, m)[0])
        return int(agent.act(o, m)[0][0])

    return policy


# ---------------------------------------------------------------- training ---
class _Logger:
    def __init__(self, path, verbose):
        self.f = None
        self.verbose = verbose
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.f = open(path, "w")

    def __call__(self, rec: dict):
        if self.f:
            self.f.write(json.dumps(rec, default=_json_default) + "\n")
            self.f.flush()

    def print(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def close(self):
        if self.f:
            self.f.close()


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _clean(d: dict) -> dict:
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in d.items()
    }


def run(config: dict | None = None) -> dict:
    """Train (or, for a heuristic, only evaluate) one configuration."""
    cfg = resolve_config(config)
    torch.set_num_threads(int(cfg["torch_threads"]))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    log = _Logger(cfg["log_path"], cfg["verbose"])
    log({"kind": "config", **cfg})

    if cfg["algo"] in HEURISTICS:
        rng = np.random.default_rng(cfg["seed"])
        res = evaluate(cfg, lambda env, obs: heuristic_action(env, cfg["algo"], rng))
        final = {"kind": "final", "algo": cfg["algo"], "sim_time": 0,
                 **{f"eval_{k}": v for k, v in res.items()}}
        log(_clean(final))
        log.print(f"[{cfg['algo']}] reward_rate={res['reward_rate']:.4f} "
                  f"slowdown={res['mean_slowdown']:.2f}")
        log.close()
        return final

    envs = [make_env(cfg) for _ in range(cfg["n_envs"])]
    obs = np.stack(
        [e.reset(seed=cfg["seed"] * 1000 + i)[0] for i, e in enumerate(envs)]
    ).astype(np.float32)
    obs_dim = obs.shape[1]
    n_actions = envs[0].action_space.n
    agent = build_agent(cfg, obs_dim, n_actions)

    sim_time = 0.0
    decisions = 0
    cum_R = cum_T = 0.0
    it = 0
    next_eval = 0.0
    t_start = _time.time()
    last_eval: dict = {}
    eval_curve = []

    def do_eval():
        nonlocal last_eval
        g = evaluate(cfg, agent_policy(agent, greedy=True))
        s = evaluate(cfg, agent_policy(agent, greedy=False))
        rec = {"kind": "eval", "iter": it, "sim_time": sim_time,
               **{f"greedy_{k}": v for k, v in g.items()},
               **{f"sample_{k}": v for k, v in s.items()}}
        log(_clean(rec))
        eval_curve.append((sim_time, g["reward_rate"], s["reward_rate"]))
        last_eval = rec
        log.print(f"  eval @ {sim_time:>10.0f}: greedy {g['reward_rate']:.4f} "
                  f"(slowdown {g['mean_slowdown']:.2f})  sample {s['reward_rate']:.4f}")

    while sim_time < cfg["time_budget"]:
        if sim_time >= next_eval:
            do_eval()
            next_eval += cfg["eval_every"]
        buf = RolloutBuffer()
        batch_R = batch_T = 0.0
        resets = overloads = 0
        for _ in range(cfg["batch_T"]):
            masks = np.stack([e.action_masks() for e in envs])
            a, v, logp = agent.act(obs, masks)
            rew = np.zeros(len(envs), dtype=np.float32)
            dur = np.zeros(len(envs), dtype=np.float32)
            term = np.zeros(len(envs), dtype=np.float32)
            trunc = np.zeros(len(envs), dtype=np.float32)
            nxt = np.empty_like(obs)
            for i, e in enumerate(envs):
                o, r, te, tr, info = e.step(int(a[i]))
                rew[i], dur[i], term[i], trunc[i] = r, info["duration"], te, tr
                if (cfg["max_queue"] is not None and not te
                        and len(e.scheduler.queue_admission) > cfg["max_queue"]):
                    tr = True
                    trunc[i] = 1.0
                    overloads += 1
                if te or tr:
                    o, _ = e.reset()
                    resets += 1
                nxt[i] = o
            buf.add(obs, a, rew, term, trunc, v, logp, time=dur, mask=masks)
            obs = nxt
            batch_R += float(rew.sum())
            batch_T += float(dur.sum())
        sim_time += batch_T
        decisions += cfg["batch_T"] * len(envs)
        cum_R += batch_R
        cum_T += batch_T

        times = torch.stack([torch.as_tensor(t) for t in buf._d["time"]])
        stats = agent.update(buf, agent.value(obs))
        it += 1
        rec = {
            "kind": "train", "iter": it, "sim_time": sim_time, "decisions": decisions,
            "emp_rate": batch_R / batch_T if batch_T > 0 else float("nan"),
            "cum_rate": cum_R / cum_T if cum_T > 0 else float("nan"),
            "tau_mean": float(times.mean()),
            "tau_mean_pos": float(times[times > 0].mean()) if (times > 0).any() else float("nan"),
            "frac_tau0": float((times == 0).float().mean()),
            "resets": resets,
            "overloads": overloads,
            "queue_mean": float(np.mean([len(e.scheduler.queue_admission) for e in envs])),
            "wall": _time.time() - t_start,
            **stats,
        }
        log(_clean(rec))
        if it % 10 == 0 or it == 1:
            log.print(
                f"[{cfg['algo']}] it {it:4d} t {sim_time:>9.0f} rho {stats['rho']:+.4f} "
                f"emp {rec['emp_rate']:+.4f} cum {rec['cum_rate']:+.4f} "
                f"H {stats['entropy']:.3f} cH {stats.get('entropy_coeff', float('nan')):.2e} "
                f"A {stats['adv_mean_raw']:+.3f}->{stats['adv_mean']:+.1e} "
                f"sdA {stats.get('adv_std', float('nan')):.3f} "
                f"V [{stats['v_min']:+.2f},{stats['v_max']:+.2f}] b {stats['value_bias']:+.2f}"
            )

    do_eval()
    if cfg["save_path"]:
        os.makedirs(os.path.dirname(os.path.abspath(cfg["save_path"])), exist_ok=True)
        torch.save({"config": cfg, "state_dict": agent.net.state_dict(),
                    "rho": agent.rho, "value_bias": agent.value_bias},
                   cfg["save_path"])
    tail = [g for (_, g, _) in eval_curve[-3:]]
    tail_s = [s for (_, _, s) in eval_curve[-3:]]
    final = {
        "kind": "final", "algo": cfg["algo"], "iters": it, "sim_time": sim_time,
        "decisions": decisions, "train_cum_rate": cum_R / cum_T if cum_T else None,
        "rho": agent.rho, "wall": _time.time() - t_start,
        "eval_reward_rate": last_eval.get("greedy_reward_rate"),
        "eval_sample_reward_rate": last_eval.get("sample_reward_rate"),
        "eval_mean_slowdown": last_eval.get("greedy_mean_slowdown"),
        "eval_reward_rate_last3": float(np.mean(tail)) if tail else None,
        # The sampled policy is the robust headline: argmax can lock into
        # "wait" forever (a fitting job is never started, the state barely
        # changes), which the stochastic policy escapes.
        "eval_sample_reward_rate_last3": float(np.mean(tail_s)) if tail_s else None,
    }
    log(_clean(final))
    log.close()
    return final
