"""Evaluate scheduling policies on the simulator and print the standard metrics.

Two groups of policies are evaluated on the same job sequences (same seeds):

1. Environment policies: they act through env.step(), exactly like an RL agent
   (same job-slot window, same SMDP/time-based stepping). This is the fair
   comparison for a trained agent.
2. Standalone heuristics (FIFO, SJF, EASY, ...): classic schedulers that see
   the whole queue and run directly on the scheduler, without the environment.

Usage (from the sched-rl-gym folder):
    python scripts/evaluate.py
    python scripts/evaluate.py --seeds 20 --sim event_based --slots 5
    python scripts/evaluate.py --workload lublin --processors 64 --jobs 300
    python scripts/evaluate.py --csv results.csv
    python scripts/evaluate.py --model models/ppo_quick     # add a trained PPO agent
"""

import argparse
import copy
import csv
import json
import random
import statistics as st
import warnings

import gymnasium
import numpy as np

import schedgym.envs  # noqa: F401  (registers the environments)
from schedgym.envs.workload import build as build_workload
from ppo_utils import action_mask, env_kwargs
from schedgym.scheduler import (
    BackfillingScheduler,
    EasyScheduler,
    FifoScheduler,
    PackerScheduler,
    RandomScheduler,
    SjfScheduler,
)

warnings.filterwarnings("ignore")

STANDALONE = {
    "FIFO": FifoScheduler,
    "SJF (whole queue)": SjfScheduler,
    "EASY": EasyScheduler,
    "Backfilling": BackfillingScheduler,
    "Packer": PackerScheduler,
    "Random": RandomScheduler,
}


# --------------------------------------------------------------- metrics ---
def metrics(scheduler, now, total_procs):
    """Metrics over ALL jobs that entered the system (unfinished jobs count with
    the time they have waited so far, so starvation is not hidden)."""
    done = scheduler.queue_completed
    pending = scheduler.jobs_in_system
    bsld = [j.bounded_slowdown for j in done] + [
        max(1.0, (now - j.submission_time) / max(10, j.execution_time)) for j in pending
    ]
    # a pending job has waited until it started, or until now if it has not
    # started yet (a reservation in the future does not count as started)
    waits = [j.wait_time for j in done] + [
        (j.start_time if 0 <= j.start_time <= now else now) - j.submission_time
        for j in pending
    ]
    work = sum(min(j.execution_time, j.requested_time) * j.requested_processors for j in done)
    everyone = done + pending
    start = min((j.submission_time for j in everyone), default=0)
    horizon = max(now - start, 1)  # duration of the run, not the absolute clock
    return {
        "bsld_mean": st.mean(bsld) if bsld else float("nan"),
        "bsld_median": st.median(bsld) if bsld else float("nan"),
        "bsld_p99": float(np.percentile(bsld, 99)) if bsld else float("nan"),
        "max_wait": max(waits) if waits else 0,
        "completed": len(done),
        "unfinished": len(pending),
        "utilization": work / (total_procs * horizon),
    }


# ------------------------------------------------- environment policies ---
def env_policies():
    def sjf(env, rng, obs):
        u = env.unwrapped
        return u.scheduler.sjf_action(u.job_slots)

    def rand(env, rng, obs):
        return int(rng.integers(0, env.action_space.n))

    def first_fit(env, rng, obs):
        u = env.unwrapped
        s = u.scheduler
        for i, j in enumerate(s.queue_admission[: u.job_slots]):
            if s.can_schedule_now(j):
                return i
        return u.job_slots  # nothing fits: wait

    return {"sjf_action": sjf, "first_fit": first_fit, "random": rand}


def run_env_policy(policy, args, seed):
    random.seed(seed)  # the workload generator uses the global `random` (bug 8)
    rng = np.random.default_rng(seed)
    kwargs = env_kwargs(config_of(args))  # identical to the training environment
    kwargs["workload"] = workload_config(args, seed)
    env = gymnasium.make("CompactRM-v0", **kwargs)
    obs, _ = env.reset(seed=seed)
    ret, done, steps = 0.0, False, 0
    while not done and steps < 200_000:
        obs, r, done, trunc, _ = env.step(policy(env, rng, obs))
        ret += float(r)
        steps += 1
        done = done or trunc
    s = env.unwrapped.scheduler
    m = metrics(s, s.current_time, args.processors)
    m["return"] = ret
    return m


def config_of(args) -> dict:
    return {
        "sim": args.sim, "slots": args.slots, "time_limit": args.time_limit,
        "processors": args.processors, "memory": args.memory, "rate": args.rate,
    }


def model_policy(model_dir):
    """A trained MaskablePPO agent, acting greedily on valid actions only."""
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(f"{model_dir}/model.zip", device="cpu")

    def act(env, rng, obs):
        action, _ = model.predict(obs, action_masks=action_mask(env), deterministic=True)
        return int(action)

    return act


# ----------------------------------------------- standalone heuristics ---
def workload_config(args, seed):
    if args.workload == "lublin":
        return {"type": "lublin", "length": args.jobs, "nodes": args.processors, "random_seed": seed}
    return {
        "type": "deeprm",
        "new_job_rate": args.rate,
        "max_job_size": args.processors,
        "max_job_len": 15,
        "small_job_chance": 0.8,
        "ignore_memory": not args.memory,
    }


def job_sequence(args, seed):
    """The jobs (with submission times) for one seed, for the standalone runs."""
    random.seed(seed)
    wl = build_workload(workload_config(args, seed))
    if hasattr(wl, "trace"):
        jobs = [copy.copy(j) for j in wl.trace]
        t0 = jobs[0].submission_time
        for j in jobs:
            j.submission_time -= t0
    else:
        jobs, t = [], 0
        while t < args.time_limit:
            t += 1
            for j in wl.step():
                j.submission_time = t
                jobs.append(j)
    for j in jobs:
        if not args.memory:
            j.requested_memory = 0
    return jobs


def run_standalone(cls, args, jobs):
    s = cls(args.processors, args.processors, ignore_memory=not args.memory)
    by_time = {}
    for j in copy.deepcopy(jobs):
        j.resources.ignore_memory = not args.memory
        by_time.setdefault(j.submission_time, []).append(j)
    last = max(by_time) if by_time else 0
    # same horizon as the environment for generated workloads; run to the end for traces
    limit = args.time_limit if args.workload == "deeprm" else 10**9
    t = 0
    while (t <= last or s.jobs_in_system) and t <= limit:
        if t in by_time:
            s.submit(by_time[t])
        s.step(1)
        t += 1
    return metrics(s, s.current_time, args.processors)


# ------------------------------------------------------------------ main ---
def summarize(name, group, runs):
    keys = runs[0].keys()
    row = {"policy": name, "group": group}
    for k in keys:
        row[k] = st.mean(r[k] for r in runs)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of job sequences (default 10)")
    ap.add_argument("--sim", default="event_based", choices=["event_based", "time_based"])
    ap.add_argument("--slots", type=int, default=5, help="job slots visible to env policies")
    ap.add_argument("--time-limit", type=int, default=200)
    ap.add_argument("--workload", default="deeprm", choices=["deeprm", "lublin"])
    ap.add_argument("--rate", type=float, default=0.7, help="DeepRM job arrival rate")
    ap.add_argument("--jobs", type=int, default=300, help="Lublin: number of jobs")
    ap.add_argument("--processors", type=int, default=10)
    ap.add_argument("--memory", action="store_true", help="model memory too (off by default: bugs 11-13)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    ap.add_argument("--model", default=None,
                    help="folder of a model trained with scripts/train_ppo.py; its environment "
                         "settings override --sim/--slots/--time-limit/--processors/--memory/--rate")
    args = ap.parse_args()
    if args.model:
        with open(f"{args.model}/config.json") as f:
            cfg = json.load(f)["env"]
        args.sim, args.slots, args.time_limit = cfg["sim"], cfg["slots"], cfg["time_limit"]
        args.processors, args.memory, args.rate = cfg["processors"], cfg["memory"], cfg["rate"]
        args.workload = "deeprm"
    if args.workload == "lublin":
        args.time_limit = None

    rows = []
    policies = env_policies()
    if args.model:
        policies = {"PPO agent": model_policy(args.model), **policies}
    for name, pol in policies.items():
        rows.append(summarize(name, "env", [run_env_policy(pol, args, s) for s in range(args.seeds)]))
    # generate each job sequence ONCE and give the very same jobs to every
    # standalone heuristic (the Lublin generator is not reproducible, bug 14)
    sequences = [job_sequence(args, s) for s in range(args.seeds)]
    for name, cls in STANDALONE.items():
        runs = [run_standalone(cls, args, seq) for seq in sequences]
        rows.append(summarize(name, "standalone", runs))

    print(f"\nworkload={args.workload} sim={args.sim} slots={args.slots} seeds={args.seeds} "
          f"processors={args.processors} memory={'on' if args.memory else 'off'}")
    print("metrics over ALL jobs that entered the system (unfinished jobs included)")
    if args.workload == "lublin":
        print("NOTE: the Lublin generator is not reproducible (bug 14), so the 'env' rows run on")
        print("      different jobs than the 'standalone' rows. Compare within each group only.")
    print()
    hdr = f"{'policy':20s} {'group':10s} {'bsld mean':>9s} {'median':>7s} {'p99':>8s} {'max wait':>8s} {'done':>6s} {'unfin.':>6s} {'util':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['policy']:20s} {r['group']:10s} {r['bsld_mean']:9.2f} {r['bsld_median']:7.2f} {r['bsld_p99']:8.2f} "
              f"{r['max_wait']:8.1f} {r['completed']:6.1f} {r['unfinished']:6.1f} {r['utilization']:5.2f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
