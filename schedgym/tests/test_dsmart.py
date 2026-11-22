"""DSMART port: rate estimators, masking, zero-duration decisions, A-centering."""

import math
import random
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from schedgym.agents.dsmart import ALGOS, RolloutBuffer, make_agent  # noqa: E402
from schedgym.agents.dsmart.rates import (  # noqa: E402
    SMART,
    RelaxedSMART,
    SmoothedSMART,
)
from schedgym.agents.dsmart.train import evaluate, resolve_config, run  # noqa: E402


def fill(agent, T=32, B=4, n_actions=6, obs_dim=5, tau0_frac=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer()
    for _ in range(T):
        obs = torch.randn(B, obs_dim, generator=g)
        mask = torch.rand(B, n_actions, generator=g) > 0.5
        mask[:, -1] = True
        a, v, logp = agent.act(obs, mask)
        assert mask[torch.arange(B), a].all(), "sampled a masked action"
        tau = torch.randint(1, 5, (B,), generator=g).float()
        tau[torch.rand(B, generator=g) < tau0_frac] = 0.0
        r = -tau * 2.0  # true rate -2 per time unit; r = 0 when tau = 0
        buf.add(obs, a, r, torch.zeros(B), torch.zeros(B), v, logp, time=tau, mask=mask)
    return buf


class TestRates(unittest.TestCase):
    def test_estimators_recover_constant_rate(self):
        for cls in (SMART, RelaxedSMART, SmoothedSMART):
            est = cls(rho_learning_rate=0.3)
            for tau in (1, 3, 2, 5, 1):
                est.calc_new_rho(-2.0 * tau, tau, None, None)
            self.assertAlmostEqual(est.rho, -2.0, places=9, msg=cls.__name__)


class TestAgents(unittest.TestCase):
    def test_every_algo_updates_with_zero_tau(self):
        for algo in ALGOS:
            agent = make_agent(algo, 5, 6, seed=0, entropy_warmup_iters=2,
                               rm_vbias_coeff=0.3, minibatches=4, epochs=2)
            for it in range(4):
                stats = agent.update(fill(agent, seed=it), torch.zeros(4))
            for k, v in stats.items():
                self.assertTrue(math.isfinite(v), f"{algo}: {k}={v}")
            if algo in ("DS", "DS_relaxed", "DS_smoothed"):
                self.assertAlmostEqual(agent.rho, -2.0, places=5, msg=algo)
                self.assertAlmostEqual(stats["adv_mean"], 0.0, places=5)
                self.assertGreater(agent.entropy_loss_coeff, 0.0)
            if algo == "APO":
                # per-DECISION rate: -2 * E[tau] = -2 * 0.5 * 2.5 = -2.5
                self.assertAlmostEqual(agent.rho, -2.5, delta=0.3)

    def test_eval_act_respects_mask(self):
        agent = make_agent("DS", 5, 6, seed=0)
        obs = torch.randn(64, 5)
        mask = torch.zeros(64, 6, dtype=torch.bool)
        mask[:, 2] = True
        self.assertTrue((agent.eval_act(obs, mask) == 2).all())

    def test_batch_without_elapsed_time_keeps_rho(self):
        agent = make_agent("DS_smoothed", 5, 6, seed=0)
        agent.update(fill(agent, tau0_frac=0.0), torch.zeros(4))
        rho = agent.rho
        agent.update(fill(agent, tau0_frac=1.0), torch.zeros(4))
        self.assertEqual(agent.rho, rho)


class TestContinuingEnv(unittest.TestCase):
    def test_time_limit_none_never_truncates(self):
        from schedgym.agents.dsmart.train import make_env

        cfg = resolve_config({})
        random.seed(0)
        np.random.seed(0)
        env = make_env(cfg)
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(600):
            a = rng.choice(np.flatnonzero(env.action_masks()))
            _, _, term, trunc, _ = env.step(a)
            self.assertFalse(trunc)
            self.assertFalse(term)
        self.assertGreater(env.scheduler.current_time, 200)

    def test_nested_env_override_keeps_workload_defaults(self):
        cfg = resolve_config({"env_kwargs": {"workload": {"new_job_rate": 0.2}}})
        wl = cfg["env_kwargs"]["workload"]
        self.assertEqual(wl["new_job_rate"], 0.2)
        self.assertEqual(wl["type"], "deeprm")
        self.assertEqual(cfg["env_kwargs"]["job_time_feature"], "age")

    def test_swf_workload(self):
        import tempfile

        lines = ["; MaxProcs: 64"]
        for i in range(1, 41):
            # id submit wait run procs cpu mem reqprocs reqtime reqmem status ...
            lines.append(f"{i} {i * 120} 0 {300 + 60 * (i % 7)} {2 ** (i % 7)} "
                         f"-1 -1 {2 ** (i % 7)} 900 -1 1 1 1 1 1 1 -1 -1")
        with tempfile.NamedTemporaryFile("w", suffix=".swf", delete=False) as f:
            f.write("\n".join(lines) + "\n")
        wl = {"type": "swf", "tracefile": f.name, "cluster_processors": 10,
              "time_scale": 60}
        cfg = resolve_config({"env_kwargs": {"workload": wl, "ignore_memory": True}})
        self.assertNotIn("new_job_rate", cfg["env_kwargs"]["workload"])
        from schedgym.agents.dsmart.train import make_env

        env = make_env(cfg)
        env.reset(seed=0)
        sizes = [j.requested_processors for j in env.simulator.workload.trace]
        self.assertLessEqual(max(sizes), 10)
        self.assertEqual(env.simulator.workload.trace[0].submission_time, 0)
        total = 0
        for _ in range(300):
            mask = env.action_masks()
            fit = np.flatnonzero(mask[:-1])
            _, r, term, trunc, info = env.step(int(fit[0]) if len(fit) else len(mask) - 1)
            total += info["duration"]
        self.assertGreater(total, 0)
        self.assertGreater(len(env.scheduler.queue_completed), 0)

    def test_reward_scale_and_eval_overrides(self):
        final = run({"algo": "DS", "time_budget": 300, "n_envs": 2, "batch_T": 32,
                     "eval_time": 100, "eval_warmup": 20, "eval_seeds": [1],
                     "reward_scale": 0.1, "entropy_warmup_iters": 2,
                     "eval_workload_overrides": [{"new_job_rate": 0.1},
                                                 {"new_job_rate": 0.15}],
                     "verbose": False})
        self.assertTrue(math.isfinite(final["eval_sample_reward_rate"]))

    def test_run_smoke(self):
        final = run({"algo": "DS", "time_budget": 300, "n_envs": 2, "batch_T": 32,
                     "eval_time": 100, "eval_warmup": 20, "eval_seeds": [1],
                     "entropy_warmup_iters": 2, "verbose": False})
        self.assertTrue(math.isfinite(final["eval_reward_rate"]))
        self.assertLessEqual(final["eval_reward_rate"], 0.0)

    def test_heuristic_eval_is_deterministic(self):
        cfg = resolve_config({"algo": "SJF", "eval_time": 200, "eval_warmup": 20,
                              "eval_seeds": [3]})
        rng = np.random.default_rng(0)
        from schedgym.agents.dsmart.train import heuristic_action

        pol = lambda env, obs: heuristic_action(env, "SJF", rng)  # noqa: E731
        self.assertEqual(evaluate(cfg, pol)["reward_rate"],
                         evaluate(cfg, pol)["reward_rate"])
