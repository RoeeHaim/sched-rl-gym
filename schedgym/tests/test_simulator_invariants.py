"""Invariant and regression tests for the simulator core and the RL envs.

The invariant tests (TestInvariants) run whole episodes and check, after every
step, properties that must always hold for a correct scheduling simulator.
The regression tests (TestKnownIssues) pin down concrete bugs found on
2026-10-19; they are expected to FAIL until the bug is fixed.
"""

import random
import unittest
import warnings

import gymnasium
import numpy as np

import schedgym.envs  # noqa: F401  (registers the envs)
from schedgym.job import Job, JobStatus
from schedgym.scheduler import NullScheduler

warnings.filterwarnings("ignore")


def make_env(name="CompactRM-v0", sim="event_based", **kw):
    return gymnasium.make(name, simulation_type=sim, disable_env_checker=True, **kw)


def assert_invariants(tc: unittest.TestCase, s: NullScheduler, ctx: str):
    now = s.current_time
    cl = s.cluster
    jobs = s.all_jobs
    tc.assertEqual(len({id(j) for j in jobs}), len(jobs), f"job in 2 queues {ctx}")
    tc.assertLessEqual(cl.processors.used_resources, cl.processors.size, ctx)
    if not s.ignore_memory:
        tc.assertLessEqual(cl.memory.used_resources, cl.memory.size, ctx)
    running_cpu = sum(j.requested_processors for j in s.queue_running)
    tc.assertEqual(running_cpu, cl.processors.used_resources, ctx)
    tc.assertEqual(running_cpu, s.used_processors, ctx)
    owner = {}
    for j in s.queue_running:
        tc.assertEqual(j.status, JobStatus.RUNNING, ctx)
        tc.assertGreaterEqual(j.start_time, j.submission_time, ctx)
        for iv in j.resources.processors:
            for p in range(iv.begin, iv.end):
                tc.assertNotIn(p, owner, f"processor {p} shared {ctx}")
                owner[p] = j.id
    for j in s.queue_completed:
        tc.assertLessEqual(j.finish_time, now, ctx)
        tc.assertGreaterEqual(j.wait_time, 0, ctx)
        tc.assertGreaterEqual(j.slowdown, 1.0 - 1e-9, ctx)
    for j in s.queue_admission:
        tc.assertEqual(j.status, JobStatus.SUBMITTED, ctx)
    tc.assertEqual(
        sum(j.requested_time * j.requested_processors for j in s.queue_admission),
        s._queued_work_total,
        ctx,
    )
    tc.assertEqual(
        sum(j.requested_processors for j in s.jobs_in_system),
        s.requested_processors_in_system,
        ctx,
    )


class TestInvariants(unittest.TestCase):
    def _episode(self, name, sim, policy, seed, steps=300, **kw):
        random.seed(seed)
        np.random.seed(seed)
        env = make_env(name, sim, **kw)
        env.reset(seed=seed)
        u = env.unwrapped
        for i in range(steps):
            obs, r, term, trunc, _ = env.step(policy(u))
            self.assertTrue(np.isfinite(r))
            if isinstance(obs, np.ndarray):
                self.assertTrue(np.all(np.isfinite(obs)))
                self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertLessEqual(r, 0.0)  # reward is -sum(1/T_j)
            assert_invariants(self, u.scheduler, f"{name}/{sim}/seed={seed}/step={i}")
            if term or trunc:
                break

    def test_random_and_sjf_policies(self):
        policies = {
            "random": lambda u: u.action_space.sample(),
            "sjf": lambda u: u.scheduler.sjf_action(u.job_slots),
            "noop": lambda u: u.job_slots,
        }
        for name in ("CompactRM-v0", "DeepRM-v0"):
            for sim in ("time_based", "event_based"):
                for pname, pol in policies.items():
                    for seed in range(3):
                        with self.subTest(env=name, sim=sim, policy=pname, seed=seed):
                            self._episode(name, sim, pol, seed)

    def test_lublin_trace_ignore_memory(self):
        for seed in range(2):
            wl = {"type": "lublin", "length": 50, "nodes": 64, "random_seed": seed}
            self._episode(
                "CompactRM-v0", "event_based",
                lambda u: u.scheduler.sjf_action(u.job_slots), seed, steps=3000,
                processors=64, memory=64, ignore_memory=True, workload=wl,
                time_limit=None,
            )

    def test_scheduling_action_does_not_advance_time(self):
        random.seed(0)
        env = make_env("CompactRM-v0", "event_based")
        env.reset(seed=0)
        s = env.unwrapped.scheduler
        env.step(env.unwrapped.job_slots)  # no-op: advance to first decision epoch
        t = s.current_time
        self.assertTrue(s.queue_admission and s.cluster.fits(s.queue_admission[0]))
        _, r, *_ = env.step(0)
        self.assertEqual(s.current_time, t)
        self.assertEqual(r, 0)

    def test_reward_matches_definition_time_based(self):
        random.seed(1)
        env = make_env("CompactRM-v0", "time_based", time_limit=None if False else 500)
        env.reset(seed=1)
        u = env.unwrapped
        for _ in range(50):
            _, r, term, *_ = env.step(u.job_slots)  # no-op advances 1 time unit
            expected = -sum(1 / j.execution_time for j in u.scheduler.jobs_in_system)
            if r != 0:
                self.assertAlmostEqual(r, expected)
            if term:
                break


class TestHandcrafted(unittest.TestCase):
    """Tiny deterministic scenarios whose answer is computed by hand."""

    def _job(self, jid, procs, length, req=None):
        j = Job(jid, 0, execution_time=length, processors_allocated=procs,
                requested_processors=procs, requested_time=req or length)
        j.requested_memory = 0
        return j

    def test_two_jobs_run_sequentially_without_overlap(self):
        # Timing-convention agnostic: b must start only after a has finished.
        s = NullScheduler(10, 10, ignore_memory=True)
        a, b = self._job(1, 6, 3), self._job(2, 6, 2)
        s.submit([a, b])
        self.assertTrue(s.step(0))
        self.assertFalse(s.step(0))  # b cannot fit while a is scheduled
        for _ in range(20):
            if b.start_time < 0:
                s.step(0)
            s.forward_time()
        self.assertEqual(a.finish_time, 3)
        self.assertGreaterEqual(b.start_time, a.finish_time)
        self.assertEqual(b.finish_time, b.start_time + 2)
        self.assertEqual(len(s.queue_completed), 2)

    def test_job_bigger_than_cluster_rejected(self):
        s = NullScheduler(4, 4, ignore_memory=True)
        with self.assertRaises(RuntimeError):
            s.submit(self._job(1, 5, 1))


class TestKnownIssues(unittest.TestCase):
    """Each test here documents a bug. Expected to fail until fixed."""

    def test_observations_are_not_aliased(self):
        # Preallocated obs buffer is returned by reference: every stored
        # observation ends up equal to the last one (breaks trajectory storage).
        for name in ("CompactRM-v0", "DeepRM-v0"):
            with self.subTest(env=name):
                env = make_env(name)
                o0, _ = env.reset(seed=0)
                o1, *_ = env.step(env.action_space.sample())
                self.assertFalse(np.shares_memory(o0, o1))

    def test_compact_time_based_episode_does_not_end_on_empty_queue(self):
        # really_done() ends the episode whenever admission+waiting are empty,
        # even though the workload keeps producing jobs.
        random.seed(1)
        env = make_env("CompactRM-v0", "time_based", time_limit=200)
        env.reset(seed=1)
        u = env.unwrapped
        # Run until the episode ends (scheduling actions do not advance time,
        # so the number of actions is not the number of time steps).
        for _ in range(10_000):
            _, _, term, *_ = env.step(u.scheduler.sjf_action(u.job_slots))
            if term:
                break
        self.assertTrue(term)
        self.assertGreater(u.scheduler.current_time, 150)

    def test_reset_seed_makes_workload_reproducible(self):
        def traj(seed):
            env = make_env("CompactRM-v0")
            env.reset(seed=seed)
            return [float(env.step(0)[1]) for _ in range(20)]
        self.assertEqual(traj(0), traj(0))

    def test_underestimated_job_finish_time(self):
        # Job is killed at requested_time (resources freed), but finish_time
        # is recorded as start + execution_time.
        s = NullScheduler(10, 10, ignore_memory=True)
        j = Job(1, 0, execution_time=10, processors_allocated=2,
                requested_processors=2, requested_time=4)
        j.requested_memory = 0
        s.submit(j)
        s.step(0)
        for _ in range(6):
            s.forward_time()
        self.assertEqual(j.status, JobStatus.COMPLETED)
        self.assertEqual(j.finish_time, 4)

    def test_event_based_wrapper_time_tracks_scheduler_time(self):
        # EventBasedDeepRmSimulator.current_time is +1 per decision, not per
        # time unit, so DeepRM's "time since last job" feature is wrong.
        random.seed(0)
        env = make_env("DeepRM-v0", "event_based")
        env.reset(seed=0)
        for _ in range(60):
            env.step(env.action_space.sample())
        u = env.unwrapped
        self.assertEqual(u.simulator.current_time, u.scheduler.current_time)

    def _job(self, jid, procs, length):
        j = Job(jid, 0, execution_time=length, processors_allocated=procs,
                requested_processors=procs, requested_time=length)
        j.requested_memory = 0
        return j

    def test_back_to_back_scheduling_no_idle_gap(self):
        # Events at time t are only applied to self.cluster when moving t->t+1,
        # and NullScheduler.schedule() checks the stale cluster.fits() before
        # the (correct) can_schedule_now(). Result: freed resources idle 1 tick.
        s = NullScheduler(10, 10, ignore_memory=True)
        a, b = self._job(1, 6, 3), self._job(2, 6, 2)
        s.submit([a, b])
        s.step(0)
        for _ in range(10):
            if b.start_time < 0:
                s.step(0)
            if b.start_time >= 0:
                break
            s.forward_time()
        self.assertEqual(b.start_time, a.finish_time)  # expected 3, actual 4

    def test_job_completes_at_its_finish_time(self):
        s = NullScheduler(4, 4, ignore_memory=True)
        j = self._job(1, 4, 1)
        s.submit(j)
        s.step(0)
        s.forward_time()  # now t=1 == finish time
        self.assertEqual(j.status, JobStatus.COMPLETED)
        self.assertEqual(s.cluster.processors.used_resources, 0)


if __name__ == "__main__":
    unittest.main()
