"""SMDP behaviour of the event-based environments.

Each transition of the event-based (SMDP) environment reports its sojourn time
in ``info["duration"]``: the number of time units between this decision and the
next one, and 0 when the chosen job starts at once.
"""

import random
import unittest

import gymnasium
import numpy as np

import schedgym.envs  # noqa: F401  (registers the environments)

ENV_IDS = ("CompactRM-v0", "DeepRM-v0")


def make_env(env_id, **kwargs):
    random.seed(0)
    np.random.seed(0)
    params = {"simulation_type": "event_based", "ignore_memory": True}
    if env_id == "DeepRM-v0":
        params["use_raw_state"] = True
    params.update(kwargs)
    return gymnasium.make(env_id, disable_env_checker=True, **params)


def wait_action(env):
    return env.action_space.n - 1


def first_fitting_job(env):
    s = env.unwrapped.scheduler
    for i, job in enumerate(s.queue_admission[: env.unwrapped.job_slots]):
        if s.can_schedule_now(job):
            return i
    return None


class TestDuration(unittest.TestCase):
    def test_duration_matches_scheduler_clock(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id)
                env.reset(seed=0)
                for _ in range(50):
                    before = env.unwrapped.scheduler.current_time
                    _, _, terminated, truncated, info = env.step(env.action_space.sample())
                    self.assertIn("duration", info)
                    self.assertEqual(info["duration"], env.unwrapped.scheduler.current_time - before)
                    self.assertGreaterEqual(info["duration"], 0)
                    if terminated or truncated:
                        break

    def test_immediate_scheduling_has_zero_duration(self):
        env = make_env("CompactRM-v0")
        env.reset(seed=0)
        for _ in range(100):
            i = first_fitting_job(env)
            if i is not None:
                _, _, _, _, info = env.step(i)
                self.assertEqual(info["duration"], 0)
                return
            env.step(wait_action(env))
        self.skipTest("no job ever fit the cluster")

    def test_waiting_takes_time(self):
        env = make_env("CompactRM-v0")
        env.reset(seed=0)
        _, _, _, _, info = env.step(wait_action(env))
        self.assertGreaterEqual(info["duration"], 1)



class TestStepRewards(unittest.TestCase):
    def test_one_raw_reward_per_elapsed_time_unit(self):
        for env_id in ENV_IDS:
            for sim in ("event_based", "time_based"):
                with self.subTest(env=env_id, sim=sim):
                    env = make_env(env_id, simulation_type=sim)
                    env.reset(seed=0)
                    for _ in range(200):
                        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
                        self.assertEqual(len(info["step_rewards"]), info["duration"])
                        self.assertAlmostEqual(reward, sum(info["step_rewards"]))
                        self.assertTrue(all(r <= 0 for r in info["step_rewards"]))
                        if terminated or truncated:
                            break

    def test_environment_does_not_discount(self):
        env = make_env("CompactRM-v0")
        self.assertFalse(hasattr(env.unwrapped, "gamma"))

    def test_every_elapsed_time_unit_is_penalised(self):
        # With a constant penalty of -1 per time unit, a wait of d units must
        # cost exactly -d: no time unit (in particular the first) is dropped.
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id)
                env.unwrapped.compute_reward = lambda jobs: -1.0
                env.reset(seed=0)
                checked = 0
                for _ in range(200):
                    _, reward, terminated, truncated, info = env.step(wait_action(env))
                    if terminated or truncated:
                        break
                    if info["duration"] > 1:
                        self.assertAlmostEqual(reward, -info["duration"])
                        checked += 1
                self.assertGreater(checked, 0, "no multi-unit wait happened")



class TestEndOfWorkload(unittest.TestCase):
    def exhaust_after(self, env, n_calls):
        """Make the workload raise StopIteration after n_calls more steps."""
        wl = env.unwrapped.simulator.simulator.workload
        original, calls = wl.step, {"n": 0}

        def step(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > n_calls:
                raise StopIteration
            return original(*args, **kwargs)

        wl.step = step

    def test_last_transition_keeps_its_time_units(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id)
                env.unwrapped.compute_reward = lambda jobs: -1.0
                env.reset(seed=0)
                self.exhaust_after(env, 3)
                for _ in range(50):
                    _, reward, terminated, truncated, info = env.step(wait_action(env))
                    self.assertEqual(len(info["step_rewards"]), info["duration"])
                    self.assertAlmostEqual(reward, -info["duration"])
                    if terminated or truncated:
                        break
                self.assertTrue(terminated or truncated)



class TestEpisodeEnd(unittest.TestCase):
    def test_time_limit_is_a_truncation(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id, time_limit=30)
                env.reset(seed=0)
                for _ in range(1000):
                    _, _, terminated, truncated, _ = env.step(wait_action(env))
                    if terminated or truncated:
                        break
                self.assertTrue(truncated)
                self.assertFalse(terminated)

    def test_workload_end_is_a_termination(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id)
                env.reset(seed=0)
                TestEndOfWorkload.exhaust_after(None, env, 3)
                for _ in range(50):
                    _, _, terminated, truncated, _ = env.step(wait_action(env))
                    if terminated or truncated:
                        break
                self.assertTrue(terminated)
                self.assertFalse(truncated)



class TestActionMask(unittest.TestCase):
    def test_mask_allows_exactly_the_jobs_that_fit_and_wait(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id)
                env.reset(seed=0)
                for _ in range(100):
                    u = env.unwrapped
                    mask = u.action_masks()
                    self.assertEqual(mask.shape, (env.action_space.n,))
                    self.assertTrue(mask[-1])
                    queue = u.scheduler.queue_admission[: u.job_slots]
                    for i in range(env.action_space.n - 1):
                        fits = i < len(queue) and bool(u.scheduler.can_schedule_now(queue[i]))
                        self.assertEqual(bool(mask[i]), fits)
                    valid = np.flatnonzero(mask)
                    _, _, terminated, truncated, _ = env.step(int(np.random.choice(valid)))
                    if terminated or truncated:
                        break

    def test_masked_job_starts_at_once(self):
        env = make_env("CompactRM-v0")
        env.reset(seed=0)
        for _ in range(100):
            mask = env.unwrapped.action_masks()
            if mask[:-1].any():
                _, _, _, _, info = env.step(int(np.flatnonzero(mask[:-1])[0]))
                self.assertEqual(info["duration"], 0)
                return
            env.step(wait_action(env))
        self.skipTest("no job ever fit the cluster")


class TestWaitMode(unittest.TestCase):
    def test_invalid_wait_mode(self):
        with self.assertRaises(ValueError):
            make_env("CompactRM-v0", wait_mode="forever")

    def test_next_event_wait_lasts_until_an_arrival_or_a_completion(self):
        for env_id in ENV_IDS:
            with self.subTest(env=env_id):
                env = make_env(env_id, wait_mode="next_event")
                env.reset(seed=0)
                s = env.unwrapped.scheduler
                checked = 0
                for _ in range(200):
                    s = env.unwrapped.scheduler
                    seen = len(s.queue_completed) + len(s.jobs_in_system)
                    done_jobs = len(s.queue_completed)
                    _, _, terminated, truncated, info = env.step(wait_action(env))
                    if terminated or truncated:
                        break
                    self.assertGreaterEqual(info["duration"], 1)
                    arrived = len(s.queue_completed) + len(s.jobs_in_system) > seen
                    finished = len(s.queue_completed) > done_jobs
                    self.assertTrue(arrived or finished)
                    checked += 1
                self.assertGreater(checked, 0)

    def test_unit_wait_is_one_time_unit_when_a_job_fits(self):
        env = make_env("CompactRM-v0", wait_mode="unit")
        env.reset(seed=0)
        for _ in range(100):
            if env.unwrapped.action_masks()[:-1].any():
                _, _, _, _, info = env.step(wait_action(env))
                self.assertEqual(info["duration"], 1)
                return
            env.step(wait_action(env))
        self.skipTest("no job ever fit the cluster")



class TestClockFeatures(unittest.TestCase):
    def run_until_queue(self, env):
        for _ in range(200):
            if env.unwrapped.scheduler.queue_admission:
                return
            env.step(wait_action(env))
        self.skipTest("the queue never filled")

    def test_default_observation_is_unchanged(self):
        a, b = make_env("CompactRM-v0"), make_env("CompactRM-v0", job_time_feature="submission",
                                                  include_time_left=True)
        oa, _ = a.reset(seed=0)
        ob, _ = b.reset(seed=0)
        np.testing.assert_array_equal(oa, ob)

    def test_age_feature_uses_waiting_time(self):
        env = make_env("CompactRM-v0", job_time_feature="age")
        env.reset(seed=0)
        self.run_until_queue(env)
        u = env.unwrapped
        job = u.scheduler.queue_admission[0]
        _, jobs, _ = u.scheduler.state(u.time_horizon, u.job_slots, u.smdp)
        age = u.scheduler.current_time - job.submission_time
        expected = (np.sign(age) * np.log(abs(age) + np.e) - 1 + 1) / (np.log(u.time_limit) + 1)
        self.assertAlmostEqual(float(u._normalize_jobs(jobs)[0][0]), expected, places=5)

    def test_time_left_can_be_turned_off(self):
        env = make_env("CompactRM-v0", include_time_left=False)
        env.reset(seed=0)
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(wait_action(env))
            self.assertEqual(obs[-1], 0.0)
            if terminated or truncated:
                break

    def test_invalid_job_time_feature(self):
        with self.assertRaises(ValueError):
            make_env("CompactRM-v0", job_time_feature="clock")


if __name__ == "__main__":
    unittest.main()
