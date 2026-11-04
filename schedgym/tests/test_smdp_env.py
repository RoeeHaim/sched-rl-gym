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


if __name__ == "__main__":
    unittest.main()
