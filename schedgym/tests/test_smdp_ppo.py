"""Tests for the SMDP discounting used in training (scripts/smdp_ppo.py).

Skipped when stable-baselines3 / sb3-contrib are not installed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import torch as th
    from gymnasium import spaces

    from smdp_ppo import (
        AverageReward,
        Discounted,
        Paper,
        SmdpMaskablePPO,
        SmdpMaskableRolloutBuffer,
        VecSmdpReward,
        intra_option_reward,
        make_criterion,
    )
except ImportError:  # pragma: no cover
    SmdpMaskableRolloutBuffer = None


@unittest.skipIf(SmdpMaskableRolloutBuffer is None, "sb3-contrib not installed")
class TestSmdpDiscounting(unittest.TestCase):
    def test_intra_option_reward(self):
        g = 0.9
        self.assertEqual(intra_option_reward([], g), 0.0)
        self.assertAlmostEqual(intra_option_reward([-1.0, -2.0, -3.0], g), -1 - 2 * g - 3 * g * g)

    def make_buffer(self, rewards, values, taus, starts, gamma, lam):
        n = len(rewards)
        buf = SmdpMaskableRolloutBuffer(
            n, spaces.Box(-1, 1, (1,)), spaces.Discrete(2), "cpu", gae_lambda=lam, gamma=gamma, n_envs=1
        )
        for i in range(n):
            buf.pending_taus = np.array([taus[i]], dtype=np.float32)
            buf.add(
                np.zeros((1, 1)), np.zeros((1, 1)), np.array([rewards[i]]), np.array([starts[i]]),
                th.tensor([[values[i]]]), th.zeros(1),
            )
        return buf

    def test_gae_uses_gamma_to_the_tau(self):
        g, lam = 0.9, 0.8
        R = [-1.0, 0.0, -2.0]
        V = [-5.0, -4.0, -3.0]
        tau = [2, 0, 3]
        last_v = -1.0
        buf = self.make_buffer(R, V, tau, [1, 0, 0], g, lam)
        buf.compute_returns_and_advantage(th.tensor([[last_v]]), np.array([False]))

        # Reference computed by hand from the formulas.
        nxt = V[1:] + [last_v]
        adv = [0.0] * 3
        a = 0.0
        for i in reversed(range(3)):
            delta = R[i] + g ** tau[i] * nxt[i] - V[i]
            a = delta + (g * lam) ** tau[i] * a
            adv[i] = a
        np.testing.assert_allclose(buf.advantages[:, 0], adv, rtol=1e-5)
        np.testing.assert_allclose(buf.returns[:, 0], np.array(adv) + V, rtol=1e-5)

    def test_zero_tau_is_not_discounted(self):
        # Two decisions at the same instant: V(s_1) enters with weight 1.
        buf = self.make_buffer([0.0], [0.0], [0], [1], 0.5, 1.0)
        buf.compute_returns_and_advantage(th.tensor([[-7.0]]), np.array([False]))
        self.assertAlmostEqual(float(buf.returns[0, 0]), -7.0)

    def test_episode_end_cuts_bootstrap(self):
        buf = self.make_buffer([-1.0, -1.0], [0.0, 0.0], [1, 1], [1, 1], 0.9, 1.0)
        buf.compute_returns_and_advantage(th.tensor([[-100.0]]), np.array([True]))
        # episode_starts[1] == 1 -> step 0 does not bootstrap from step 1
        self.assertAlmostEqual(float(buf.returns[0, 0]), -1.0)
        self.assertAlmostEqual(float(buf.returns[1, 0]), -1.0)

    def test_missing_tau_raises(self):
        buf = SmdpMaskableRolloutBuffer(1, spaces.Box(-1, 1, (1,)), spaces.Discrete(2), "cpu", n_envs=1)
        with self.assertRaises(RuntimeError):
            buf.add(np.zeros((1, 1)), np.zeros((1, 1)), np.array([0.0]), np.array([1]),
                    th.zeros((1, 1)), th.zeros(1))


@unittest.skipIf(SmdpMaskableRolloutBuffer is None, "sb3-contrib not installed")
class TestCriteria(unittest.TestCase):
    """One environment, three ways to turn (step_rewards, duration) into returns."""

    make_buffer = TestSmdpDiscounting.make_buffer

    def gae(self, criterion, R, V, tau, starts, last_v, done, lam=0.8, gamma=0.9):
        buf = self.make_buffer(R, V, tau, starts, gamma, lam)
        buf.criterion = criterion
        buf.compute_returns_and_advantage(th.tensor([[last_v]]), np.array([done]))
        return buf

    def test_explicit_discounted_matches_default(self):
        R, V, tau, starts = [-1.0, 0.0, -2.0], [-5.0, -4.0, -3.0], [2, 0, 3], [1, 0, 0]
        default = self.gae(None, R, V, tau, starts, -1.0, False)
        explicit = self.gae(Discounted(0.9), R, V, tau, starts, -1.0, False)
        np.testing.assert_allclose(default.advantages, explicit.advantages)

    def test_intra_rewards(self):
        r = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(Discounted(0.9).intra(r), -1 - 2 * 0.9 - 3 * 0.81)
        self.assertAlmostEqual(Paper(0.9).intra(r), -1 - 2 * 0.9 - 3 * 0.81)
        self.assertAlmostEqual(AverageReward().intra(r), -6.0)
        self.assertEqual(AverageReward().intra([]), 0.0)

    def test_paper_does_not_discount_between_decisions(self):
        lam = 0.8
        R, V, tau = [-1.0, -2.0], [-5.0, -4.0], [3, 4]
        buf = self.gae(Paper(0.9), R, V, tau, [1, 0], -1.0, False, lam=lam)
        nxt = [V[1], -1.0]
        a, adv = 0.0, [0.0, 0.0]
        for i in reversed(range(2)):
            a = (R[i] + nxt[i] - V[i]) + lam * a
            adv[i] = a
        np.testing.assert_allclose(buf.advantages[:, 0], adv, rtol=1e-5)

    def test_average_subtracts_rho_times_duration(self):
        lam = 0.8
        crit = AverageReward(rho_lr=1.0)  # rho = sum(R) / sum(duration) of this rollout
        R, V, tau = [-6.0, -3.0], [1.0, 2.0], [2, 4]
        buf = self.gae(crit, R, V, tau, [1, 0], 0.5, False, lam=lam)
        rho = sum(R) / sum(tau)
        self.assertAlmostEqual(crit.rho, rho)
        nxt = [V[1], 0.5]
        a, adv = 0.0, [0.0, 0.0]
        for i in reversed(range(2)):
            a = (R[i] - rho * tau[i] + nxt[i] - V[i]) + lam * a
            adv[i] = a
        np.testing.assert_allclose(buf.advantages[:, 0], adv, rtol=1e-5)

    def test_average_does_not_cut_at_episode_boundary(self):
        crit = AverageReward(rho_lr=0.0)  # keep rho = 0 to isolate the bootstrap
        buf = self.gae(crit, [-1.0, -1.0], [0.0, 0.0], [1, 1], [1, 1], -100.0, True, lam=1.0)
        # continuing task: the reset does not cut the chain, so with lambda = 1
        # step 0 sees the whole tail: -1 - 1 - 100 (a discounted criterion gives -1)
        self.assertAlmostEqual(float(buf.returns[1, 0]), -101.0)
        self.assertAlmostEqual(float(buf.returns[0, 0]), -102.0)

    def test_average_rho_converges_for_a_fixed_reward_rate(self):
        crit = AverageReward(rho_lr=0.5)
        for _ in range(40):
            self.gae(crit, [-0.6, -1.5, 0.0], [0.0, 0.0, 0.0], [2, 5, 0], [1, 0, 0], 0.0, False)
        self.assertAlmostEqual(crit.rho, -0.3, places=5)

    def test_average_ignores_rollouts_without_elapsed_time(self):
        crit = AverageReward(rho_lr=1.0, rho_init=-0.25)
        self.gae(crit, [0.0, 0.0], [0.0, 0.0], [0, 0], [1, 0], 0.0, False)
        self.assertAlmostEqual(crit.rho, -0.25)

    def test_average_td_rho_update(self):
        crit = AverageReward(rho_lr=0.5, rho_method="td")
        R, V, tau = [-4.0], [1.0], [2]
        self.gae(crit, R, V, tau, [1], 3.0, False)
        # delta = R - 0 * tau + V' - V = -4 + 3 - 1 = -2; rho += 0.5 * (-2) / 2
        self.assertAlmostEqual(crit.rho, -0.5)

    def test_make_criterion(self):
        self.assertIsInstance(make_criterion("discounted", 0.9), Discounted)
        self.assertIsInstance(make_criterion("paper", 0.9), Paper)
        self.assertIsInstance(make_criterion("average", 0.9), AverageReward)
        with self.assertRaises(ValueError):
            make_criterion("nope", 0.9)
        with self.assertRaises(ValueError):
            AverageReward(rho_method="nope")



@unittest.skipIf(SmdpMaskableRolloutBuffer is None, "sb3-contrib not installed")
class TestTruncation(unittest.TestCase):
    """A truncated episode bootstraps with discount(duration) * V(terminal), not gamma * V."""

    def run_one_step(self, criterion, duration):
        import gymnasium
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv

        class OneStep(gymnasium.Env):
            observation_space = spaces.Box(-1, 1, (1,), np.float32)
            action_space = spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, np.float32), {}

            def step(self, action):
                info = {"duration": duration, "step_rewards": [0.0] * duration}
                return np.ones(1, np.float32), 0.0, False, True, info

        env = DummyVecEnv([lambda: ActionMasker(OneStep(), lambda e: np.ones(2, bool))])
        env = VecSmdpReward(env, criterion)
        model = SmdpMaskablePPO("MlpPolicy", env, criterion=criterion, n_steps=1, batch_size=1,
                                n_epochs=1, gamma=0.5)
        # freeze the value of every state at 10 so the bootstrap is predictable
        for p in model.policy.value_net.parameters():
            th.nn.init.zeros_(p)
        th.nn.init.constant_(model.policy.value_net.bias, 10.0)
        _, callback = model._setup_learn(1, None)
        callback.on_training_start(locals(), globals())
        model.collect_rollouts(model.env, callback, model.rollout_buffer, 1)
        return float(model.rollout_buffer.rewards[0, 0])

    def test_discounted_bootstraps_with_gamma_to_the_duration(self):
        # gamma = 0.5, duration = 3: 0.5 ** 3 * 10 = 1.25 (SB3 alone would add 0.5 * 10)
        self.assertAlmostEqual(self.run_one_step(Discounted(0.5), 3), 1.25, places=4)

    def test_paper_bootstraps_without_discount(self):
        self.assertAlmostEqual(self.run_one_step(Paper(0.5), 3), 10.0, places=4)

    def test_average_adds_no_bootstrap(self):
        self.assertAlmostEqual(self.run_one_step(AverageReward(), 3), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
