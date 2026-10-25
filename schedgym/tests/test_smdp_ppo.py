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

    from smdp_ppo import SmdpMaskableRolloutBuffer, intra_option_reward
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


if __name__ == "__main__":
    unittest.main()
