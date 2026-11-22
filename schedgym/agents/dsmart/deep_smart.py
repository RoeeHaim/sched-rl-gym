"""DeepSMART (DSMART): average-reward PPO for semi-Markov decision processes.

Ported from ``erelon/SMDP_Agents`` (``smdp_agents/deep_smart.py``, commit
84bd5db); the ``DeepSMART`` class body is the reference's verbatim except that
the entropy probe of the pressure boost uses the stored action masks. It
implements Algorithm 1 and Appendix D (Algorithm 2) of "It takes time:
Average-Reward SMDP Reinforcement Learning for Real-World Robots":

1. **No discounting** (``gamma = 1``, Eq. 18). The TD residual carries
   ``r - rho*tau`` (Eq. 6-7).
2. **A-centering** (Eq. 23): the batch advantage is centred on its own mean.
3. **Calibrated entropy pressure** (Sec. 3.3, Eq. 25, Algorithm 2):
   ``c_H = p * max(EMA std(A), warm-up median std(A))``, times a bounded
   log-boost while the advantage scale starves and entropy is not rising.
4. **Critic value-bias centering** (Eq. 9-10) lives in the PPO core:
   ``target = return - v * b`` with ``b`` an EMA of ``V(s_n)``
   (``rm_vbias_coeff`` is ``v``).

The rho estimator is a mixin (``schedgym.agents.dsmart.rates``)::

    name         class                    rho
    DS           SmartDeepSMART           sum(r) / sum(tau)               Eq. 13
    DS_relaxed   RsmartDeepSMART          EMA(r) / EMA(tau)               Eq. 14
    DS_smoothed  SmoothedSmartDeepSMART   time-decayed EMA of r / tau     Eq. 15
    APO          APO                      EMA(r) / EMA(1): per DECISION   control

All agents here are categorical (sched-rl-gym's action space is discrete).
"""
import math

import torch

from .ppo import PPO
from .rates import SMART, RelaxedSMART, SmoothedSMART


class DeepSMART(PPO):
    """Average-reward PPO with A-centering and calibrated entropy pressure.

    Subclass it together with a rate estimator; see the module docstring.
    """

    longrun = True

    def __init__(self, obs_dim, act_dim, a_centering=True,
                 calibrated_pressure=True, entropy_pressure=0.02,
                 entropy_warmup_iters=20, pressure_track=True,
                 pressure_track_steps=2_000_000, pressure_boost=True,
                 boost_efold_steps=400_000, boost_frac=0.30, boost_max=8.0,
                 **kwargs):
        """
        a_centering, calibrated_pressure
            The two changes DeepSMART makes beyond undiscounted PPO, both on by
            default. Turn one off to measure what it contributes; with both off
            this is average-reward PPO with a fixed entropy coefficient.

            Note that ``entropy_pressure=0`` is NOT how to disable the bonus: the
            mechanism would still run and overwrite ``entropy_loss_coeff`` with 0
            every iteration, silently discarding a fixed coefficient you had set.
            Use ``calibrated_pressure=False``, which leaves the coefficient alone.
        entropy_pressure
            The target ratio ``p = c_H / std(A)``. This is the hyperparameter to
            search; it is environment-specific and we make no claim of a
            universal value.
        entropy_warmup_iters
            Iterations used to measure the advantage scale before the bonus is
            switched on. ``c_H`` is whatever ``entropy_loss_coeff`` was (0 by
            default) until the measurement completes.
        pressure_track, pressure_track_steps
            Re-derive ``c_H`` from a slow estimate of the advantage scale, so the
            delivered ratio stays at ``p`` as the scale moves. The time constant
            is in ENVIRONMENT STEPS, not iterations: how many iterations fall in
            a given span of simulated time depends on the dwell regime, so an
            iteration-based rate would mean different things in different
            environments.
        pressure_boost, boost_efold_steps, boost_frac, boost_max
            Raise ``c_H`` above the tracked value while the advantage scale sits
            below ``boost_frac`` of its own running peak, which is the signal
            that exploration is starving. The trigger is RELATIVE to the run's
            own history because absolute set-points did not transfer across
            environments. The boost is suppressed while policy entropy is already
            rising, floored at 1 (it only ever adds) and capped at ``boost_max``.
        """
        if calibrated_pressure and float(entropy_pressure) <= 0.0:
            raise ValueError(
                "entropy_pressure must be > 0 when calibrated_pressure is on; "
                "pass calibrated_pressure=False to use a fixed "
                "entropy_loss_coeff instead.")
        kwargs.setdefault("entropy_loss_coeff", 0.0)
        super().__init__(obs_dim, act_dim, **kwargs)
        self.a_centering = bool(a_centering)
        self.calibrated_pressure = bool(calibrated_pressure)
        self.entropy_pressure = float(entropy_pressure)
        self.entropy_warmup_iters = int(entropy_warmup_iters)
        self.pressure_track = bool(pressure_track)
        self.pressure_track_steps = float(pressure_track_steps)
        self.pressure_boost = bool(pressure_boost)
        self.boost_efold_steps = float(boost_efold_steps)
        self.boost_frac = float(boost_frac)
        self.boost_max = float(boost_max)

        self.adv_std = float("nan")          # advantage scale of the last batch
        self.realised_pressure = float("nan")  # c_H / std(A), the delivered ratio
        self.boost = 1.0
        self._warmup = []
        self._adv_std_frozen = None
        self._scale_ema = None
        self._scale_peak = 0.0
        self._log_boost = 0.0
        self._entropy_ema = None
        self._entropy_slope = 0.0

    # --- holding time -------------------------------------------------------
    def dwell(self, time):
        """The holding time the algorithm attributes to each decision.

        Overridden by :class:`APO`, which ignores duration entirely.
        """
        return time

    def rate_residual(self, reward, time):
        return reward - self.rho * self.dwell(time)

    def update_rho(self, reward, value, time):
        return super().update_rho(reward, value, self.dwell(time))

    # --- advantage conditioning --------------------------------------------
    def shape_advantage(self, adv, valid, batch):
        """Measure the advantage scale, set ``c_H`` from it, then A-centre."""
        m = valid > 0
        a = adv[m]
        if a.numel() < 2:
            return adv
        self.adv_std = float(a.std().item())
        if self.calibrated_pressure:
            self._update_entropy_coeff(batch)
        if not self.a_centering:
            return super().shape_advantage(adv, valid, batch)
        return adv - a.mean()

    # --- calibrated entropy pressure ---------------------------------------
    def _update_entropy_coeff(self, batch):
        """Set ``c_H`` so that ``c_H / std(A)`` tracks the configured ratio ``p``.

        The entropy bonus and the policy-gradient term act on the same quantity —
        the width of the policy — but in different units. The objective's force
        on ``log sigma`` scales with the curvature of the advantage landscape,
        which carries the task's reward units and moves during training; the
        bonus contributes a constant ``c_H``. Under a shared gradient-norm clip
        only the RATIO of the two survives into the update direction, so the
        ratio is what we specify. A fixed coefficient is a fixed force against a
        moving one: it means something different on every task, and something
        different at the end of a run than at its start.
        """
        scale = self.adv_std
        if not scale == scale or scale <= 0.0:       # NaN or degenerate
            return

        if self._adv_std_frozen is None:
            self._warmup.append(scale)
            if len(self._warmup) >= self.entropy_warmup_iters:
                w = sorted(self._warmup)
                n = len(w)
                self._adv_std_frozen = (w[n // 2] if n % 2
                                        else 0.5 * (w[n // 2 - 1] + w[n // 2]))
                self._scale_ema = self._adv_std_frozen
                self.entropy_loss_coeff = self.entropy_pressure * self._adv_std_frozen
            return

        steps = self._batch_steps(batch)
        if self.pressure_track:
            g = self._rate(steps, self.pressure_track_steps)
            self._scale_ema = (1.0 - g) * self._scale_ema + g * scale
        floor = max(self._scale_ema, self._adv_std_frozen)
        self.entropy_loss_coeff = self.entropy_pressure * floor
        if self.pressure_boost:
            self.entropy_loss_coeff *= self._boost_step(batch, steps)
        self.realised_pressure = self.entropy_loss_coeff / scale

    def _boost_step(self, batch, steps):
        """Multiplicative boost over the tracked floor while exploration starves."""
        self._scale_peak = max(self._scale_peak, self._scale_ema)
        starving = (self._scale_peak > 0.0
                    and self._scale_ema < self.boost_frac * self._scale_peak)

        # Is the width already recovering? Measured on the behaviour policy's
        # entropy and smoothed twice (level, then slope) so one noisy batch
        # cannot flip the decision.
        mask = batch.get("mask")
        entropy = self.policy_entropy(
            batch["obs"].reshape(-1, batch["obs"].shape[-1]),
            None if mask is None else mask.reshape(-1, mask.shape[-1]))
        g = self._rate(steps, self.pressure_track_steps)
        if self._entropy_ema is None:
            self._entropy_ema, slope = entropy, 0.0
        else:
            slope = entropy - self._entropy_ema
            self._entropy_ema = (1.0 - g) * self._entropy_ema + g * entropy
        self._entropy_slope = (1.0 - g) * self._entropy_slope + g * slope

        beta = self._rate(steps, self.boost_efold_steps)
        step = beta if (starving and self._entropy_slope <= 0.0) else -beta
        self._log_boost = min(max(self._log_boost + step, 0.0),
                              math.log(self.boost_max))
        self.boost = math.exp(self._log_boost)
        return self.boost

    @staticmethod
    def _rate(steps, horizon):
        """Per-iteration rate for a time constant given in environment steps."""
        if horizon <= 0.0:
            return 0.0
        return min(1.0, float(steps) / float(horizon))

    def _batch_steps(self, batch):
        """Environment steps this iteration consumed: the summed holding time."""
        time = batch.get("time")
        if time is None:
            return float(batch["rew"].numel())
        return float(self.dwell(time).sum().item())

    def update(self, buffer, bootstrap_value):
        stats = super().update(buffer, bootstrap_value)
        stats.update(adv_std=self.adv_std, entropy_coeff=self.entropy_loss_coeff,
                     realised_pressure=self.realised_pressure, boost=self.boost)
        return stats


# --------------------------------------------------------------- variants ---
class APO(DeepSMART, RelaxedSMART):
    """Average-reward PPO that ignores duration: every decision counts as one.

    The MDP reading of average reward. On an SMDP it optimises reward per
    DECISION rather than reward per unit time, so it is the control that isolates
    what time-awareness buys.
    """

    rho_reduce = "mean"

    def dwell(self, time):
        return torch.ones_like(time)


class RsmartDeepSMART(DeepSMART, RelaxedSMART):
    """``rho = EWMA(reward) / EWMA(tau)`` — a ratio of two smoothed quantities."""

    rho_reduce = "mean"


class SmartDeepSMART(DeepSMART, SMART):
    """``rho = sum(reward) / sum(tau)`` over the whole run; it cannot forget."""

    rho_reduce = "sum"


class SmoothedSmartDeepSMART(DeepSMART, SmoothedSMART):
    """``rho`` smoothed in ELAPSED TIME: it forgets per unit of simulated time
    rather than per transition, so its memory does not change when the dwell does.

    ``rho_reduce="none"`` because the estimator decays by ``exp(-lambda*tau)``
    per transition; aggregating the batch would only be exact if the rate were
    constant across it, which a rollout does not guarantee.
    """

    rho_reduce = "none"


ALGOS = {
    "DS": SmartDeepSMART,
    "DS_relaxed": RsmartDeepSMART,
    "DS_smoothed": SmoothedSmartDeepSMART,
    "APO": APO,
    "PPO": PPO,
}


def make_agent(algo, obs_dim, n_actions, **kwargs):
    """Build an agent by its paper name (``DS``, ``DS_relaxed``, ``DS_smoothed``,
    ``APO``) or ``PPO`` for the discounted baseline."""
    try:
        cls = ALGOS[algo]
    except KeyError:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(ALGOS)}") from None
    if cls is PPO:
        for k in ("a_centering", "calibrated_pressure", "entropy_pressure",
                  "entropy_warmup_iters", "pressure_track", "pressure_track_steps",
                  "pressure_boost", "boost_efold_steps", "boost_frac", "boost_max"):
            kwargs.pop(k, None)
    return cls(obs_dim, n_actions, **kwargs)
