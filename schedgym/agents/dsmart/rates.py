"""Reward-rate estimators behind the three DSMART rho variants.

Vendored verbatim from ``erelon/SMDP_Agents`` (``smdp_agents/average_rates.py``,
commit 84bd5db), keeping only the estimators the deep agents use:

    CumulativeTimeRate                    DS          rho = sum(r) / sum(tau)      (Eq. 13)
    ExponentialMovingRatioRate            DS_relaxed  rho = EMA(r) / EMA(tau)       (Eq. 14)
    NormalizedExponentialMovingTimeRate   DS_smoothed rho smoothed in elapsed time  (Eq. 15)

The tabular agents of the reference (``SMART``, ``RelaxedSMART``,
``SmoothedSMART``) are replaced by the thin mixins at the bottom of this file:
the deep agents only ever call ``calc_new_rho``, so the Q-table machinery is not
needed.
"""

import math


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_nonnegative_duration(duration: float) -> float:
    duration = _require_finite("duration", duration)
    if duration < 0:
        raise ValueError("duration must not be negative")
    return duration


def _require_duration(duration: float) -> float:
    duration = _require_nonnegative_duration(duration)
    if duration == 0:
        raise ValueError("duration must be greater than zero")
    return duration


def _require_beta(beta: float) -> float:
    beta = _require_finite("beta", beta)
    if not 0 < beta <= 1:
        raise ValueError("beta must be in the interval (0, 1]")
    return beta


def _time_decay(beta: float) -> float:
    """Decay rate whose retention over a unit duration is ``1 - beta``.

    ``log1p`` because ``1 - beta`` cancels for small beta; ``beta == 1`` means
    no memory.
    """
    beta = _require_beta(beta)
    return math.inf if beta == 1 else -math.log1p(-beta)


# Building blocks:  EMA, normalized EMA, and time-decayed EMA

class ExponentialMovingAverage:
    """Zero-initialized EMA with a call-specific multiplicative weight."""

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        self.value = (1 - self.beta) * self.value + self.beta * value * weight
        return self.value


class CumulativeAverage:
    """Running mean of ``value * weight``, with the EMA's call signature.

    The whole-history counterpart of :class:`ExponentialMovingAverage`: a gain of
    ``1 / n`` instead of a fixed ``beta``, so an estimator written against this
    interface can be built over either and forget or not accordingly.  There is no
    zero-initialization bias to remove, so no normalized variant is needed.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.count = 0

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        self.count += 1
        # Incremental rather than sum/count: the running form does not lose the
        # small terms once the totals grow large.
        self.value += (value * weight - self.value) / self.count
        return self.value


class NormalizedEMA:
    """EMA divided by the EMA of its weights, removing the zero-init bias."""

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.unnormalized = ExponentialMovingAverage(self.beta)
        self.normalizer = ExponentialMovingAverage(self.beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.unnormalized.reset()
        self.normalizer.reset()

    def update(self, value: float, weight: float) -> float:
        value = _require_finite("value", value)
        weight = _require_finite("weight", weight)
        unnormalized = self.unnormalized.update(value, weight)
        normalizer = self.normalizer.update(weight, 1.0)
        self.value = unnormalized / normalizer
        return self.value


class TimeDecayedEMA:
    """Zero-initialized EMA that decays per unit of elapsed time.

    Retains ``exp(-lambda * duration)`` instead of a fixed ``1 - beta`` per
    step, so the average depends on elapsed time rather than on how that time
    was chopped into transitions.  ``level`` is already a rate.
    """

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        self.lambda_ = _time_decay(self.beta)
        self.reset()

    def reset(self) -> None:
        self.value = 0.0

    def update(self, level: float, duration: float) -> float:
        level = _require_finite("level", level)
        duration = _require_duration(duration)
        retention = math.exp(-self.lambda_ * duration)
        gain = -math.expm1(-self.lambda_ * duration)
        self.value = retention * self.value + gain * level
        return self.value


# The estimator contract

class RewardRateEstimator:
    """Base class for reward-rate estimators.

    Subclasses implement ``_update``, which receives a validated transition and
    returns the new ``rho``, and extend ``reset`` when they hold further state.
    A weight of 1.0 is the unweighted estimator, so callers state a weight only
    to deviate from it -- as ``WeightedHarmonic`` does by passing the reward.
    """

    #: Estimators that divide by the duration need it strictly positive.  Those
    #: that only ever multiply by it accept an instantaneous transition and set
    #: this to True; a negative duration stays an error either way.
    accepts_zero_duration = False

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.rho = 0.0

    def update(self, reward: float, duration: float, weight: float = 1.0) -> float:
        reward = _require_finite("reward", reward)
        duration = (_require_nonnegative_duration(duration)
                    if self.accepts_zero_duration else _require_duration(duration))
        weight = _require_finite("weight", weight)
        self.rho = self._update(reward, duration, weight)
        return self.rho

    def _update(self, reward: float, duration: float, weight: float) -> float:
        raise NotImplementedError


class ExponentialRateEstimator(RewardRateEstimator):
    """Estimator parameterized by a smoothing rate ``beta``.

    Subclasses must construct their building blocks before delegating here,
    because ``__init__`` resets the whole estimator.
    """

    def __init__(self, beta: float):
        self.beta = _require_beta(beta)
        super().__init__()


# Exponential reward-rate estimators

class _TimeDecayedRate(ExponentialRateEstimator):
    """Estimator backed by a time-decayed average of the transition rate."""

    def __init__(self, beta: float):
        self.unnormalized = TimeDecayedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.unnormalized.reset()

    @property
    def lambda_(self) -> float:
        return self.unnormalized.lambda_

    def _smoothed_rate(self, reward: float, duration: float, weight: float) -> float:
        return self.unnormalized.update(reward * weight / duration, duration)


class ExponentialMovingTimeRate(_TimeDecayedRate):
    """Unnormalized exponential moving time-rate estimator."""

    def _update(self, reward: float, duration: float, weight: float) -> float:
        return self._smoothed_rate(reward, duration, weight)


class NormalizedExponentialMovingTimeRate(_TimeDecayedRate):
    """Time-rate estimator divided by the decayed weight, removing the zero-init bias.

    Same construction as ``NormalizedEMA``, in elapsed time: the driving weight
    is the constant 1.0, so the normalizer converges to
    ``1 - prod(exp(-lambda * duration_i))``.
    """

    def __init__(self, beta: float):
        self.normalizer = TimeDecayedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.normalizer.reset()

    def _update(self, reward: float, duration: float, weight: float) -> float:
        unnormalized = self._smoothed_rate(reward, duration, weight)
        return unnormalized / self.normalizer.update(1.0, duration)


class ExponentialMovingRatioRate(ExponentialRateEstimator):
    """Ratio of a smoothed reward to a smoothed duration."""

    def __init__(self, beta: float):
        self.reward_ema = NormalizedEMA(beta)
        self.duration_ema = NormalizedEMA(beta)
        super().__init__(beta)

    def reset(self) -> None:
        super().reset()
        self.reward_ema.reset()
        self.duration_ema.reset()

    @property
    def mean_reward(self) -> float:
        return self.reward_ema.value

    @property
    def mean_duration(self) -> float:
        return self.duration_ema.value

    def _update(self, reward: float, duration: float, weight: float) -> float:
        reward_ema = self.reward_ema.update(reward, weight)
        duration_ema = self.duration_ema.update(duration, 1.0)
        # _require_duration keeps duration_ema != 0
        return reward_ema / duration_ema


# Cumulative reward-rate estimators

class CumulativeTimeRate(RewardRateEstimator):
    """Cumulative weighted reward divided by unweighted elapsed duration."""

    def reset(self) -> None:
        super().reset()
        self.total_reward = 0.0
        self.total_duration = 0.0

    def _update(self, reward: float, duration: float, weight: float) -> float:
        self.total_reward += reward * weight
        self.total_duration += duration
        return self.total_reward / self.total_duration


# ---------------------------------------------------------------- mixins ---
# Each mixin owns one estimator and exposes the ``calc_new_rho`` hook that
# ``PPO.update_rho`` calls. They stand in for the reference's tabular
# ``SMART`` / ``RelaxedSMART`` / ``SmoothedSMART`` agents, whose only role in the
# deep agents is that hook (and the ``rho_learning_rate`` they are built with).


class RateAgent:
    """Root of the rho hierarchy: holds ``rho`` and the rho learning rate."""

    def __init__(self, rho_learning_rate=0.3, **kwargs):
        super().__init__()
        self.rho_learning_rate = rho_learning_rate
        self.rho = 0.0

    def calc_new_rho(self, reward, time, td_target, td_error):
        """Plain PPO keeps ``rho`` at zero."""


class SMART(RateAgent):
    """``rho = sum(r) / sum(tau)`` over the whole run (DS, Eq. 13)."""

    def __init__(self, rho_learning_rate=0.3, **kwargs):
        super().__init__(rho_learning_rate=rho_learning_rate, **kwargs)
        self.rate = CumulativeTimeRate()

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.rate.update(reward, time, 1.0)


class RelaxedSMART(RateAgent):
    """``rho = EMA(r) / EMA(tau)`` (DS_relaxed / R-SMART, Eq. 14)."""

    def __init__(self, rho_learning_rate=0.3, **kwargs):
        super().__init__(rho_learning_rate=rho_learning_rate, **kwargs)
        self.ratio_rate = ExponentialMovingRatioRate(rho_learning_rate)

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.ratio_rate.update(reward, time)


class SmoothedSMART(RateAgent):
    """``rho`` time-decayed in elapsed time (DS_smoothed / S-SMART, Eq. 15)::

        rho <- exp(-lambda*tau) * rho + (1 - exp(-lambda*tau)) * (r / tau)

    with ``lambda = -log(1 - rho_learning_rate)``, bias-corrected by the decayed
    weight.
    """

    def __init__(self, rho_learning_rate=0.3, **kwargs):
        super().__init__(rho_learning_rate=rho_learning_rate, **kwargs)
        self.time_rate = NormalizedExponentialMovingTimeRate(rho_learning_rate)

    @property
    def lambda_(self):
        return self.time_rate.lambda_

    def calc_new_rho(self, reward, time, td_target, td_error):
        self.rho = self.time_rate.update(reward, time)
