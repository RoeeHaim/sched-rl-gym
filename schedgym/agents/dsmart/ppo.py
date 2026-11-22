"""PPO core and rollout buffer for the DSMART agents (torch only).

Ported from ``erelon/SMDP_Agents`` (``smdp_agents/ppo.py`` and
``smdp_agents/gaussian_mlp.py``, commit 84bd5db). The update path is the
reference's line for line; the changes are:

* **Action masking** (sched-rl-gym only exposes ``action_masks()``). The
  categorical actor masks invalid logits before sampling, in the update and in
  the entropy, and the buffer stores the mask of every decision. Masks are
  optional: without them every action is valid, exactly as in the reference.
* **Zero-duration decisions.** Starting a job does not advance the clock, so
  this environment reports ``tau = 0`` for every scheduling action. Their
  reward is 0, so they carry no information about the rate: the per-transition
  rho update (``rho_reduce="none"``) skips them (exact for the time-decayed
  estimator, since ``exp(-lambda*0) = 1``), and a batch with no elapsed time
  leaves rho unchanged instead of dividing by zero.
* The Gaussian head is dropped: the scheduling action space is discrete.
* A few extra diagnostics in the stats dict (advantage mean before/after
  shaping, value statistics).

The holding time ``time`` feeds GAE through ``rate_residual`` (``r - rho*tau``)
and the SMDP discount ``gamma^tau`` (1 for the average-reward agents).
"""

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from .rates import RateAgent

MASKED_LOGIT = -1e8


def _t(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _vmean(x, valid):
    """Mean of x over valid (>0) entries; plain mean if none are valid."""
    s = valid.sum()
    return (x * valid).sum() / s if s > 0 else x.mean()


def mlp(in_size, hidden, out_size, nonlinearity=nn.Tanh):
    """[in -> hidden... -> out] with a linear last layer."""
    sizes, layers = [in_size] + list(hidden), []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nonlinearity()]
    layers += [nn.Linear(sizes[-1], out_size)]
    return nn.Sequential(*layers)


class CategoricalMLP(nn.Module):
    """Discrete actor-critic: a logits MLP plus a separate state-value MLP."""

    def __init__(self, obs_dim, n_actions, hidden=(64, 64)):
        super().__init__()
        self.logits = mlp(obs_dim, hidden, n_actions)
        self.v = mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.logits(obs), self.v(obs).squeeze(-1)


def masked_dist(logits, mask):
    if mask is not None:
        logits = torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))
    return Categorical(logits=logits)


class RolloutBuffer:
    """Accumulates a time-major [T, B] on-policy batch, one ``add`` per step."""

    _FIELDS = ("obs", "act", "rew", "term", "trunc", "val", "logp", "time", "mask")

    def __init__(self):
        self.clear()

    def clear(self):
        self._d = {k: [] for k in self._FIELDS}

    def __len__(self):
        return len(self._d["rew"])

    def add(self, obs, action, reward, terminated, truncated, value, logp,
            time=1.0, mask=None):
        vals = (obs, action, reward, terminated, truncated, value, logp, time, mask)
        for k, v in zip(self._FIELDS, vals):
            if v is None:
                self._d[k].append(None)
                continue
            self._d[k].append(v.detach().cpu() if torch.is_tensor(v)
                              else torch.as_tensor(np.asarray(v)))

    def stacked(self, device):
        out = {}
        for k in self._FIELDS:
            seq = self._d[k]
            if k == "mask":
                out[k] = (None if any(m is None for m in seq)
                          else torch.stack([m.to(device=device, dtype=torch.bool)
                                            for m in seq]))
                continue
            if k == "time" and all(t.ndim == 0 for t in seq):  # scalar dwell -> [T,B]
                seq = [t.expand_as(self._d["rew"][i]) for i, t in enumerate(seq)]
            out[k] = torch.stack([_t(t, device) for t in seq])
        return out


class PPO(RateAgent):
    """Discounted PPO + the deep-RL machinery. Base for the average-reward
    variants, which add the ``rho`` correction by also inheriting a rate mixin
    (see :mod:`schedgym.agents.dsmart.deep_smart`)."""

    longrun = False  # variants flip this on to enable the rho correction

    def __init__(self, obs_dim, act_dim, hidden=(64, 64),
                 learning_rate=3e-4, rho_lr=0.1, rm_vbias_coeff=1.0,
                 value_loss_coeff=1.0, entropy_loss_coeff=0.01, clip_grad_norm=10.0,
                 discount=None, gae_lambda=0.95, epochs=10, minibatches=20,
                 ratio_clip=0.2, normalize_advantage=False, bootstrap_timelimit=True,
                 batch_T=200, device="cpu", seed=None):
        # rho_lr maps to the rate estimator's rho_learning_rate (MRO cooperative).
        super().__init__(rho_learning_rate=rho_lr)
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.discrete = True
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.hidden = tuple(hidden)
        self.net = CategoricalMLP(obs_dim, act_dim, hidden).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # Average-reward variants default to no discounting; plain PPO to 0.99.
        self.discount = (1.0 if self.longrun else 0.99) if discount is None else discount
        self.gae_lambda = gae_lambda
        self.epochs, self.minibatches = epochs, minibatches
        self.ratio_clip = ratio_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_grad_norm = clip_grad_norm
        self.normalize_advantage = normalize_advantage
        self.bootstrap_timelimit = bootstrap_timelimit
        self.batch_T = batch_T
        self.rho_lr = rho_lr
        self.rho_learning_rate = rho_lr
        self.rho = 0.0
        # Off for plain PPO so the value-bias / rho terms vanish.
        self.rm_vbias_coeff = rm_vbias_coeff if self.longrun else 0.0
        self.value_bias = None if self.longrun else 0.0

    # --- acting -------------------------------------------------------------
    def _mask(self, mask):
        return None if mask is None else _t(mask, self.device, torch.bool)

    @torch.no_grad()
    def act(self, obs, mask=None):
        """Sample an action. Returns (action, value, logp) as cpu tensors."""
        logits, value = self.net(_t(obs, self.device))
        dist = masked_dist(logits, self._mask(mask))
        action = dist.sample()
        return action.cpu(), value.cpu(), dist.log_prob(action).cpu()

    @torch.no_grad()
    def eval_act(self, obs, mask=None):
        """Deterministic action for evaluation (argmax of the masked logits)."""
        logits = self.net(_t(obs, self.device))[0]
        mask = self._mask(mask)
        if mask is not None:
            logits = torch.where(mask, logits, torch.full_like(logits, MASKED_LOGIT))
        return logits.argmax(-1).cpu()

    @torch.no_grad()
    def value(self, obs):
        return self.net(_t(obs, self.device))[-1].cpu()

    # --- average-reward rate ------------------------------------------------
    #
    # ``rho_reduce`` controls how the [T, B] batch feeds the rate estimator:
    #   "mean" -> one call with the batch means  (per-batch EWMA, e.g. Rsmart/APO)
    #   "sum"  -> one call with the batch sums    (cumulative, e.g. SMART)
    #   "none" -> one call per transition         (time-decayed S-SMART)
    rho_reduce = "mean"

    def update_rho(self, reward, value, time):
        """Refresh ``rho`` and ``value_bias`` from a fresh [T, B] batch."""
        if not self.longrun:
            return
        vm = value.mean().item()
        self.value_bias = vm if self.value_bias is None \
            else (1 - self.rho_lr) * self.value_bias + self.rho_lr * vm
        r, t = reward.reshape(-1), time.reshape(-1)
        if not float(t.sum().item()) > 0.0:
            return  # no elapsed time in this batch: nothing to learn about the rate
        if self.rho_reduce == "none":
            for ri, ti in zip(r.tolist(), t.tolist()):
                if ti > 0:  # tau=0 decisions carry r=0 and no time
                    self.calc_new_rho(ri, ti, None, None)
        elif self.rho_reduce == "sum":
            self.calc_new_rho(r.sum().item(), t.sum().item(), None, None)
        else:  # "mean"
            self.calc_new_rho(r.mean().item(), t.mean().item(), None, None)

    @torch.no_grad()
    def policy_entropy(self, obs, mask=None):
        """Mean entropy of the (masked) behaviour policy over a batch."""
        return float(masked_dist(self.net(obs)[0], mask).entropy().mean().item())

    def shape_advantage(self, adv, valid, batch):
        """Transform the advantage after GAE and before the optimisation epochs."""
        if self.normalize_advantage:
            m = valid > 0
            adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-6)
        return adv

    def rate_residual(self, reward, time):
        """The average-reward correction inside the TD residual: ``r - rho*tau``."""
        return reward - self.rho * time

    # --- update -------------------------------------------------------------
    def _gae(self, reward, value, nd, bootstrap_value, time):
        adv = torch.zeros_like(reward)
        nxt, gae = bootstrap_value, 0.0
        for t in reversed(range(reward.shape[0])):
            disc = self.discount ** time[t]
            delta = (self.rate_residual(reward[t], time[t])
                     + disc * nxt * nd[t] - value[t])
            gae = delta + disc * self.gae_lambda * nd[t] * gae
            adv[t] = gae
            nxt = value[t]
        return adv, adv + value

    def update(self, buffer, bootstrap_value):
        """One PPO iteration over the collected [T, B] batch. Returns stats."""
        b = buffer.stacked(self.device)
        reward, value, time = b["rew"], b["val"], b["time"]
        nd = 1.0 - b["term"]  # terminations reset the return; truncations bootstrap

        self.update_rho(reward, value, time)
        adv, ret = self._gae(reward, value, nd, _t(bootstrap_value, self.device), time)
        ret = ret - self.rm_vbias_coeff * (self.value_bias or 0.0)

        valid = (1.0 - b["trunc"]) if self.bootstrap_timelimit else torch.ones_like(reward)
        m = valid > 0
        adv_mean_raw = float(adv[m].mean().item()) if m.any() else float("nan")
        adv = self.shape_advantage(adv, valid, b)
        adv_mean = float(adv[m].mean().item()) if m.any() else float("nan")

        obs = b["obs"].reshape(-1, b["obs"].shape[-1])
        act = b["act"].reshape(-1).long()
        mask = None if b["mask"] is None else b["mask"].reshape(-1, b["mask"].shape[-1])
        old_logp, adv, ret, valid = (x.reshape(-1) for x in (b["logp"], adv, ret, valid))
        n = obs.shape[0]
        mb = max(1, n // self.minibatches)
        stats = {"loss": [], "pi_loss": [], "v_loss": [], "grad_norm": [],
                 "entropy": [], "approx_kl": [], "clip_frac": []}
        for _ in range(self.epochs):
            for idx in torch.randperm(n, device=self.device).split(mb):
                logits, v = self.net(obs[idx])
                dist = masked_dist(logits, None if mask is None else mask[idx])
                new_logp, ent = dist.log_prob(act[idx]), dist.entropy()
                log_ratio = new_logp - old_logp[idx]
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
                w = valid[idx]
                pi_loss = -_vmean(torch.min(ratio * adv[idx], clipped * adv[idx]), w)
                v_loss = self.value_loss_coeff * _vmean(0.5 * (v - ret[idx]) ** 2, w)
                entropy = _vmean(ent, w)
                loss = pi_loss + v_loss - self.entropy_loss_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_frac"].append(
                        ((ratio - 1).abs() > self.ratio_clip).float().mean().item())
                stats["loss"].append(loss.item())
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["grad_norm"].append(gn.item())
                stats["entropy"].append(entropy.item())
        return {"rho": self.rho, "value_bias": self.value_bias or 0.0,
                "adv_mean_raw": adv_mean_raw, "adv_mean": adv_mean,
                "v_mean": float(value.mean().item()),
                "v_min": float(value.min().item()),
                "v_max": float(value.max().item()),
                "ret_mean": float(ret.mean().item()),
                **{k: float(np.mean(v)) for k, v in stats.items()}}
