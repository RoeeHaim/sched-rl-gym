"""Sanity checks on a DSMART training log (JSONL from ``train_dsmart.py``).

Checks, over the second half of training:
  * rho tracks the empirical rate sum(r)/sum(tau) (cumulative for DS, a
    recent window of batches for the forgetting estimators);
  * advantages are zero-centred after A-centering;
  * critic values stay bounded (|V| small relative to the return scale);
  * policy entropy does not collapse.

    python scripts/dsmart_sanity.py runs/ds.jsonl [more.jsonl ...]
"""

import json
import sys

import numpy as np


def check(path, window=10):
    recs = [json.loads(line) for line in open(path)]
    cfg = next(r for r in recs if r["kind"] == "config")
    tr = [r for r in recs if r["kind"] == "train"]
    ev = [r for r in recs if r["kind"] == "eval"]
    algo = cfg["algo"]
    half = tr[len(tr) // 2 :]
    out = {"file": path, "algo": algo, "iters": len(tr)}

    rho = np.array([r["rho"] for r in half])
    if algo == "DS":
        target = np.array([r["cum_rate"] for r in half])
    else:
        emp = np.array([r["emp_rate"] for r in tr])
        roll = np.convolve(emp, np.ones(window) / window, mode="full")[: len(emp)]
        target = roll[len(tr) // 2 :]
    rel = np.abs(rho - target) / np.maximum(np.abs(target), 1e-8)
    out["rho_rel_err_median"] = float(np.median(rel))
    out["rho_ok"] = bool(np.median(rel) < (0.05 if algo == "DS" else 0.5))

    adv = np.array([abs(r["adv_mean"]) for r in half])
    sd = np.array([r.get("adv_std") or 1.0 for r in half])
    out["adv_mean_over_std_max"] = float(np.max(adv / sd))
    out["adv_ok"] = bool(np.max(adv / sd) < 1e-3) if algo != "PPO" else None

    v = np.array([max(abs(r["v_min"]), abs(r["v_max"])) for r in half])
    out["v_abs_max"] = float(v.max())
    out["v_bounded"] = bool(np.all(np.isfinite(v)) and v[-1] <= 3 * max(v[: len(v) // 2 + 1].max(), 1.0))

    ent = np.array([r["entropy"] for r in tr])
    out["entropy_start"] = float(ent[: max(1, len(ent) // 10)].mean())
    out["entropy_end"] = float(ent[-max(1, len(ent) // 10) :].mean())
    out["entropy_ok"] = bool(out["entropy_end"] > 0.05)
    if ev:
        out["eval_first"] = ev[0]["greedy_reward_rate"]
        out["eval_last"] = ev[-1]["greedy_reward_rate"]
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(check(p)))
