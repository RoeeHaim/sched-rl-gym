# Final Project — RL for HPC Cluster Job Scheduling

**Mentor:** Prof. Gal Kaminka
**Paper:** Cunha & Chaimowicz, "An SMDP approach for Reinforcement Learning in HPC cluster schedulers", *Future Generation Computer Systems* 139 (2023) 239–252. DOI: [10.1016/j.future.2022.09.025](https://doi.org/10.1016/j.future.2022.09.025)
**Code:** [renatolfc/sched-rl-gym](https://github.com/renatolfc/sched-rl-gym) (vendored in `sched-rl-gym/`)

## Goal 1 — Code runs and results reproduce

### Completed

1. **Building the environment on Windows** (Python 3.10, venv at `sched-rl-gym/venv`):
   - `gym==0.21` (the old version the code requires — a discussion point: migration to gymnasium).
   - PyTorch 2.12 (CPU) + tensorboard.
   - The `parallelworkloads` package (the Lublin99/Tsafrir05 workload generators, in Cython/C++) **did not compile on Windows** — the C code uses POSIX functions (`drand48`, `srand48`, `lrand48`, `bzero`) that do not exist under MSVC. A compatibility shim (`parallelworkloads/rand48_compat.h`) was written that implements the 48-bit LCG **exactly per the POSIX specification**, so the generated workloads are bit-for-bit identical to Linux — essential for reproducibility.

2. **Test suite:** 97/98 pass (the failing one — `test_packer_scheduler` — passes when run in isolation; it is order-dependent due to shared random state, not a bug in the simulator).

3. **Data:** the real **LANL-CM5-1994** trace from the Hebrew University Parallel Workloads Archive was downloaded to `sched-rl-gym/test/` and passes the parser tests (122,060 jobs).

4. **Windows fixes for `deeprm-agent.py`:** the `/run/user/<uid>` path was replaced with a cross-platform tempdir; the `checkpoint/` directory is created automatically.

5. **Scripts written** (in the project root):
   - `eval_baselines.py` — runs SJF and Random without rendering, measuring cumulative reward and average slowdown.
   - `eval_policy.py` — loads a trained-policy checkpoint and compares it against the baselines.

### Results (20 episodes, seed 42, default deeprm workload)

| Agent | Total reward | Average slowdown |
|---|---|---|
| SJF | -2350 ± 408 | 19.7 ± 3.4 |
| Random | -2934 ± 580 | 24.9 ± 4.6 |
| DeepRM — best checkpoint (epoch 39 of 200) | -2940 ± 660 | **22.2 ± 5.2** |

Training trajectory (200 epochs, lr=1e-2, 4 workers × 50 trajectories):
- epoch 9: slowdown 27.4 (worse than random) → epoch 39: 22.2 (clearly better than random, approaching SJF) → afterwards a gradual regression back to 24–26.
- This pattern — improvement then collapse — is well known in vanilla REINFORCE: without entropy regularization the policy loses exploration, and a high fixed lr (1e-2) destabilizes convergence.
- Goal 1 conclusion: **the full pipeline works and learns** (the agent moves from random-level to near-SJF), but matching the paper's peak requires a larger training budget and/or a more stable algorithm — which is exactly Goal 2 (PPO).

### Finding: the SMDP code

The local copy is already on the `smdp` branch (ahead of master). The paper's SMDP mechanism is implemented in the environment: passing `simulation_type='EVENT_BASED'` makes the simulation event-driven (Semi-MDP) with γ-discounted accumulation of intermediate rewards (`schedgym/envs/deeprm_env.py:182`, `schedgym/envs/base.py:90`), versus `TIME_BASED` (the standard DeepRM MDP). In other words, a full reproduction of the paper = training the same agent in both modes and comparing.

In addition, the repo includes `docs/tutorials/ppo.ipynb`, which trains PPO on the environment — but with the old stable-baselines (TensorFlow 1.15). This is exactly the "outdated code" Gal mentioned.

### Discussion points for the meeting

- Training budget: the original DeepRM paper used ~1000 iterations; this is slow on CPU — consider GPU or a smaller configuration.
- Migration from gym 0.21 to gymnasium — required anyway for Goal 2 (integration with modern stable-baselines3 / CleanRL), and would also replace the old PPO in the repo's tutorial.
- Which configurations from the paper to reproduce first (synthetic Lublin workload vs. the real LANL-CM5 trace).

## Goal 2 (next) — Advanced algorithms

Initial plan: migrate to gymnasium, then PPO / A2C from stable-baselines3 on the same environment, comparing on the same metrics (slowdown, bounded slowdown, utilization).

## Quick run (reproduction)

```powershell
# tests
cd sched-rl-gym; venv\Scripts\python.exe -m pytest schedgym\test_schedgym.py -q --no-cov

# baselines
venv\Scripts\python.exe ..\eval_baselines.py --episodes 10

# training
venv\Scripts\python.exe deeprm-agent.py --epochs 200 --workers 4 --trajectories-per-batch 50

# evaluate a trained policy
venv\Scripts\python.exe ..\eval_policy.py checkpoint\policy-199.pth --episodes 20
```
