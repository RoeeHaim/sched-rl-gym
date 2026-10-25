"""Train a MaskablePPO agent on CompactRM-v0 (SMDP mode by default).

Two discounting modes (--discount):
  smdp      (default) gamma is per *time unit*: the reward of a transition is
            discounted inside it (stage 1) and the next state's value is
            discounted by gamma**tau, tau = elapsed time (stage 2).
            See scripts/smdp_ppo.py.
  decision  standard PPO: gamma per decision, ignoring how much time passed
            (the earlier "option A").

Usage (from the sched-rl-gym folder):
    python scripts/train_ppo.py                          # 2M steps, 8 parallel envs
    python scripts/train_ppo.py --timesteps 200000 --n-envs 4 --out models/quick
    tensorboard --logdir models/                         # watch training curves

Output folder (default models/ppo_<date>):
    model.zip                    final model
    config.json                  environment + training settings (used by evaluate.py)
    checkpoints/                 intermediate models
    tb/                          TensorBoard logs
"""

import argparse
import json
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from ppo_utils import DEFAULT_CONFIG, make_env
from smdp_ppo import SmdpMaskablePPO, VecSmdpReward

# Training episodes use seeds far away from the evaluation seeds (0, 1, 2, ...),
# so the agent is never evaluated on job sequences it was trained on.
TRAIN_SEED_OFFSET = 1_000_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=2_000_000, help="total decisions to train on")
    ap.add_argument("--n-envs", type=int, default=8, help="parallel environments")
    ap.add_argument("--sim", default=DEFAULT_CONFIG["sim"], choices=["event_based", "time_based"])
    ap.add_argument("--slots", type=int, default=DEFAULT_CONFIG["slots"])
    ap.add_argument("--time-limit", type=int, default=DEFAULT_CONFIG["time_limit"])
    ap.add_argument("--processors", type=int, default=DEFAULT_CONFIG["processors"])
    ap.add_argument("--memory", action="store_true", help="model memory too")
    ap.add_argument("--rate", type=float, default=DEFAULT_CONFIG["rate"])
    ap.add_argument("--estimates", default=None, choices=["gaussian-over", "gaussian", "gaussian-under"],
                    help="inaccurate user run-time estimates (default: exact). gaussian-over: users "
                         "over-estimate, as in real traces. gaussian-under triggers bug 5")
    ap.add_argument("--estimate-noise", type=float, default=DEFAULT_CONFIG["estimate_noise"],
                    help="std of the estimate error relative to the run time (with --estimates)")
    ap.add_argument("--discount", default="smdp", choices=["smdp", "decision"],
                    help="smdp: gamma per time unit using tau; decision: gamma per decision")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor (per time unit with --discount smdp)")
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048, help="steps per env per update")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-norm-reward", action="store_true",
                    help="disable reward normalization (rewards here reach hundreds per step)")
    ap.add_argument("--checkpoint-every", type=int, default=200_000, help="in total timesteps")
    ap.add_argument("--out", default=None, help="output folder (default models/ppo_<date>)")
    args = ap.parse_args()

    cfg = {
        "sim": args.sim, "slots": args.slots, "time_limit": args.time_limit,
        "processors": args.processors, "memory": args.memory, "rate": args.rate,
        "estimates": args.estimates, "estimate_noise": args.estimate_noise,
    }
    out = Path(args.out or f"models/ppo_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump({"env": cfg, "train": vars(args)}, f, indent=2)

    def factory(rank):
        return lambda: make_env(cfg, base_seed=TRAIN_SEED_OFFSET + args.seed * 1000 + rank)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    envs = vec_cls([factory(i) for i in range(args.n_envs)])
    # VecMonitor logs episode return/length (rollout/ep_rew_mean in TensorBoard).
    envs = VecMonitor(envs)  # below VecSmdpReward: logged returns stay raw/undiscounted
    if args.discount == "smdp":
        envs = VecSmdpReward(envs, gamma=args.gamma)  # stage 1: R_i from info["step_rewards"]
    # Rewards are -sum(1/execution_time) over jobs in the system, accumulated over
    # the elapsed time, so a single 'wait' can be worth -400. Unscaled, the value
    # loss swamps the policy loss and PPO collapses to 'always wait'. Only rewards
    # are normalized (not observations), so evaluation needs no saved statistics.
    if not args.no_norm_reward:
        envs = VecNormalize(envs, norm_obs=False, norm_reward=True, gamma=args.gamma)

    algo = SmdpMaskablePPO if args.discount == "smdp" else MaskablePPO
    model = algo(
        "MlpPolicy", envs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, learning_rate=args.lr, n_steps=args.n_steps,
        batch_size=args.batch_size, ent_coef=args.ent_coef,
        seed=args.seed, tensorboard_log=str(out / "tb"), verbose=1,
    )
    checkpoints = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),
        save_path=str(out / "checkpoints"), name_prefix="ppo",
    )
    print(f"discount: {args.discount} (gamma={args.gamma}, lambda={args.gae_lambda})")
    print(f"training {args.timesteps} steps on {args.n_envs} envs, config {cfg}")
    print(f"output: {out}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=checkpoints)
    model.save(out / "model")
    envs.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out / 'model.zip'}")
    print(f"evaluate with: python scripts/evaluate.py --model {out}")


if __name__ == "__main__":
    main()
