#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a trained DeepRM policy checkpoint against SJF and random.

Usage: python eval_policy.py <checkpoint.pth> [--episodes N]
"""

import argparse
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import schedgym.envs as deeprm  # noqa: F401 (registers DeepRM-v0)

from eval_baselines import AGENTS, run_episode


class PGNet(nn.Module):
    "Same architecture as deeprm-agent.py"

    def __init__(self, env):
        super().__init__()
        self.input_height = env.observation_space.shape[0]
        self.input_width = env.observation_space.shape[1]
        self.output_size = env.action_space.n
        self.nn = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_height * self.input_width, 512)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(512, 256)),
            ('relu2', nn.ReLU()),
        ]))
        self.out = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_height * self.input_width)
        return F.softmax(self.out(self.nn(x)), dim=1)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = self(state)
        return Categorical(probs).sample().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-episode-length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # The RL policy sees the flattened image state (use_raw_state=False),
    # while the heuristics use the raw state tuple.
    policy_env = gym.make('DeepRM-v0').unwrapped
    model = PGNet(policy_env)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    agents = dict(AGENTS)
    results = {}

    for name, policy in agents.items():
        env = gym.make('DeepRM-v0', use_raw_state=True).unwrapped
        env.seed(args.seed)
        np.random.seed(args.seed)
        vals = [run_episode(env, policy, args.max_episode_length)
                for _ in range(args.episodes)]
        env.close()
        results[name] = vals

    policy_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    vals = [run_episode(policy_env, model.select_action,
                        args.max_episode_length)
            for _ in range(args.episodes)]
    policy_env.close()
    results['deeprm'] = vals

    print(f'{"agent":>8} | {"total reward":>20} | {"avg slowdown":>16}')
    print('-' * 52)
    for name, vals in results.items():
        rewards = [v[0] for v in vals]
        slowdowns = [v[1] for v in vals]
        print(
            f'{name:>8} | {np.mean(rewards):9.2f} ± {np.std(rewards):7.2f} | '
            f'{np.nanmean(slowdowns):7.3f} ± {np.nanstd(slowdowns):5.3f}'
        )


if __name__ == '__main__':
    main()
