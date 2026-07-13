#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate heuristic baseline agents (SJF / Random) on the DeepRM env.

Headless version of sched-rl-gym's sjf-agent.py: no rendering, multiple
episodes, and reports the metrics used in the paper (total discounted
reward and average job slowdown).
"""

import argparse

import gymnasium as gym
import numpy as np

import schedgym.envs as deeprm  # noqa: F401 (registers DeepRM-v0)


def split_observation(observation):
    """Groups the raw observation into (current, wait, backlog, time).

    The raw state is a 6-tuple (processors and memory as separate arrays)
    unless the env ignores memory, in which case it is a 4-tuple.
    """
    if len(observation) == 6:
        c_procs, c_mem, w_procs, w_mem, backlog, time = observation
        return (
            np.stack((c_procs, c_mem)),
            np.stack((w_procs, w_mem)),
            backlog,
            time,
        )
    current, wait, backlog, time = observation
    return current[None], wait[None], backlog, time


def sjf_action(observation):
    "Selects the job SJF (Shortest Job First) would select."
    current, wait, _, _ = split_observation(observation)
    best = wait.shape[2] + 1  # infinity
    best_idx = wait.shape[1]

    free = np.ones(current.shape[0]) * current.shape[-1] - np.sum(
        current[:, 0, :] != 0
    )

    for slot in range(wait.shape[1]):
        required_resources = (wait[:, slot, 0, :] != 0).sum(axis=1)
        if np.all(required_resources) and np.all(required_resources <= free):
            tmp = np.sum(wait[0, slot, :, 0])
            if tmp < best:
                best_idx = slot
                best = tmp
    return best_idx


def random_action(observation):
    _, wait, _, _ = split_observation(observation)
    return np.random.randint(0, wait.shape[1] + 1)


AGENTS = {'sjf': sjf_action, 'random': random_action}


def run_episode(env, policy, max_episode_length):
    ob, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_episode_length):
        ob, reward, terminated, truncated, _ = env.step(policy(ob))
        total_reward += reward
        if terminated or truncated:
            break
    slowdowns = env.scheduler.slowdown
    return total_reward, np.mean(slowdowns) if slowdowns else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-episode-length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for name, policy in AGENTS.items():
        env = gym.make('DeepRM-v0', use_raw_state=True).unwrapped
        env.seed(args.seed)
        np.random.seed(args.seed)
        rewards, slowdowns = [], []
        for _ in range(args.episodes):
            r, s = run_episode(env, policy, args.max_episode_length)
            rewards.append(r)
            slowdowns.append(s)
        env.close()
        print(
            f'{name:>8}: total reward = {np.mean(rewards):8.2f} '
            f'± {np.std(rewards):6.2f} | '
            f'avg slowdown = {np.nanmean(slowdowns):6.3f} '
            f'± {np.nanstd(slowdowns):5.3f}'
        )


if __name__ == '__main__':
    main()
