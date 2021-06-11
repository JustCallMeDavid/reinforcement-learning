"""
Implements a 1d-gridworld example and solves it via policy iteration. Environment consists of 10 states traversed
in a one-dimensional left or right manner. The leftmost state gives a +10 and the rightmost a -5 reward.
"""

import numpy as np
import copy

env = {}
for s in range(10):
    for a in [-1, +1]:
        # terminal states
        if s == 0:
            env[(s, a)] = (10, 0)
        elif s == 9:
            env[(s, a)] = (-5, 9)
        else:
            env[(s, a)] = (0, s + a)

policy = {}
for s in range(10):
    # equiprobable initialization
    policy[s] = [(-1, 0.5), (1, 0.5)]


def policy_evaluation(policy, env, discount=1.0, theta=0.0001):
    Vs = np.zeros(10)
    delta = np.inf
    while delta > theta:
        delta = 0
        for s in range(10):
            v_s = 0
            if s == 0 or s == 9:
                continuing = 0
            else:
                continuing = 1
            for action, action_prob in policy[s]:
                reward, next_state = env[(s, action)]
                v_s += action_prob * (reward + discount * continuing * Vs[next_state])
            delta = max(delta, np.abs(v_s - Vs[s]))
            Vs[s] = v_s
    return Vs


def policy_iteration(policy, env, discount_factor=1.0, policy_eval_fn=policy_evaluation):
    change = True
    while change:
        change = False
        Vs = policy_eval_fn(policy, env, discount_factor)
        old_policy = copy.deepcopy(policy)
        for s in range(10):
            policy[s] = [([-1, 1][np.argmax([
                env[(s, -1)][0] + discount_factor * Vs[env[(s, -1)][1]],
                env[(s, 1)][0] + discount_factor * Vs[env[(s, 1)][1]]
            ])], 1)]
        if policy != old_policy:
            change = True
    return policy, Vs


print(policy_iteration(policy, env))
