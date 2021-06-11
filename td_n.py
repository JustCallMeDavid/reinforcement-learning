import numpy as np
import matplotlib.pyplot as plt
from typing import *
import random
import pandas as pd
import seaborn as sb
import copy

class Env:
    def __init__(self, dims: Tuple, start: Tuple, goal: Tuple, obstacle_list: List):
        self.dims = dims
        self.actions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        self.start = start
        self.goal = goal
        self.obstacles = obstacle_list
        self.current_state = start

    def _check_bounds(self, s):
        """checks whether the agent crossed the env bounds and resets it within them"""
        pos_y = s[0]
        pos_x = s[1]

        if s[0] < 0:
            pos_y = 0
        if s[1] < 0:
            pos_x = 0
        if s[0] >= self.dims[0]:
            pos_y = self.dims[0] - 1
        if s[1] >= self.dims[1]:
            pos_x = self.dims[1] - 1

        return tuple([pos_y, pos_x])

    def _add_tuple(self, t1, t2):
        return (t1[0]+t2[0],t1[1]+t2[1])

    def step(self, a):
        new_state = self._add_tuple(self.current_state, a)
        new_state = self._check_bounds(new_state)

        if new_state in self.obstacles:
            new_state = self.current_state  # change in env not valid, reset state

        if new_state == self.goal:
            reward = 1.0
        else:
            reward = 0.0

        self.current_state = new_state
        return reward

    def plot(self):
        arr = np.zeros(self.dims)
        arr[self.start] = 1
        arr[self.goal] = 2
        for o in self.obstacles:
            arr[o] = 3
        plt.imshow(arr)
        plt.show()

    def reset(self):
        self.current_state = self.start


def choose_action(state, q_value, env):
    if np.random.uniform(0, 1) < 0.05:  # eps
        return env.actions[random.randint(0, len(env.actions) - 1)]
    else:
        values = q_value[state[0], state[1]]
        potential_actions = [a for a, v in values.items() if v == np.max(list(values.values()))]
        return potential_actions[random.randint(0, len(potential_actions) - 1)]




def td_n(env, qs, n):
    t = 0
    T = np.inf
    states = [env.current_state]
    rewards = [0]
    actions = []
    while True:
        t += 1

        if t < T:
            a = choose_action(env.current_state, qs, env)
            reward = env.step(a)
            states.append(env.current_state)
            rewards.append(reward)
            actions.append(a)

            if env.current_state == env.goal:
                T = t

        ut = t - n
        if ut >= 0:
            returns = 0.0
            for t_ in range(ut + 1, min(T, ut + n) + 1):
                returns += pow(0.95, t_ - ut - 1) * rewards[t_]
            if ut + n <= T:
                returns += pow(0.95, n) * np.max(list(qs[states[ut + n]].values()))
            q_up = (states[ut], actions[ut])
            if q_up != env.goal:
                qs[q_up[0]][q_up[1]] += 0.1 * (returns - qs[q_up[0]][q_up[1]])
        if ut == T - 1:
            break

    return len(actions)


def create_qs():
    qs = {}
    for y in range(env.dims[0]):
        for x in range(env.dims[1]):
            qs[(y, x)] = {}
            for a in env.actions:
                qs[(y, x)][a] = 0  # initialize all qs to 0

    return qs


env = Env(dims=(6, 9), start=(2, 0), goal=(0, 8),
          obstacle_list=[(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)])
env.plot()



reps = 30
episodes = 100

td_list = []
for run in range(reps):
    for n in [10]:
        qs_regular = create_qs()
        qs_delayed = create_qs()
        for ep in range(episodes):
            env.reset()
            td_list.append([run, ep,n,  td_n(env, qs_regular, n)])


df_td = pd.DataFrame(td_list, columns=['Run', 'Episode', 'n', 'NumStepsTaken'])
df_plt_td = df_td.groupby(by=['Episode', 'n']).NumStepsTaken.mean().reset_index()
sb.lineplot(data=df_plt_td, x='Episode', y='NumStepsTaken')
plt.show()