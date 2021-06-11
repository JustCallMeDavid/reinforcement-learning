import numpy as np
import matplotlib.pyplot as plt
from typing import *
import random
import pandas as pd
import seaborn as sb


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
        return (t1[0] + t2[0], t1[1] + t2[1])

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


class EnvModel:
    def __init__(self):
        self.model = dict()

    def feed(self, state, action, next_state, reward):
        self.model[(state, action)] = (next_state, reward)

    def sample(self):
        state_action = random.choice(list(self.model.keys()))
        return state_action, self.model[state_action]


def dynaQ(model, env, planning_steps, qs):
    prev_state = env.start
    steps = 0

    while prev_state != env.goal:

        steps += 1
        action = choose_action(prev_state, qs, env)
        reward = env.step(action)

        qs[prev_state][action] += 0.1 * (
                reward + 0.95 * np.max(list(qs[env.current_state].values())) - qs[prev_state][action])

        model.feed(prev_state, action, env.current_state, reward)  # update the model

        for t in range(planning_steps):
            s_a, n_r = model.sample()
            qs[s_a[0]][s_a[1]] += 0.1 * (n_r[1] + 0.95 * np.max(list(qs[n_r[0]].values())) - qs[s_a[0]][s_a[1]])

        prev_state = env.current_state

        if steps > 1000000:
            break

    return steps


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
plt.show()

reps = 100
episodes = 50
steps = {0: 0, 5: 0, 50: 0}
entry_list = []
for planning_step in steps.keys():
    for run in range(reps):
        model = EnvModel()
        qs = create_qs()
    for ep in range(episodes):
        env.reset()
        entry_list.append([planning_step, run, ep, dynaQ(model, env, planning_step, qs)])

df = pd.DataFrame(entry_list, columns=['PlanningStep', 'Run', 'Episode', 'NumStepsTaken'])

df_plt_dyna = df.groupby(by=['PlanningStep',
                             'Episode']).NumStepsTaken.mean().reset_index()
sb.lineplot(data=df_plt_dyna, x='Episode', hue='PlanningStep', y='NumStepsTaken')
plt.show()
