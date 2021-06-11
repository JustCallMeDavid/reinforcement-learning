import numpy as np
import matplotlib.pyplot as plt
import random


class WindyGridWorld:

    def __init__(self, height, width, starting_state, terminal_state, wind_vector, action_set):
        self.height = height
        self.width = width
        self.starting_state = starting_state
        self.grid = np.zeros((height, width))
        self._current_location = starting_state
        self.terminal_state = terminal_state
        self.terminated = False
        self.wind_vector = wind_vector

        assert len(self.wind_vector) == self.width

        if action_set == 'regular':
            self.ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if action_set == 'king':
            self.ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        if action_set == 'king+':
            self.ACTIONS = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    def action(self, a):
        new_location = tuple(np.add(np.add(self.current_location, a), (self.wind_vector[self.current_location[1]], 0)))

        if new_location[0] >= self.height:
            new_location = (self.height - 1, new_location[1])
        if new_location[1] >= self.width:
            new_location = (new_location[0], self.width - 1)
        if new_location[0] < 0:
            new_location = (0, new_location[1])
        if new_location[1] < 0:
            new_location = (new_location[0], 0)

        self._current_location = new_location

        if tuple(self.current_location) == self.terminal_state:
            self.terminated = True
            return 0

        return -1

    def reset(self):
        self._current_location = self.starting_state
        self.terminated = False

    @property
    def current_location(self):
        return tuple(self._current_location)


class Agent:

    def __init__(self, epsilon, alpha, gamma, env: WindyGridWorld):
        self.Qs = {}
        self.epsilon = epsilon
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        inds = [(i, j) for i in range(env.height) for j in range(env.width)]
        for pos in (inds):
            self.Qs[pos] = {}
            for a in env.ACTIONS:
                self.Qs[pos][a] = 0

    def act(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.ACTIONS)
        else:
            return max(self.Qs[env.current_location], key=self.Qs[env.current_location].get)

    def update_qs(self, old_state, reward, new_state, a):
        self.Qs[old_state][a] = (1 - self.alpha) * self.Qs[old_state][a] \
                                + self.alpha * (reward + self.gamma * max(self.Qs[new_state].values()))


env = WindyGridWorld(height=7, width=10, starting_state=(4, 0), terminal_state=(4, 7), action_set='king+',
                     wind_vector=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

agent = Agent(env=env, alpha=0.1, gamma=1.0, epsilon=0.1)
reward_per_episode = []
for trial in range(500):
    cumulative_reward = 0
    env.reset()
    while not env.terminated:
        old_state = env.current_location
        a = agent.act()
        reward = env.action(a)
        assert type(a) == tuple
        new_state = env.current_location
        assert new_state in agent.Qs.keys()
        agent.update_qs(old_state, reward, new_state, a)
        cumulative_reward += reward
    reward_per_episode.append(cumulative_reward)
plt.plot(reward_per_episode)
plt.show()
