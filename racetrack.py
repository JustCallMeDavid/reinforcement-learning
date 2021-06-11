import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import random
import copy
import tqdm


class Map(ABC):
    OUTSIDE_MAP = -1
    INSIDE_MAP = 1
    START_STATE = 0
    TERMINAL_STATE = 2
    AGENT = 3

    def __init__(self):
        self.map = self._build_environment()
        self.shape = self.map.shape

    @abstractmethod
    def _build_environment(self):
        pass

    def _create_row(self, pad_left, tiles, pad_right):
        assert pad_left >= 0 and pad_right >= 0 and tiles >= 0, "Counts cannot be negative"
        return (pad_left * [self.OUTSIDE_MAP]) + (tiles * [self.INSIDE_MAP]) + (pad_right * [self.OUTSIDE_MAP])

    def plot_map(self, agent_position=None):
        if agent_position is None:
            plt.imshow(self.map)
        else:
            map = copy.deepcopy(self.map)
            map[agent_position[0], agent_position[1]] = self.AGENT
            plt.imshow(map)

        plt.show()

    def get_position_space(self):
        return list(tuple(map(tuple, np.argwhere(self.map == self.TERMINAL_STATE)))) + \
               list(tuple(map(tuple, np.argwhere(self.map == self.START_STATE)))) + \
               list(tuple(map(tuple, np.argwhere(self.map == self.INSIDE_MAP))))

    def get_start_space(self):
        return list(tuple(map(tuple, np.argwhere(self.map == self.START_STATE))))

    def get_terminal_space(self):
        return list(tuple(map(tuple, np.argwhere(self.map == self.TERMINAL_STATE))))

    def get_random_start_pos(self):
        return random.choice(list(tuple(map(tuple, np.argwhere(np.array(self.map) == self.START_STATE)))))

    def crossed_finish_line(self, agent_position, velocity_x, velocity_y):
        finish_coords = tuple(map(tuple, np.argwhere(self.map == self.TERMINAL_STATE)))

        for x in range(velocity_x + 1):
            for y in range(velocity_y + 1):
                pos_y = agent_position[0] - y
                pos_x = agent_position[1] + x
                if tuple([pos_y, pos_x]) in finish_coords:
                    return True

        return False

    def crossed_track_boundary(self, agent_position, velocity_x, velocity_y):
        pos_y = agent_position[0] - velocity_y
        pos_x = agent_position[1] + velocity_x
        return pos_x >= self.shape[1] or pos_y >= self.shape[0] \
               or pos_x < 0 or pos_y < 0 or (agent_position[0] - velocity_y,
                                             agent_position[1] + velocity_x) in tuple(
            map(tuple, np.argwhere(self.map == self.OUTSIDE_MAP)))


class RT1(Map):

    def _build_environment(self):
        m = np.vstack((
            self._create_row(3, 14, 0), self._create_row(2, 15, 0), self._create_row(2, 15, 0),
            self._create_row(1, 16, 0), self._create_row(0, 17, 0), self._create_row(0, 17, 0),
            self._create_row(0, 10, 7), self._create_row(0, 9, 8), self._create_row(0, 9, 8),
            self._create_row(0, 9, 8), self._create_row(0, 9, 8), self._create_row(0, 9, 8),
            self._create_row(0, 9, 8), self._create_row(0, 9, 8), self._create_row(1, 8, 8),
            self._create_row(1, 8, 8), self._create_row(1, 8, 8), self._create_row(1, 8, 8),
            self._create_row(1, 8, 8), self._create_row(1, 8, 8), self._create_row(1, 8, 8),
            self._create_row(1, 8, 8), self._create_row(2, 7, 8), self._create_row(2, 7, 8),
            self._create_row(2, 7, 8), self._create_row(2, 7, 8), self._create_row(2, 7, 8),
            self._create_row(2, 7, 8), self._create_row(2, 7, 8), self._create_row(3, 6, 8),
            self._create_row(3, 6, 8), self._create_row(3, 6, 8),
        ))

        # add finish states
        m[0:6, 16] = self.TERMINAL_STATE
        # add start states
        m[31, 3:9] = self.START_STATE

        return m


class RT2(Map):
    # 30 , 33

    def _build_environment(self):
        m = np.vstack(tuple(reversed((
            self._create_row(0, 24, 9), self._create_row(0, 24, 9), self._create_row(0, 24, 9),
            self._create_row(1, 23, 9), self._create_row(2, 22, 9), self._create_row(3, 21, 9),
            self._create_row(4, 20, 9), self._create_row(5, 19, 9), self._create_row(6, 18, 9),
            self._create_row(7, 17, 9), self._create_row(8, 16, 9), self._create_row(9, 15, 9),
            self._create_row(10, 14, 9), self._create_row(11, 13, 9), self._create_row(12, 12, 9),
            self._create_row(13, 11, 9), self._create_row(14, 10, 9), self._create_row(15, 9, 9),
            self._create_row(15, 10, 8), self._create_row(15, 12, 6), self._create_row(15, 13, 5),
            self._create_row(15, 16, 2), self._create_row(14, 19, 0), self._create_row(13, 20, 0),
            self._create_row(12, 21, 0), self._create_row(11, 22, 0), self._create_row(11, 22, 0),
            self._create_row(11, 22, 0), self._create_row(11, 22, 0), self._create_row(12, 21, 0),
            self._create_row(13, 20, 0), self._create_row(17, 16, 0),
        ))))

        # add finish states
        m[0:10, 32] = self.TERMINAL_STATE
        # add start states
        m[31, 0:24] = self.START_STATE

        return m


class State:
    def __init__(self, position, velocity_x, velocity_y):
        self.position = position
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def __eq__(self, other):
        return np.all(self.position == other.position) \
               and self.velocity_x == other.velocity_x \
               and self.velocity_y == other.velocity_y

    def __hash__(self) -> int:
        return hash(tuple(self.position)) + hash((self.velocity_x, self.velocity_y))

    def __str__(self):
        return f"Position {self.position} Velocity ({self.velocity_y},{self.velocity_x})"


class StateAction:
    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __hash__(self):
        return self.state.__hash__() + hash(self.action)

    def state_eq(self, state):
        return self.state == state

    def action_eq(self, action):
        return self.action == action


class StateActionReward:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward


class Episode:
    def __init__(self):
        self.state_action_reward = []

    def add(self, val):
        self.state_action_reward.append(val)


class Environment:
    MAX_VELOCITY = 4
    MIN_VELOCITY = 0
    ACTION_SPACE = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1)]

    def __init__(self, m, noise):
        self.map = m
        self.velocity_x = 0
        self.velocity_y = 0
        self.agent_position = None
        self.current_episode = None
        self.terminated = False
        self.noise = noise
        self.boundary_crossed = False

        self._place_agent_start_pos()
        self._init_episode()

    def _init_episode(self):
        self.current_episode = Episode()

    def _check_velocities(self):
        if self.velocity_x < self.MIN_VELOCITY:
            self.velocity_x = self.MIN_VELOCITY
        if self.velocity_y < self.MIN_VELOCITY:
            self.velocity_y = self.MIN_VELOCITY

        if self.velocity_x > self.MAX_VELOCITY:
            self.velocity_x = self.MAX_VELOCITY
        if self.velocity_y > self.MAX_VELOCITY:
            self.velocity_y = self.MAX_VELOCITY

    def _agent_movement(self):
        if self.map.crossed_track_boundary(agent_position=self.agent_position,
                                           velocity_x=self.velocity_x,
                                           velocity_y=self.velocity_y):
            self._place_agent_start_pos()
            self.velocity_x = 0
            self.velocity_y = 0
            self.boundary_crossed = True

        elif self.map.crossed_finish_line(agent_position=self.agent_position,
                                          velocity_x=self.velocity_x,
                                          velocity_y=self.velocity_y):
            self.terminated = True

        else:
            self.agent_position \
                = tuple([self.agent_position[0] - self.velocity_y, self.agent_position[1] + self.velocity_x])

            assert self.agent_position in self.map.get_position_space()

    def action(self, a):
        assert not self.terminated, "Environment is terminated. Reinitialize first."

        if np.random.uniform(low=0.0, high=1.0) > self.noise:
            # velocities left unchanged regardless of action taken w. prob. 0.1
            self.velocity_x += a[0]
            self.velocity_y += a[1]
        self._check_velocities()

        self.current_episode.add(StateActionReward(
            state=State(position=self.agent_position, velocity_x=self.velocity_x, velocity_y=self.velocity_y),
            # use adapted velocities but original action value
            reward=self._compute_reward(),
            action=a))

        self._agent_movement()

        return self._compute_reward()

    def _compute_reward(self):
        if self.map.crossed_finish_line(agent_position=self.agent_position,
                                        velocity_x=self.velocity_x,
                                        velocity_y=self.velocity_y):
            return 0
        else:
            return -1

    def set_env_state(self, state):
        assert state.position in self.map.get_position_space(), "Invalid position passed."
        assert state.position not in self.map.get_terminal_space(), "Initialized with terminal state."
        self.velocity_x = state.velocity_x
        self.velocity_y = state.velocity_y
        self.agent_position = state.position

    def reinitialize_env(self):
        self.velocity_x = 0
        self.velocity_y = 0
        self._place_agent_start_pos()
        self._init_episode()
        self.terminated = False

    def _place_agent_start_pos(self):
        self.agent_position = self.map.get_random_start_pos()

    def get_state_space(self):
        state_space = []
        for position in self.map.get_position_space():
            for v_x in range(self.MIN_VELOCITY, self.MAX_VELOCITY + 1):
                for v_y in range(self.MIN_VELOCITY, self.MAX_VELOCITY + 1):
                    state_space.append(State(position=position, velocity_x=v_x, velocity_y=v_y))
        return state_space

    def get_state_action_space(self):
        return_dict = {s: [] for s in self.get_state_space()}
        for s in return_dict.keys():
            for a in self.ACTION_SPACE:
                return_dict[s].append(StateAction(s, a))
        return return_dict

    def get_random_action(self):
        return random.choice(self.ACTION_SPACE)

    def get_current_state(self):
        return State(self.agent_position, velocity_x=self.velocity_x, velocity_y=self.velocity_y)


def print_policy(env: Environment, start_state: State, policy):
    m = copy.deepcopy(env.map)
    e = copy.deepcopy(env)

    current_state = start_state
    e.set_env_state(current_state)
    while current_state.position in env.map.get_position_space() and not e.terminated and not e.boundary_crossed:
        m.map[current_state.position[0], current_state.position[1]] = m.AGENT
        e.action(max(policy[current_state], key=lambda key: policy[current_state][key]))
        old_state = current_state
        current_state = e.get_current_state()
        if old_state == current_state:
            break
    m.plot_map()


class Agent:

    def __init__(self, epsilon, alpha, gamma, env):
        self.Qs = {}
        self.epsilon = epsilon
        self.env = env
        self.alpha = alpha
        self.gamma = gamma

        for st in env.get_state_space():
            self.Qs[st] = {}
            for a in env.ACTION_SPACE:
                self.Qs[st][a] = 0  # initialize all Qs to 0

    def act(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.ACTION_SPACE)
        else:
            return max(self.Qs[env.get_current_state()], key=self.Qs[env.get_current_state()].get)

    def update_qs(self, old_state, reward, new_state, a):
        self.Qs[old_state][a] = (1 - self.alpha) * self.Qs[old_state][a] \
                                + self.alpha * (reward + self.gamma * max(self.Qs[new_state].values()))



def play(env, agent, total_steps, print_final = True):
    reward_per_episode = []
    for trial in tqdm.tqdm(range(total_steps)):  # Run trials
        cumulative_reward = 0  # Initialise values of each game
        env.reinitialize_env()  # re-init environment
        while not env.terminated:  # Run until max steps or until game is finished
            old_state = env.get_current_state()
            a = agent.act()
            reward = env.action(a)
            assert type(a) == tuple
            new_state = env.get_current_state()
            assert new_state in agent.Qs.keys()
            agent.update_qs(old_state, reward, new_state, a)
            cumulative_reward += reward
        reward_per_episode.append(cumulative_reward)  # Append reward for current trial to performance log
    plt.plot(reward_per_episode)
    plt.show()

    if print_final:
        # print final policies from starting states
        for state_pos in env.map.get_start_space():
            st = State(position=state_pos, velocity_x=0, velocity_y=0)  # starting position
            et = Environment(env.map, noise=0.0)
            et.set_env_state(st)
            print_policy(et, st, agent.Qs)


racetrack_1 = RT1()
env = Environment(m=racetrack_1, noise=0.0)
env.map.plot_map()
agent = Agent(env=env, alpha=0.1, gamma=1.0, epsilon=0.001)
play(env, agent, 40000)

racetrack_2 = RT2()
env = Environment(m=racetrack_2, noise=0.0)
env.map.plot_map()
agent = Agent(env=env, alpha=0.1, gamma=1.0, epsilon=0.001)
play(env, agent, 40000)
