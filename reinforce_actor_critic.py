import gym
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Categorical

env = gym.make('CartPole-v1')
env.seed(1234)
torch.manual_seed(1234)


class Baseline(nn.Module):
    def __init__(self, in_features, out_size):
        super(Baseline, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_features, out_features=64, bias=False)
        self.tc_fc2 = nn.Linear(in_features=64, out_features=out_size, bias=False)

    def forward(self, x):
        return self.tc_fc2(F.dropout(F.gelu(self.tc_fc1(x)), 0.5)).squeeze()


class Policy(nn.Module):
    def __init__(self, in_features, out_size):
        super(Policy, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_features, out_features=64, bias=False)
        self.tc_fc2 = nn.Linear(in_features=64, out_features=out_size, bias=False)

    def forward(self, x):
        return F.softmax(self.tc_fc2(F.dropout(F.gelu(self.tc_fc1(x)), 0.5)), dim=0)


policy = Policy(in_features=4, out_size=env.action_space.n)
baseline = Baseline(in_features=4, out_size=1)
stats = {'TotalSteps': []}

opt = torch.optim.Adam(policy.parameters(), lr=0.1)
opt_critic = torch.optim.Adam(baseline.parameters(), lr=0.1)

run_rew = 0
iter = tqdm(range(50), postfix=f'Reward: {run_rew}')

solved = False
for episode in iter:

    old_params = policy.parameters()

    obs = torch.tensor(env.reset()).float()
    terminated = False
    ep_reward = 0
    i = 1
    step_cnt = 0

    while not terminated:
        step_cnt += 1
        a_probs = Categorical(policy(obs))
        a = a_probs.sample()

        new_obs, reward, terminated, _ = env.step(action=a.item())
        new_obs = torch.tensor(new_obs).float()
        reward = torch.tensor(reward)
        ep_reward += reward

        value_next = 0
        if not terminated:
            value_next = baseline(new_obs)
        td_target = reward + 0.99 * value_next
        state_val = baseline(obs)
        delta = td_target - state_val

        opt_critic.zero_grad()
        loss_base = torch.nn.functional.smooth_l1_loss(td_target, state_val)
        loss_base.backward()
        opt_critic.step()

        opt.zero_grad()
        loss_policy = i * delta.item() * -a_probs.log_prob(a)
        loss_policy.backward()
        opt.step()

        run_rew = 0.05 * ep_reward + 0.95 * run_rew
        i *= 0.99
        obs = new_obs

        if ep_reward > 480 and not solved:
            print('Solved!')
            solved = True

    iter.set_postfix({"Reward": run_rew})
    stats['TotalSteps'].append(step_cnt)

df = pd.DataFrame(stats)
df.TotalSteps.plot()
plt.show()
