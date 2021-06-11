import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions import Categorical

env = gym.make('CartPole-v1')


class Baseline(nn.Module):
    def __init__(self, in_features, out_size):
        super(Baseline, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_features, out_features=128, bias=False)
        self.tc_fc2 = nn.Linear(in_features=128, out_features=out_size, bias=False)

    def forward(self, x):
        return self.tc_fc2(F.dropout(F.leaky_relu(self.tc_fc1(x), 0.5)))


class Policy(nn.Module):
    def __init__(self, in_features, out_size):
        super(Policy, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_features, out_features=128, bias=False)
        self.tc_fc2 = nn.Linear(in_features=128, out_features=out_size, bias=False)

    def forward(self, x):
        return F.softmax(self.tc_fc2(F.dropout(F.leaky_relu(self.tc_fc1(x), 0.5))), dim=0)


def update(opt, train_data):
    gts = []
    log_prob = []
    policy_loss = []
    baseline_values = []
    baseline_loss = []
    baseline_loss_values = []
    R = 0

    # create G
    for td in reversed(train_data):
        R = td[0] + 0.99 * R
        gts.insert(0, R)
        log_prob.insert(0, td[1])
        baseline_values.insert(0, td[2])

    gts = torch.tensor(gts).float()
    gts = gts - (gts.mean()) / (gts.std() + np.finfo(np.float32).eps.item())

    # calculate deltas and update value estimator
    for base_val, rew in zip(baseline_values, gts):
        baseline_loss.append(base_criterion(base_val.squeeze(), rew))
        baseline_loss_values.append(torch.abs(base_val - rew).detach().numpy())
    baseline_loss_tensor = torch.stack(baseline_loss)
    opt_base.zero_grad()
    loss_base = torch.sum(baseline_loss_tensor)
    loss_base.backward()
    opt_base.step()

    # update policy
    for lp, bl in zip(log_prob, baseline_loss_values):
        policy_loss.append(-lp * bl.item())
    policy_loss = torch.stack(policy_loss)
    opt.zero_grad()
    loss = torch.sum(policy_loss)
    loss.backward()
    opt.step()


policy = Policy(in_features=4, out_size=env.action_space.n)
baseline = Baseline(in_features=4, out_size=1)
base_criterion = torch.nn.L1Loss()
stats = {'TotalSteps': []}

# lower learning rate beneficial for convergence
opt = torch.optim.Adam(policy.parameters(), lr=0.01)
opt_base = torch.optim.Adam(baseline.parameters(), lr=0.001)

avg_rew = 0
iter = tqdm(range(300), postfix=f'Reward: {avg_rew}')
for episode in iter:
    old_params = policy.parameters()
    obs = torch.tensor(env.reset()).float()  # returns the initial observation (agent start state)
    terminated = False
    train_data = list()  # contains the training instances of the current episode
    ep_reward = 0
    # run through the episode
    while not terminated:
        a_probs = Categorical(policy(obs))
        base_val = baseline(obs)
        a = a_probs.sample()

        # take action
        obs, reward, terminated, _ = env.step(action=a.item())
        obs = torch.tensor(obs).float()
        ep_reward += reward

        train_data.append((reward, a_probs.log_prob(a), base_val))
        avg_rew = 0.05 * ep_reward + 0.95 * avg_rew

    iter.set_postfix({"Reward": avg_rew})
    update(train_data=train_data, opt=opt)
    assert (policy.parameters() != old_params)
    stats['TotalSteps'].append(len(train_data))

df = pd.DataFrame(stats)
df.TotalSteps.plot()
plt.show()
