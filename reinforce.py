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


class Policy(nn.Module):
    def __init__(self, in_features, out_size):
        super(Policy, self).__init__()
        self.tc_fc1 = nn.Linear(in_features=in_features, out_features=128, bias=False)
        self.tc_fc2 = nn.Linear(in_features=128, out_features=out_size, bias=False)

    def forward(self, x):
        return F.softmax(self.tc_fc2(F.dropout(F.leaky_relu(self.tc_fc1(x), 0.6))), dim=0)


def update(opt, train_data):
    gts = []
    log_prob = []
    policy_loss = []
    R = 0
    for td in reversed(train_data):
        R = td[0] + 0.99 * R
        gts.insert(0, R)
        log_prob.insert(0, td[1])
    gts = torch.tensor(gts).float()
    gts = (gts - gts.mean()) / (gts.std() + np.finfo(np.float32).eps.item())
    for lp, rew in zip(log_prob, gts):
        policy_loss.append(-lp * rew)
    policy_loss = torch.stack(policy_loss)
    opt.zero_grad()
    loss = torch.sum(policy_loss)
    loss.backward()
    opt.step()
    lr_scheduler.step()


policy = Policy(in_features=4, out_size=env.action_space.n)
stats = {'TotalSteps': []}  # dictionary holding the data for each episode

opt = torch.optim.Adam(policy.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.95)
run_rew = 0
iter = tqdm(range(1000), postfix=f'Reward: {run_rew}')
for episode in iter:
    old_params = policy.parameters()
    obs = torch.tensor(env.reset()).float()
    terminated = False
    train_data = list()
    ep_reward = 0

    # run through the episode
    while not terminated:
        a_probs = Categorical(policy(obs))
        a = a_probs.sample()

        # take action
        obs, reward, terminated, _ = env.step(action=a.item())
        obs = torch.tensor(obs).float()
        ep_reward += reward

        train_data.append((reward, a_probs.log_prob(a)))
        run_rew = 0.05 * ep_reward + 0.95 * run_rew

    iter.set_postfix({"Reward": run_rew})
    update(train_data=train_data, opt=opt)
    assert (policy.parameters() != old_params)
    stats['TotalSteps'].append(len(train_data))

df = pd.DataFrame(stats)
df.TotalSteps.plot()
plt.show()
