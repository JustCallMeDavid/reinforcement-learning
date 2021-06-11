import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

runs = 1000
steps = 1000
eps_is = [0.0, 0.1, 0.01]  # random choice factor
result_list = []

for eps in eps_is:
    for run in range(runs):
        # initialize vars
        np.random.seed(2 * 3 * 7 * run)
        a_is = list(np.random.normal(loc=0, scale=1, size=10))
        n_ais = [0] * len(a_is)
        approx_ais = [0] * len(a_is)
        r_is = [np.random.normal(loc=a_i, scale=1, size=1000) for a_i in a_is]

        for r in range(steps):
            action_idx = None

            if np.random.uniform(low=0.0, high=1.0, size=1)[0] > eps:
                # normal choice, highest value action taken
                action_idx = np.argmax(approx_ais)
            else:
                # random choice
                action_idx = np.random.randint(low=0, high=len(a_is), size=1)[0]

            rew = r_is[action_idx][r]
            n_ais[action_idx] += 1
            approx_ais[action_idx] = approx_ais[action_idx] + 1 / n_ais[action_idx] * (rew - approx_ais[action_idx])
            result_list.append({'eps': eps, 'run': run, "step": r, "action_chosen": action_idx,
                                "best_action": np.argmax(np.array(a_is)),
                                "reward": rew})

result_frame = pd.DataFrame(result_list)
result_frame.eps = result_frame.eps.astype('category')

# average reward
avg_reward = result_frame.groupby(
    by=['eps', 'step']).reward.mean().reset_index()  # compute the mean reward over the different runs
sb.lineplot(data=avg_reward, x='step', hue='eps', y='reward')
plt.show()

# average optimal
result_frame['correct'] = result_frame.action_chosen == result_frame.best_action
avg_correct = result_frame.groupby(
    by=['eps', 'step']).correct.mean().reset_index()  # compute the mean correct score over the different runs
sb.lineplot(data=avg_correct, x='step', hue='eps', y='correct')
plt.show()
