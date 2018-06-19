#!/usr/bin/env python
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from k_of_n_mdp_policy_opt.utils import load_pkl
from k_of_n_mdp_policy_opt.environments.gridworld import Gridworld


def plot(experiment, num_plot_columns=4):
    data = load_pkl(os.path.join(experiment, 'every_k'))

    num_rows = data['num_rows']
    num_columns = data['num_columns']
    num_actions = data['num_actions']
    source = data['source']
    goal = data['goal']
    goal_reward = data['goal_reward']

    uncertainty = []
    for r, c in data['unknown_reward_positions']:
        uncertainty.append(
            (r, c, r"$\sigma={}$".format(data['uncertainty_std']))
        )  # yapf:disable

    num_methods = len(data['methods'])
    num_plot_rows = int(math.ceil(num_methods / float(num_plot_columns))) + 1

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(
        num_plot_rows,
        num_plot_columns,
        int(math.ceil(num_plot_columns / 2.0))
    )  # yapf:disable
    Gridworld.Painter.draw(
        np.array(data['baseline']['Mean']['policy']).reshape(
            [num_rows, num_columns, num_actions]
        ),
        np.array(data['baseline']['Mean']['state_distribution']).reshape(
            [num_rows, num_columns]
        ),
        source=source,
        goal=(goal[0], goal[1], goal_reward),
        uncertainty=uncertainty,
        ax=ax
    )  # yapf:disable

    i = 0
    for k in sorted(data['methods'].keys(), key=lambda name: int(name[2:])):
        ax = fig.add_subplot(
            num_plot_rows,
            num_plot_columns,
            i + num_plot_columns + 1
        )  # yapf:disable
        Gridworld.Painter.draw(
            np.array(data['methods'][k]['policy']).reshape(
                [num_rows, num_columns, num_actions]
            ),
            np.array(data['methods'][k]['state_distribution']).reshape(
                [num_rows, num_columns]
            ),
            source=source,
            goal=(goal[0], goal[1], goal_reward),
            uncertainty=uncertainty,
            ax=ax
        )  # yapf:disable
        i += 1
    fig.savefig(
        os.path.join(experiment.path(), '{}.pdf'.format(experiment.name)),
        bbox_inches='tight'
    )  # yapf:disable


if __name__ == '__main__':
    plot(os.path.join(os.getcwd(), 'tmp', 'safe_rl_gridworld_k_of_n'))
