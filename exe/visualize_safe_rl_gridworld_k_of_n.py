#!/usr/bin/env python
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import numpy as np
from k_of_n_mdp_policy_opt.utils.experiment import PickleExperiment
from k_of_n_mdp_policy_opt.environments.gridworld import Gridworld


def plot(experiment, num_plot_columns=4):
    data = experiment.load('every_k')

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
        )

    num_methods = len(data['methods'])
    num_plot_rows = int(math.ceil(num_methods / float(num_plot_columns))) + 1

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(
        num_plot_rows,
        num_plot_columns,
        int(math.ceil(num_plot_columns / 2.0))
    )
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
    )

    i = 0
    for k in sorted(data['methods'].keys(), key=lambda name: int(name[2:])):
        ax = fig.add_subplot(
            num_plot_rows,
            num_plot_columns,
            i + num_plot_columns + 1
        )
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
        )
        i += 1
    fig.savefig(
        os.path.join(experiment.path(), '{}.pdf'.format(experiment.name)),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    random_seed = 10
    experiment = PickleExperiment(
        'safe_rl_gridworld_k_of_n',
        root=os.path.join(os.getcwd(), 'tmp'),
        seed=random_seed,
        log_level=tf.logging.INFO)
    plot(experiment)
