#!/usr/bin/env python
import tensorflow as tf
import os
import math
import numpy as np
from amii_tf_mdp.robust.k_of_n import DeterministicKofnConfig
from amii_tf_mdp.discounted_mdp import generalized_policy_iteration_op, \
    associated_ops, value_ops
from amii_tf_mdp.utils.timer import Timer
from amii_tf_mdp.utils.quadrature import midpoint_quadrature
from amii_tf_mdp.utils.experiment import PickleExperiment
from amii_tf_mdp.utils.random import reset_random_state
from amii_tf_mdp.robust.uncertain_reward_discounted_continuing_kofn import \
    UncertainRewardDiscountedContinuingKofn
from amii_tf_mdp.environments.gridworld import Gridworld


random_seed = 10
eval_random_seed = 42

experiment = PickleExperiment(
    'safe_rl_gridworld_k_of_n',
    root=os.path.join(os.getcwd(), 'tmp'),
    seed=random_seed,
    log_level=tf.logging.INFO)
experiment.ensure_present()

num_rows = 3
num_columns = 10
gridworld = Gridworld(num_rows, num_columns)

num_actions = gridworld.num_cardinal_directions()
num_states = num_rows * num_columns
gamma = 0.9

final_data = {
    'num_training_iterations': 500,
    'num_eval_mdp_reps': 1,
    'n': 1000,
    'num_states': num_states,
    'num_actions': num_actions,
    'discount_factor': gamma,
    'methods': {}
}
num_sampled_mdps = final_data['n']

root_op = tf.cast(
    gridworld.indicator_state_op(num_rows - 1, num_columns - 1),
    tf.float32
)
goal = (0, num_columns - 1)
P = tf.reshape(
    gridworld.cardinal_transition_model_op(sink=goal),
    (num_states * num_actions, num_states)
)

unknown_reward_positions = [(1, c) for c in range(1, num_columns)]
unknown_reward_means = [0.0 for _ in range(1, num_columns)]

known_rewards_op = (
    gridworld.cardinal_reward_model_op(
        unknown_reward_positions,
        unknown_reward_means,
        sink=goal
    ) + gridworld.cardinal_reward_model_op(
        [goal],
        [0.1],
        sink=goal
    )
)

reward_models = []
for _ in range(num_sampled_mdps):
    reward_models.append(
        known_rewards_op + gridworld.cardinal_reward_model_op(
            unknown_reward_positions,
            [
                tf.random_normal(stddev=0.1, shape=[])
                for _ in unknown_reward_positions
            ],
            sink=goal
        )
    )

reward_models_op = tf.squeeze(tf.stack(reward_models, axis=1))
assert (reward_models_op.shape[0] == num_states * num_actions)
assert (reward_models_op.shape[1] == num_sampled_mdps)
assert (len(reward_models_op.shape) == 2)


def eval_policy(evs):
    reset_random_state(eval_random_seed)
    eval_timer = Timer('Evaluation')
    evaluation_evs = []
    evs = tf.squeeze(evs)
    with eval_timer:
        for i in range(final_data['num_eval_mdp_reps']):
            evaluation_evs.append(sess.run(evs))
        evaluation_evs = np.array(evaluation_evs)
        evaluation_evs = evaluation_evs.reshape(
            [final_data['num_eval_mdp_reps'] * num_sampled_mdps])
        evaluation_evs.sort()
        area = midpoint_quadrature(evaluation_evs, (0, 100))
    if len(evaluation_evs) > 0:
        print('# Bottom: ')
        print(evaluation_evs[:5])
        print('# Top: ')
        print(evaluation_evs[-5:])
        print('# Area: {}'.format(area))
    eval_timer.log_duration_s()
    print('')
    print('')
    return evaluation_evs, eval_timer.duration_s(), area


def train_and_save_k_of_n(*methods):
    method_data = final_data['methods']

    all_ev_ops = [m.ev_op for m in methods]
    all_update_ops = [m.update_op for m in methods]
    all_n_evs_ops = [tf.squeeze(m.evs_op) for m in methods]
    all_max_iteration_placeholders = {
        m.max_num_training_pe_iterations: int(1e1)
        for m in methods
    }

    training_checkpoints = []
    all_training_evs = []

    experiment.reset_random_state()
    regret_update_timer = Timer('Regret Update')
    with regret_update_timer:
        for t in range(final_data['num_training_iterations']):
            if t % 10 == 0:
                training_checkpoints.append(t)
                gadget_evs, all_n_evs = sess.run([all_ev_ops, all_n_evs_ops])
                all_training_evs.append(gadget_evs)
                all_n_evs = np.array(all_n_evs)
                assert all_n_evs.shape[0] == len(methods)
                assert all_n_evs.shape[1] == num_sampled_mdps

                print('{}       {}      {}      {}'.format(
                    t, 'Worst', 'Best', 'Gadget'))
                for i in range(len(methods)):
                    my_n_evs = all_n_evs[i, :]
                    print('{}: {}       {}      {}'.format(
                        methods[i].name(), my_n_evs.min(), my_n_evs.max(),
                        gadget_evs[i]))
                print('')
            for k in all_max_iteration_placeholders.keys():
                all_max_iteration_placeholders[k] = int(
                    max(all_max_iteration_placeholders[k], t))
            sess.run(all_update_ops, feed_dict=all_max_iteration_placeholders)
    print('')
    regret_update_timer.log_duration_s()
    print('')

    all_training_evs = np.array(all_training_evs)
    for i in range(len(methods)):
        method_data[methods[i].name()] = {
            'training': {
                'checkpoints': training_checkpoints,
                'evs': all_training_evs[:, i],
                'duration_s': regret_update_timer.duration_s() / len(methods)
            }
        }

    reset_random_state(eval_random_seed)
    eval_timer = Timer('Evaluation')

    evaluation_evs = []
    with eval_timer:
        for i in range(final_data['num_eval_mdp_reps']):
            all_n_evs = sess.run(all_n_evs_ops)
            evaluation_evs.append(all_n_evs)
        evaluation_evs = np.array(evaluation_evs)
        evaluation_evs = np.moveaxis(evaluation_evs, 1, -1).reshape(
            [final_data['num_eval_mdp_reps'] * num_sampled_mdps,
             len(methods)])
        evaluation_evs.sort(axis=0)

        for i in range(len(methods)):
            evs = evaluation_evs[:, i]
            area = midpoint_quadrature(evs, (0, 100))

            method_data[methods[i].name()] = {
                'evaluation': {
                    'evs': evs,
                    'duration_s': eval_timer.duration_s() / len(methods),
                    'midpoint_area': area
                },
                'policy': sess.run(methods[i].policy_op),
                'steady_state_distribution': (
                    methods[i].steady_state_distribution(sess)
                )
            }
            if len(evs) > 0:
                print('# {}'.format(methods[i].name()))
                print('## Bottom: ')
                print(evs[:5])
                print('## Top: ')
                print(evs[-5:])
                print('## Area: {}'.format(area))
                print('')
    eval_timer.log_duration_s()


def eval_baseline(root_op,
                  transition_model_op,
                  reward_models_op,
                  known_rewards_op,
                  gamma=0.9):
    name = 'Mean'
    print('# {}'.format(name))

    final_data['baseline'] = {}
    final_data['baseline'][name] = {'name': name}
    d = final_data['baseline'][name]

    policy_op = generalized_policy_iteration_op(
        transition_model_op,
        known_rewards_op,
        gamma=gamma,
        t=10,
        pi_threshold=1e-15,
        max_num_pe_iterations=lambda s: s + 2)

    Pi, eval_action_values_op, _, baseline_evs_op = associated_ops(
        policy_op,
        root_op,
        transition_model_op,
        reward_models_op,
        gamma=gamma,
        normalize_policy=False,
        max_num_iterations=int(1e3))

    evaluation_evs, evaluation_duration_s, area = eval_policy(baseline_evs_op)

    training_action_values_op, _, training_ev_op = value_ops(
        Pi, root_op, transition_model_op, known_rewards_op, gamma=gamma)
    training_ev_op = tf.squeeze(training_ev_op)

    d['training'] = {'ev': sess.run(training_ev_op)}
    d['evaluation'] = {
        'evs': evaluation_evs,
        'duration_s': evaluation_duration_s,
        'midpoint_area': area
    },
    d['policy'] = sess.run(policy_op)


# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        eval_baseline(
            root_op, P, reward_models_op, known_rewards_op, gamma=gamma)

        k_of_n_methods = []
        for i in [0] + list(range(99, final_data['n'], 100)):
            config = DeterministicKofnConfig(i + 1, final_data['n'])
            k_of_n_methods.append(
                UncertainRewardDiscountedContinuingKofn(
                    config, root_op, P, reward_models_op, gamma))
        sess.run(tf.global_variables_initializer())
        train_and_save_k_of_n(*k_of_n_methods)
        experiment.save(final_data, 'every_k')
