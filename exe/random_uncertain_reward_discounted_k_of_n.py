#!/usr/bin/env python
import tensorflow as tf
import sys
import os
import numpy as np
from amii_tf_mdp.robust.k_of_n import k_of_n_mdp_weights
from amii_tf_mdp.discounted_mdp import generalized_policy_iteration_op, \
    dual_action_value_policy_evaluation_op, root_value_op, inst_regrets_op
from amii_tf_mdp.utils.timer import Timer
from amii_tf_mdp.utils.quadrature import midpoint_quadrature
from amii_tf_mdp.utils.experiment import PickleExperiment
from amii_tf_mdp.utils.random import reset_random_state
from amii_tf_mdp.utils.tensor import row_normalize_op, block_ones, \
    matrix_to_block_matrix_op

random_seed = 10
eval_random_seed = 42

experiment = PickleExperiment(
    'random_uncertain_reward_discounted_k_of_n',
    root=os.path.join(os.getcwd(), 'tmp'),
    seed=random_seed,
    log_level=tf.logging.INFO)
experiment.ensure_present()

num_states = int(1e1)
num_actions = int(1e1)
gamma = 0.9
final_data = {
    'num_training_iterations': 200,
    'num_eval_mdp_reps': 10,
    'n_weights': [0.0] * 999 + [1.0],
    'num_states': num_states,
    'num_actions': num_actions,
    'discount_factor': gamma,
    'methods': {}
}
num_sampled_mdps = len(final_data['n_weights'])

root_op = row_normalize_op(np.random.normal(size=[num_states, 1]))
P = row_normalize_op(
    np.random.normal(size=[num_states * num_actions, num_states]))

known_reward_positions_op = np.random.randint(
    0, high=2, size=[num_states * num_actions, 1])
assert (known_reward_positions_op.max() == 1)
assert (known_reward_positions_op.min() == 0)

known_rewards_op = (np.random.normal(size=[num_states * num_actions, 1]) *
                    known_reward_positions_op)

unknown_reward_positions_op = 1 - known_reward_positions_op

reward_models_op = tf.squeeze(
    tf.stack(
        [(tf.random_normal(stddev=1, shape=[num_states * num_actions, 1]) *
          unknown_reward_positions_op) + known_rewards_op
         for _ in range(num_sampled_mdps)],
        axis=1))
avg_reward_model_op = known_rewards_op
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
        evaluation_evs.resize(
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


# TODO Maybe move this into discounted_mdp
def matrix_inst_regrets_op(policy, Q):
    num_states = policy.shape[0].value
    num_actions = policy.shape[1].value
    V = matrix_to_block_matrix_op(policy) @ Q
    ev = tf.transpose(block_ones(num_states, num_actions)) @ V
    return Q - ev


def train_and_save_k_of_n(i,
                          root_op,
                          transition_model_op,
                          reward_models_op,
                          gamma=0.9):
    name = 'k={}'.format(i + 1)
    print('# {}'.format(name))

    final_data['methods'][name] = {'name': name}
    d = final_data['methods'][name]

    d['k_weights'] = [0.0] * num_sampled_mdps
    d['k_weights'][i] = 1.0

    q_regrets = tf.Variable(tf.zeros([num_states, num_actions]))
    Pi = matrix_to_block_matrix_op(row_normalize_op(q_regrets))

    action_values_op = dual_action_value_policy_evaluation_op(
        transition_model_op,
        Pi,
        reward_models_op,
        gamma=gamma,
        threshold=1e-15,
        max_num_iterations=-1)

    evs = root_value_op(root_op, Pi @ action_values_op)
    assert (len(evs.shape) == 2)
    assert (evs.shape[0] == num_sampled_mdps)
    assert (evs.shape[1] == 1)

    mdp_weights = tf.expand_dims(
        k_of_n_mdp_weights(final_data['n_weights'], d['k_weights'],
                           tf.squeeze(evs)),
        axis=1)

    ev = tf.squeeze(tf.transpose(evs) @ mdp_weights)

    r_s_op = tf.reshape(
        inst_regrets_op(action_values_op, Pi=Pi) @ mdp_weights,
        shape=q_regrets.shape)

    regret_update = tf.assign(q_regrets, tf.maximum(0.0, q_regrets + r_s_op))

    experiment.reset_random_state()

    regret_update_timer = Timer('Regret Update')
    q_regrets.initializer.run()
    training_evs = {0: sess.run(ev)}
    with regret_update_timer:
        for t in range(final_data['num_training_iterations']):
            gadget_ev, _ = sess.run([ev, regret_update])
            training_evs[t + 1] = gadget_ev

            if t % 10 == 0:
                sys.stdout.write('\r{}    {}'.format(t, gadget_ev))
    print('')
    regret_update_timer.log_duration_s()
    print('')

    evaluation_evs, evaluation_duration_s, area = eval_policy(evs)

    if 'num_sequences' not in final_data:
        final_data['num_sequences'] = (
            q_regrets.shape[0].value * q_regrets.shape[1].value)
    d['training'] = {
        'evs': training_evs,
        'duration_s': regret_update_timer.duration_s()
    }
    d['evaluation'] = {
        'evs': evaluation_evs,
        'duration_s': evaluation_duration_s,
        'midpoint_area': area
    }


def eval_baseline(root_op,
                  transition_model_op,
                  reward_models_op,
                  avg_reward_model_op,
                  gamma=0.9):
    name = 'Mean'
    print('# {}'.format(name))

    final_data['baseline'] = {}
    final_data['baseline'][name] = {'name': name}
    d = final_data['baseline'][name]

    policy_op = generalized_policy_iteration_op(
        transition_model_op,
        avg_reward_model_op,
        gamma=gamma,
        t=10,
        pi_threshold=1e-15,
        max_num_pi_iterations=lambda s: s + 2)
    Pi = matrix_to_block_matrix_op(policy_op)

    eval_action_values_op = dual_action_value_policy_evaluation_op(
        transition_model_op,
        Pi,
        reward_models_op,
        gamma=gamma,
        threshold=1e-15,
        max_num_iterations=-1)
    baseline_evs_op = root_value_op(root_op, Pi @ eval_action_values_op)

    evaluation_evs, evaluation_duration_s, area = eval_policy(baseline_evs_op)

    training_action_values_op = dual_action_value_policy_evaluation_op(
        transition_model_op,
        Pi,
        avg_reward_model_op,
        gamma=gamma,
        threshold=1e-15,
        max_num_iterations=-1)
    training_ev_op = tf.squeeze(
        root_value_op(root_op, Pi @ training_action_values_op))

    d['training'] = {'ev': sess.run(training_ev_op)}
    d['evaluation'] = {
        'evs': evaluation_evs,
        'duration_s': evaluation_duration_s,
        'midpoint_area': area
    }


# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        eval_baseline(
            root_op, P, reward_models_op, avg_reward_model_op, gamma=gamma)
        for i in [0] + list(range(99, num_sampled_mdps, 100)):
            train_and_save_k_of_n(i, root_op, P, reward_models_op, gamma=gamma)
        experiment.save(final_data, 'every_k')
