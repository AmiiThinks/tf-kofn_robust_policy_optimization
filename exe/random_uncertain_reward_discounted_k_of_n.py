#!/usr/bin/env python
import tensorflow as tf
import os
import math
import numpy as np
from amii_tf_mdp.robust.k_of_n import k_of_n_mdp_weights
from amii_tf_mdp.discounted_mdp import generalized_policy_iteration_op, \
    inst_regrets_op, associated_ops, value_ops
from amii_tf_mdp.utils.timer import Timer
from amii_tf_mdp.utils.quadrature import midpoint_quadrature
from amii_tf_mdp.utils.experiment import PickleExperiment
from amii_tf_mdp.utils.random import reset_random_state
from amii_tf_mdp.utils.tensor import row_normalize_op

random_seed = 10
eval_random_seed = 42

experiment = PickleExperiment(
    'random_uncertain_reward_discounted_k_of_n',
    root=os.path.join(os.getcwd(), 'tmp'),
    seed=random_seed,
    log_level=tf.logging.INFO)
experiment.ensure_present()

num_states = int(1e1)
num_actions = int(5)
gamma = 0.9
final_data = {
    'num_training_iterations': 1000,
    'num_eval_mdp_reps': 100,
    'n_weights': [0.0] * 999 + [1.0],
    'num_states': num_states,
    'num_actions': num_actions,
    'discount_factor': gamma,
    'methods': {}
}
num_sampled_mdps = len(final_data['n_weights'])

root_op = row_normalize_op(np.random.normal(size=[num_states, 1]))
P = row_normalize_op(
    np.random.uniform(size=[num_states * num_actions, num_states]))

known_reward_positions = np.random.randint(
    0, high=2, size=[num_states * num_actions, 1])
assert (known_reward_positions.max() == 1)
assert (known_reward_positions.min() == 0)

known_rewards = -500 * known_reward_positions

unknown_reward_positions = 1 - known_reward_positions

reward_models_op = tf.squeeze(
    tf.stack(
        [(tf.random_normal(
            stddev=1000, shape=[num_states * num_actions, 1]) *
          unknown_reward_positions) + known_rewards
         for _ in range(num_sampled_mdps)],
        axis=1))
assert (reward_models_op.shape[0] == num_states * num_actions)
assert (reward_models_op.shape[1] == num_sampled_mdps)
assert (len(reward_models_op.shape) == 2)

avg_reward_model = known_rewards


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


class KofnConfig(object):
    def __init__(self, i, n_weights):
        self.n_weights = n_weights
        self.k = i + 1
        self.k_weights = [0.0] * self.num_sampled_mdps()
        self.k_weights[i] = 1.0

    def num_sampled_mdps(self):
        return len(self.n_weights)

    def mdp_weights_op(self, evs_op):
        return tf.expand_dims(
            k_of_n_mdp_weights(self.n_weights, self.k_weights,
                               tf.squeeze(evs_op)),
            axis=1)

    def name(self):
        return 'k={}'.format(self.k)


class UncertainRewardKOfNMethod(object):
    def __init__(self, config, root_op, transition_model_op, reward_models_op,
                 gamma):
        if reward_models_op.shape[1].value != config.num_sampled_mdps():
            print(reward_models_op.shape[1].value)
            print(config.num_sampled_mdps())
        assert reward_models_op.shape[1].value == config.num_sampled_mdps()
        self.config = config
        self.root_op = root_op
        self.transition_model_op = transition_model_op
        self.reward_models_op = reward_models_op

        self.q_regrets_op = tf.Variable(
            tf.zeros([self.num_states(), self.num_actions()]))

        self.Pi, self.action_values_op, self.state_values_op, self.evs_op = (
            associated_ops(
                self.q_regrets_op,
                root_op,
                transition_model_op,
                reward_models_op,
                gamma=gamma))

        assert (len(self.evs_op.shape) == 2)
        assert (self.evs_op.shape[0].value == reward_models_op.shape[1].value)
        assert (self.evs_op.shape[1].value == 1)

        self.mdp_weights_op = self.config.mdp_weights_op(self.evs_op)

        self.ev_op = tf.squeeze(
            tf.transpose(self.evs_op) @ self.mdp_weights_op)

        self.max_num_training_pe_iterations = tf.placeholder(tf.int32)
        (self.training_action_values_op, self.training_state_values_op,
         self.training_evs_op) = value_ops(
             self.Pi,
             root_op,
             transition_model_op,
             reward_models_op,
             gamma=gamma,
             max_num_iterations=self.max_num_training_pe_iterations)

        self.training_mdp_weights_op = self.config.mdp_weights_op(
            self.training_evs_op)

        self.r_s_op = tf.reshape(
            inst_regrets_op(self.training_action_values_op, Pi=self.Pi)
            @ self.training_mdp_weights_op,
            shape=self.q_regrets_op.shape)

        # next_q_regrets = tf.maximum(0.0, self.q_regrets_op + self.r_s_op)
        next_q_regrets = self.q_regrets_op + self.r_s_op
        self.update_op = tf.assign(self.q_regrets_op, next_q_regrets)

    def num_states(self):
        return self.root_op.shape[0].value

    def num_actions(self):
        return int(self.transition_model_op.shape[0].value / self.num_states())

    def name(self):
        return self.config.name()


def train_and_save_k_of_n(*methods):
    method_data = final_data['methods']

    all_ev_ops = [m.ev_op for m in methods]
    all_update_ops = [m.update_op for m in methods]
    all_n_evs_ops = [tf.squeeze(m.evs_op) for m in methods]
    all_max_iteration_placeholders = {
        m.max_num_training_pe_iterations: 2
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
                    math.ceil(math.log(t + 1) + 2))
            sess.run(
                all_update_ops,
                feed_dict=all_max_iteration_placeholders)
    print('')
    regret_update_timer.log_duration_s()
    print('')

    all_training_evs = np.array(all_training_evs)

    # TODO
    # if 'num_sequences' not in final_data:
    #     final_data['num_sequences'] = (
    #         q_regrets_op.shape[0].value * q_regrets_op.shape[1].value)
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

            # all_n_evs = np.array(all_n_evs)
            # print('{}      {}'.format('Worst', 'Best'))
            # for i in range(len(methods)):
            #     my_n_evs = all_n_evs[i, :]
            #     print('{}: {}      {}'.format(methods[i].name(),
            #                                   my_n_evs.min(), my_n_evs.max()))
            # print('')
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
                }
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
                  avg_reward_model,
                  gamma=0.9):
    name = 'Mean'
    print('# {}'.format(name))

    final_data['baseline'] = {}
    final_data['baseline'][name] = {'name': name}
    d = final_data['baseline'][name]

    policy_op = generalized_policy_iteration_op(
        transition_model_op,
        avg_reward_model,
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
        normalize_policy=False)

    evaluation_evs, evaluation_duration_s, area = eval_policy(baseline_evs_op)

    training_action_values_op, _, training_ev_op = value_ops(
        Pi, root_op, transition_model_op, avg_reward_model, gamma=gamma)
    training_ev_op = tf.squeeze(training_ev_op)

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
            root_op, P, reward_models_op, avg_reward_model, gamma=gamma)

        k_of_n_methods = []
        for i in [0] + list(range(99, num_sampled_mdps, 100)):
        # for i in range(num_sampled_mdps):
            config = KofnConfig(i, final_data['n_weights'])
            k_of_n_methods.append(
                UncertainRewardKOfNMethod(config, root_op, P, reward_models_op,
                                          gamma))
        sess.run(tf.global_variables_initializer())
        train_and_save_k_of_n(*k_of_n_methods)
        experiment.save(final_data, 'every_k')
