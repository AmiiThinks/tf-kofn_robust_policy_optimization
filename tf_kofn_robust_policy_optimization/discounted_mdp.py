import tensorflow as tf
import numpy as np
from .utils.tensor import l1_projection_to_simplex, ind_max_op
from .utils.tensor import \
    matrix_to_block_matrix_op as policy_block_matrix_op


def state_action_successor_policy_evaluation_op(transitions,
                                                policy,
                                                gamma=0.9,
                                                threshold=1e-15,
                                                max_num_iterations=-1):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value
    num_actions = transitions.shape[1].value
    H_0 = tf.constant(
        1.0 / (num_states * num_actions),
        shape=(num_states * num_actions, num_states * num_actions))

    transitions = tf.tile(
        tf.expand_dims(transitions, axis=-1), [1, 1, 1, num_actions])
    state_action_to_state_action = tf.reshape(
        transitions * tf.expand_dims(tf.expand_dims(policy, axis=0), axis=0),
        H_0.shape)

    def H_dp1_op(H_d):
        return (((1 - gamma) * tf.eye(H_d.shape[0].value)) +
                (gamma * state_action_to_state_action @ H_d))

    def error_above_threshold(H_d, H_dp1):
        return tf.reduce_sum(tf.abs(H_dp1 - H_d)) > threshold

    def cond(d, H_d, H_dp1):
        error_op = error_above_threshold(H_d, H_dp1)
        return tf.squeeze(
            tf.logical_or(
                tf.logical_and(max_num_iterations < 1, error_op),
                tf.logical_and(tf.less(d, max_num_iterations), error_op)))

    return tf.while_loop(
        cond,
        lambda d, _, H_d: [d + 1, H_d, H_dp1_op(H_d)],
        [1, H_0, H_dp1_op(H_0)],
        parallel_iterations=1)[-1]


def dual_action_value_policy_evaluation_op(transitions,
                                           policy,
                                           r,
                                           gamma=0.9,
                                           threshold=1e-15,
                                           max_num_iterations=-1):
    policy = tf.convert_to_tensor(policy)
    r = tf.convert_to_tensor(r)
    return tf.reshape(
        tf.tensordot(
            state_action_successor_policy_evaluation_op(
                transitions,
                policy,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations),
            r,
            axes=[[1], [0]]) / (1.0 - gamma),
        policy.shape.concatenate(r.shape[2:]))


def primal_action_value_policy_evaluation_op(P,
                                             Pi,
                                             r,
                                             gamma=0.9,
                                             threshold=1e-15,
                                             max_num_iterations=-1):
    P = tf.convert_to_tensor(P)
    num_states = P.shape[1].value
    num_actions = int(P.shape[0].value / num_states)
    q_0 = tf.constant(
        np.random.normal(size=[num_states * num_actions, 1]),
        dtype=tf.float32,
        name='primal_action_value_policy_evaluation_op/q_0')

    def q_dp1_op(q_d):
        discounted_return = gamma * P @ Pi @ q_d
        return r + discounted_return

    def error_above_threshold(q_d, q_dp1):
        return tf.reduce_sum(tf.abs(q_dp1 - q_d)) > threshold

    def cond(d, q_d, q_dp1):
        error_op = error_above_threshold(q_d, q_dp1)
        return tf.squeeze(
            tf.logical_or(
                tf.logical_and(max_num_iterations < 1, error_op),
                tf.logical_and(tf.less(d, max_num_iterations), error_op)))

    return tf.while_loop(
        cond,
        lambda d, _, q_d: [d + 1, q_d, q_dp1_op(q_d)],
        [1, q_0, q_dp1_op(q_0)],
        parallel_iterations=1,
        name='primal_action_value_policy_evaluation_op/while_loop')[-1]


def generalized_policy_iteration_op(P,
                                    r,
                                    alpha=None,
                                    gamma=0.9,
                                    t=10,
                                    pi_threshold=1e-15,
                                    max_num_pe_iterations=lambda s: -1):
    if alpha is None: alpha = 1.0 / t
    num_states = P.shape[1].value
    num_actions = int(P.shape[0].value / num_states)
    assert (len(P.shape) == 2)
    assert (len(r.shape) == 2)

    def next_q(s, q):
        return tf.reshape(
            primal_action_value_policy_evaluation_op(
                P,
                policy_block_matrix_op(ind_max_op(q, axis=1)),
                r,
                gamma=gamma,
                threshold=pi_threshold,
                max_num_iterations=(max_num_pe_iterations(s))),
            shape=[num_states, num_actions])

    return ind_max_op(
        tf.while_loop(
            lambda s, _: s < t,
            lambda s, q: [s + 1, q + alpha * (next_q(s, q) - q)],
            [0, tf.zeros([num_states, num_actions])],
            parallel_iterations=1)[-1],
        axis=1)


def root_value_op(mu, v):
    return tf.transpose(tf.matmul(mu, v, transpose_a=True))


def value_ops(Pi, root_op, transition_model_op, reward_model_op, **kwargs):
    action_values_op = dual_action_value_policy_evaluation_op(
        transition_model_op, Pi, reward_model_op, **kwargs)

    state_values_op = Pi @ action_values_op
    ev_op = root_value_op(root_op, state_values_op)

    return action_values_op, state_values_op, ev_op


def associated_ops(action_weights,
                   root_op,
                   transition_model_op,
                   reward_model_op,
                   normalize_policy=True,
                   **kwargs):
    if normalize_policy:
        policy = l1_projection_to_simplex(action_weights, axis=1)
    else:
        policy = action_weights
    Pi = policy_block_matrix_op(policy)

    action_values_op, state_values_op, ev_op = value_ops(
        Pi, root_op, transition_model_op, reward_model_op, **kwargs)

    return Pi, action_values_op, state_values_op, ev_op


def state_successor_policy_evaluation_op(P,
                                         Pi,
                                         gamma=0.9,
                                         threshold=1e-15,
                                         max_num_iterations=-1):
    P = tf.convert_to_tensor(P)
    num_states = P.shape[1].value
    M_0 = tf.constant(1.0 / num_states, shape=(num_states, num_states))

    def M_dp1_op(M_d):
        return (((1 - gamma) * tf.eye(M_d.shape[0].value)) +
                (gamma * Pi @ P @ M_d))

    def error_above_threshold(M_d, M_dp1):
        return tf.reduce_sum(tf.abs(M_dp1 - M_d)) > threshold

    def cond(d, M_d, M_dp1):
        error_op = error_above_threshold(M_d, M_dp1)
        return tf.squeeze(
            tf.logical_or(
                tf.logical_and(max_num_iterations < 1, error_op),
                tf.logical_and(tf.less(d, max_num_iterations), error_op)))

    return tf.while_loop(
        cond,
        lambda d, _, M_d: [d + 1, M_d, M_dp1_op(M_d)],
        [1, M_0, M_dp1_op(M_0)],
        parallel_iterations=1)[-1]


def state_distribution(state_successor_rep, state_probs):
    '''
    Probability of terminating in each state.

    |States| by 1 Tensor
    '''
    return tf.matmul(state_successor_rep, state_probs, transpose_a=True)
