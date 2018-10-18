import tensorflow as tf
from .utils.tensor import l1_projection_to_simplex, ind_max_op
from .utils.tensor import \
    matrix_to_block_matrix_op as policy_block_matrix_op


def state_action_successor_policy_evaluation_op(transitions,
                                                policy,
                                                gamma=0.9,
                                                threshold=1e-15,
                                                max_num_iterations=-1,
                                                H_0=None):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value
    num_actions = transitions.shape[1].value
    num_state_actions = num_states * num_actions

    if H_0 is None:
        H_0 = tf.constant(
            1.0 / num_state_actions,
            shape=(num_state_actions, num_state_actions))

    policy = tf.convert_to_tensor(policy)
    state_action_to_state_action = tf.reshape(
        tf.expand_dims(transitions, axis=-1)
        * tf.reshape(
            gamma * policy,
            [1, 1] + [dim.value for dim in policy.shape]
        ),
        H_0.shape
    )  # yapf:disable

    def H_dp1_op(H_d):
        future_return = state_action_to_state_action @ H_d
        return tf.linalg.set_diag(future_return,
                                  tf.diag_part(future_return) + 1.0 - gamma)

    def error_above_threshold(H_d, H_dp1):
        return tf.greater(tf.reduce_sum(tf.abs(H_dp1 - H_d)), threshold)

    def cond(d, H_d, H_dp1):
        error_is_high = True if threshold is None else error_above_threshold(
            H_d, H_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(max_num_iterations, 1), error_is_high),
            tf.logical_and(tf.less(d, max_num_iterations), error_is_high))

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
    extra_dims = r.shape[2:]
    shape = policy.shape.concatenate(extra_dims)
    H = state_action_successor_policy_evaluation_op(
        transitions,
        policy,
        gamma=gamma,
        threshold=threshold,
        max_num_iterations=max_num_iterations)
    action_values = tf.reshape(
        tf.tensordot(
            H,
            tf.reshape(r, H.shape[0:1].concatenate(extra_dims)),
            axes=[[1], [0]]), shape)
    if gamma < 1:
        action_values = action_values / (1.0 - gamma)
    return action_values


def dual_state_value_policy_evaluation_op(transitions,
                                          policy,
                                          r,
                                          gamma=0.9,
                                          threshold=1e-15,
                                          max_num_iterations=-1):
    policy = tf.convert_to_tensor(policy)
    r = tf.convert_to_tensor(r)
    M = state_successor_policy_evaluation_op(
        transitions,
        policy,
        gamma=gamma,
        threshold=threshold,
        max_num_iterations=max_num_iterations)
    if len(r.shape) > 2:
        M = tf.expand_dims(M, axis=-1)
        policy = tf.expand_dims(policy, axis=-1)
    weighted_rewards = tf.reduce_sum(r * policy, axis=1, keepdims=True)
    state_values = tf.tensordot(M, weighted_rewards, axes=[[1], [0]])
    if gamma < 1:
        state_values = state_values / (1.0 - gamma)
    return state_values


def primal_action_value_policy_evaluation_op(transitions,
                                             policy,
                                             r,
                                             gamma=0.9,
                                             threshold=1e-15,
                                             max_num_iterations=-1,
                                             q_0=None):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value
    num_actions = transitions.shape[1].value

    if q_0 is None:
        q_0 = tf.zeros([num_states, num_actions])

    discounted_policy = gamma * policy

    def q_dp1_op(q_d):
        discounted_return = tf.reduce_sum(
            transitions * tf.reshape(
                tf.reduce_sum(discounted_policy * q_d, axis=-1),
                [1, 1, num_states]),
            axis=-1)
        return r + discounted_return

    def error_above_threshold(q_d, q_dp1):
        return tf.reduce_sum(tf.abs(q_dp1 - q_d)) > threshold

    def cond(d, q_d, q_dp1):
        error_is_high = True if threshold is None else error_above_threshold(
            q_d, q_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(max_num_iterations, 1), error_is_high),
            tf.logical_and(tf.less(d, max_num_iterations), error_is_high))

    return tf.while_loop(
        cond,
        lambda d, _, q_d: [d + 1, q_d, q_dp1_op(q_d)],
        [1, q_0, q_dp1_op(q_0)],
        parallel_iterations=1,
        name='primal_action_value_policy_evaluation_op/while_loop')[-1]


def generalized_policy_iteration_op(transitions,
                                    r,
                                    alpha=1.0,
                                    gamma=0.9,
                                    t=10,
                                    pi_threshold=1e-15,
                                    max_num_pe_iterations=lambda s: 1,
                                    q_0=None,
                                    value_threshold=1e-15):
    transitions = tf.convert_to_tensor(transitions)

    if q_0 is None:
        q_0 = tf.zeros(transitions.shape[:2])

    def next_q(d, q):
        return primal_action_value_policy_evaluation_op(
            transitions,
            ind_max_op(q, axis=-1),
            r,
            gamma=gamma,
            threshold=pi_threshold,
            max_num_iterations=max_num_pe_iterations(d),
            q_0=q)

    def v(q):
        return tf.reduce_sum(
            transitions * tf.expand_dims(ind_max_op(q, axis=-1), axis=-1),
            axis=1)

    def error_above_threshold(q_d, q_dp1):
        return tf.reduce_sum(tf.abs(v(q_dp1) - v(q_d))) > value_threshold

    def cond(d, q_d, q_dp1):
        error_is_high = True if value_threshold is None else error_above_threshold(
            q_d, q_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(t, 1), error_is_high),
            tf.logical_and(tf.less(d, t), error_is_high))

    return ind_max_op(
        tf.while_loop(
            cond,
            lambda d, _, q_dp1: [
                d + 1, q_dp1, q_dp1 + alpha * (next_q(d, q_dp1) - q_dp1)
            ],
            [1, q_0, next_q(0, q_0)],
            parallel_iterations=1)[-1],
        axis=-1)


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


def state_successor_policy_evaluation_op(transitions,
                                         policy,
                                         gamma=0.9,
                                         threshold=1e-15,
                                         max_num_iterations=-1,
                                         M_0=None):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value

    if M_0 is None:
        M_0 = tf.constant(1.0 / num_states, shape=(num_states, num_states))

    weighted_transitions = (
        transitions * tf.expand_dims(gamma * policy, axis=-1))

    state_to_state = tf.reduce_sum(weighted_transitions, axis=1)

    def M_dp1_op(M_d):
        future_return = M_d @ state_to_state
        return tf.linalg.set_diag(future_return,
                                  tf.diag_part(future_return) + 1.0 - gamma)

    def error_above_threshold(M_d, M_dp1):
        return tf.greater(tf.reduce_sum(tf.abs(M_dp1 - M_d)), threshold)

    def cond(d, M_d, M_dp1):
        error_is_high = True if threshold is None else error_above_threshold(
            M_d, M_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(max_num_iterations, 1), error_is_high),
            tf.logical_and(tf.less(d, max_num_iterations), error_is_high))

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
