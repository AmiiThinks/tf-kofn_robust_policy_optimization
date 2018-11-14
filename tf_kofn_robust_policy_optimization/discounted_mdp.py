import tensorflow as tf
from deprecation import deprecated
from tf_contextual_prediction_with_expert_advice import \
    l1_projection_to_simplex, \
    greedy_policy
from tf_kofn_robust_policy_optimization.utils.tensor import \
    matrix_to_block_matrix_op as policy_block_matrix_op


def dual_action_value_policy_evaluation_op(transitions, policy, r, gamma=0.9):
    '''r may have an initial batch dimension.'''
    v = dual_state_value_policy_evaluation_op(
        transitions, policy, r, gamma=gamma)
    return r + gamma * tf.tensordot(v, transitions, axes=[-1, -1])


def dual_state_value_policy_evaluation_op(transitions, policy, r, gamma=0.9):
    '''r may have an initial batch dimension.'''
    M = state_successor_policy_evaluation_op(transitions, policy, gamma=gamma)
    r = tf.convert_to_tensor(r)
    if len(r.shape) > 2:
        weighted_rewards = tf.einsum('bsa,sa->bs', r, policy)
    else:
        weighted_rewards = tf.reduce_sum(r * policy, axis=-1)
    return tf.tensordot(weighted_rewards, M, axes=[-1, -1])


def primal_state_value_on_policy_op(transitions, policy, r, discount, v=None):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value
    num_actions = transitions.shape[1].value
    num_state_actions = num_states * num_actions
    v = tf.convert_to_tensor(v)

    if v is None:
        v = tf.zeros([num_states, 1])
    if len(v.shape) < 2:
        v = tf.expand_dims(v, -1)
    P = tf.reshape(transitions, [num_state_actions, num_states])

    return tf.reduce_sum(
        policy * (r + discount * tf.reshape(P @ v, [num_states, num_actions])),
        axis=-1)


def primal_state_value_policy_evaluation(transitions,
                                         policy,
                                         r,
                                         gamma=0.9,
                                         threshold=1e-15,
                                         max_num_iterations=-1,
                                         v_0=None):
    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value

    if v_0 is None:
        v_0 = tf.zeros([num_states])

    def error_above_threshold(v_d, v_dp1):
        return tf.reduce_sum(tf.abs(v_dp1 - v_d)) > threshold

    def cond(d, v_d, v_dp1):
        error_is_high = True if threshold is None else error_above_threshold(
            v_d, v_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(max_num_iterations, 1), error_is_high),
            tf.logical_and(tf.less(d, max_num_iterations), error_is_high))

    return tf.while_loop(
        cond,
        lambda d, _, v_d: [
            d + 1,
            v_d,
            primal_state_value_on_policy_op(transitions, policy, r, gamma, v_d)
        ],
        [1, v_0, primal_state_value_on_policy_op(transitions, policy, r, gamma, v_0)],
        parallel_iterations=1,
        name='primal_state_value_policy_evaluation/while_loop')[-1]


def primal_action_value_policy_evaluation_op(transitions,
                                             policy,
                                             r,
                                             gamma=0.9,
                                             **kwargs):
    v = primal_state_value_policy_evaluation(
        transitions, policy, r, gamma=gamma, **kwargs)
    return r + gamma * tf.tensordot(v, transitions, axes=[-1, -1])


def generalized_policy_iteration_op(transitions,
                                    r,
                                    alpha=1.0,
                                    gamma=0.9,
                                    t=10,
                                    pi_threshold=1e-15,
                                    max_num_pe_iterations=lambda s: 1,
                                    v_0=None,
                                    value_threshold=1e-15):
    def q(v):
        return r + gamma * tf.tensordot(v, transitions, axes=[-1, -1])

    def next_v(d, v):
        return primal_state_value_policy_evaluation(
            transitions,
            greedy_policy(q(v)),
            r,
            gamma,
            threshold=pi_threshold,
            max_num_iterations=max_num_pe_iterations(d),
            v_0=v_0)

    transitions = tf.convert_to_tensor(transitions)
    num_states = transitions.shape[0].value

    if v_0 is None:
        v_0 = tf.zeros([num_states])

    def error_above_threshold(v_d, v_dp1):
        return tf.reduce_sum(tf.abs(v_dp1 - v_d)) > value_threshold

    def cond(d, v_d, v_dp1):
        error_is_high = True if value_threshold is None else error_above_threshold(
            v_d, v_dp1)
        return tf.logical_or(
            tf.logical_and(tf.less(t, 1), error_is_high),
            tf.logical_and(tf.less(d, t), error_is_high))

    return greedy_policy(q(
        tf.while_loop(
            cond,
            lambda d, _, v_dp1: [
                d + 1, v_dp1, v_dp1 + alpha * (next_v(d, v_dp1) - v_dp1)
            ],
            [1, v_0, next_v(0, v_0)],
            parallel_iterations=1)[-1]))


def state_successor_policy_evaluation_op(transitions, policy, gamma=0.9):
    '''
    The discounted unnormalized successor representation for the given
    transitions and policy.

    If gamma is less than 1, multiplying each element by 1 - gamma recovers
    the row-normalized version.
    '''
    negative_state_to_state = tf.einsum('san,sa->sn', transitions,
                                        -gamma * policy)
    eye_minus_gamma_state_to_state = tf.linalg.set_diag(
        negative_state_to_state, 1.0 + tf.diag_part(negative_state_to_state))

    return tf.matrix_inverse(eye_minus_gamma_state_to_state)


def state_distribution(state_successor_rep, state_probs):
    '''
    Probability of terminating in each state.

    |States| by 1 Tensor

    Parameters:
    - state_successor_rep: |States| by |States| successor representation.
    - state_probs: (m by) |States| vector of initial state probabilities.
    '''
    state_probs = tf.convert_to_tensor(state_probs)
    if len(state_probs.shape) < 2:
        return tf.matmul(
            tf.expand_dims(state_probs, axis=0), state_successor_rep)[0]
    else:
        return tf.matmul(state_probs, state_successor_rep)


@deprecated(
    details=(
        'Outdated and poorly named. Use state and action policy evaluation methods directly instead.'
    )
)  # yapf:disable
def value_ops(Pi, root_op, transition_model_op, reward_model_op, **kwargs):
    action_values_op = dual_action_value_policy_evaluation_op(
        transition_model_op, Pi, reward_model_op, **kwargs)

    state_values_op = Pi @ action_values_op
    ev_op = tf.reduce_sum(root_op * state_values_op, axis=-1)

    return action_values_op, state_values_op, ev_op


@deprecated(
    details=(
        'Outdated and poorly named. Use state and action policy evaluation methods directly instead.'
    )
)  # yapf:disable
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
