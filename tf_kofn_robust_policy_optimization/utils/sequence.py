import tensorflow as tf
from .probability import prob_state_and_action


def num_pr_sequences(horizon, num_states, num_actions):
    if horizon < 0:
        return 0
    else:
        return int(
            num_states * (
                (num_states * num_actions)**(horizon + 1) - 1
            ) / (
                num_states * num_actions - 1
            )
        )  # yapf:disable


def num_pr_prefixes(horizon, num_states, num_actions):
    return int(num_pr_sequences(horizon, num_states, num_actions) / num_states)


def num_ir_sequences(horizon, num_states):
    return horizon * num_states


def num_pr_sequences_at_timestep(t, num_states, num_actions):
    if t < 0:
        return 0
    else:
        return num_states * (num_states * num_actions)**t


def static_size(tensor):
    p = 1
    for i in range(len(tensor.shape)):
        if tensor.shape[i].value is not None and tensor.shape[i].value > 0:
            p *= tensor.shape[i].value
    return p


def prob_sequence_state_and_action(prob_sequence_action_state,
                                   strat=None,
                                   num_actions=None):
    '''
    Params:
    - prob_sequence_action_state: Tensor with rank > 2 (* by |A| by |S|).
    - [strat]: (Optional) Rank-2 Tensor (|Seq| * |A| * |S| by |A'|).
        Defaults to all ones so that every row is the probability of the
        state under all pure strategies. Must supply the number of actions
        in the `num_actions` parameter if `strat` is `None`.
    - [num_actions]: TODO

    Returns:
    - Rank-3 Tensor (|Seq| * |A| by |S| by |A'|). Prob of being in each
        sequence ending in each state state and sampling each action under
        strategy `strat`.
    '''
    if strat is None:
        if num_actions is None:
            num_actions = prob_sequence_action_state.shape[-2].value
        strat = tf.ones((static_size(prob_sequence_action_state), num_actions))
    else:
        num_actions = strat.shape[-1].value
    prob_tensor = prob_state_and_action(
        tf.reshape(
            prob_sequence_action_state,
            shape=[static_size(prob_sequence_action_state)]
        ),
        strat=strat,
        num_actions=num_actions
    )  # yapf:disable
    return tf.reshape(
        prob_tensor,
        shape=(
            int(
                static_size(prob_tensor) /
                (prob_sequence_action_state.shape[-1].value * num_actions)
            ),
            prob_sequence_action_state.shape[-1].value,
            num_actions
        )
    )  # yapf:disable


def prob_next_sequence_state_action_and_next_state(transition_model,
                                                   prob_sequence_action_state,
                                                   strat=None):
    '''
    Params:
    - transition_model: |S| by |A| by |S| Tensor.
    - prob_sequence_action_state: Tensor with rank > 2 (* by |A| by |S|).
    - [strat]: (Optional) Rank-2 Tensor (|Seq| * |A| * |S| by |A|).
        Defaults to all ones so that every row is the probability of the
        state under all pure strategies. Must supply the number of actions
        in the `num_actions` parameter if `strat` is `None`.

    Returns:
    - Rank-4 Tensor (|Seq| * |A| by |S| by |A| by |S|). Prob of being in each
        state after each of the next sequences sampling each action under
        strategy `strat`.
    '''
    prob_sequence_state_and_action_ = prob_sequence_state_and_action(
        prob_sequence_action_state, strat=strat)
    prob_state_action_sequence_next_state = (
        tf.transpose(
            tf.expand_dims(transition_model, 3),
            [0, 1, 3, 2]
        ) *
        tf.expand_dims(
            tf.transpose(prob_sequence_state_and_action_, [1, 2, 0]),
            3
        )
    )  # yapf:disable
    prob_sequence_state_action_next_state = tf.transpose(
        prob_state_action_sequence_next_state,
        [2, 0, 1, 3]
    )  # yapf:disable
    return prob_sequence_state_action_next_state
