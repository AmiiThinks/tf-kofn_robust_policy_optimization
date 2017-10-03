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
        )


def num_ir_sequences(horizon, num_states): return horizon * num_states


def num_pr_sequences_at_timestep(t, num_states, num_actions):
    if t < 0:
        return 0
    else:
        return num_states * (num_states * num_actions) ** t


def prob_sequence_state_and_action(
    prob_sequence_action_state,
    strat=None,
    num_actions=None
):
    '''
    Params:
    - prob_sequence_action_state: Tensor with rank > 2 (* by |A| by |S|).
    - [strat]: (Optional) Rank-2 Tensor (|Seq| * |A| * |S| by |A'|).
        Defaults to all ones so that every row is the probability of the
        state under all pure strategies. Must supply the number of actions
        in the `num_actions` parameter if `strat` is `None`.
    - [num_actions]: (Optional) The number of actions. Only required when
        `strat` is `None`.

    Returns:
    - Rank-3 Tensor (|Seq| * |A| by |S| by |A'|). Prob of being in each
        sequence ending in each state state and sampling each action under
        strategy `strat`.
    '''
    if strat is None:
        strat = tf.ones(
            (
                tf.reshape(prob_sequence_action_state, [-1]).shape[0].value,
                num_actions
            )
        )
    return tf.reshape(
        prob_state_and_action(
            tf.reshape(prob_sequence_action_state, shape=[-1]),
            strat=strat,
            num_actions=num_actions
        ),
        shape=(
            -1,
            prob_sequence_action_state.shape[-1].value,
            num_actions
        )
    )


def prob_next_sequence_state_action_and_next_state(
    transition_model,
    prob_sequence_action_state,
    strat=None,
    num_actions=None
):
    '''
    Params:
    - prob_sequence_action_state: Tensor with rank > 2 (* by |A| by |S|).
    - [strat]: (Optional) Rank-2 Tensor (|Seq| * |A| * |S| by |A|).
        Defaults to all ones so that every row is the probability of the
        state under all pure strategies. Must supply the number of actions
        in the `num_actions` parameter if `strat` is `None`.
    - [num_actions]: (Optional) The number of actions. Only required when
        `strat` is `None`.

    Returns:
    - Rank-4 Tensor (|Seq| * |A| by |S| by |A| by |S|). Prob of being in each
        state after each of the next sequences sampling each action under
        strategy `strat`.
    '''
    prob_sequence_state_and_action_ = prob_sequence_state_and_action(
        prob_sequence_action_state,
        strat=strat,
        num_actions=num_actions
    )
    prob_state_action_sequence_next_state = (
        tf.transpose(
            tf.expand_dims(transition_model, 3),
            [0, 1, 3, 2]
        ) *
        tf.expand_dims(
            tf.transpose(prob_sequence_state_and_action_, [1, 2, 0]),
            3
        )
    )
    prob_sequence_state_action_next_state = tf.transpose(
        prob_state_action_sequence_next_state,
        [2, 0, 1, 3]
    )
    return prob_sequence_state_action_next_state
