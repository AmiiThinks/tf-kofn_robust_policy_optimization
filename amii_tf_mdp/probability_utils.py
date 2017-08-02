import tensorflow as tf


def prob_action(state_distribution, strat):
    '''
    Params:
    - state_distribution: Rank-1 Tensor (|S|).
    - strat: Rank-2 Tensor (|S| by |A|).

    Returns:
    - Rank-1 Tensor (|A|). Prob of taking each action.
    '''
    return tf.tensordot(strat, state_distribution, axes=((0,), (0,)))


def prob_state_and_action(state_distribution, strat):
    return tf.transpose(tf.transpose(strat) * state_distribution)


def prob_state_given_action(state_distribution, strat):
    '''
    Params:
    - state_distribution: Rank-1 Tensor (|S|).
    - strat: Rank-2 Tensor (|S| by |A|).

    Returns:
    - Rank-2 Tensor (|A| by |S|). Prob of each state given each action.
    '''
    # |S| by |A|
    prob_state_and_action_ = prob_state_and_action(state_distribution, strat)
    prob_action_ = tf.reduce_sum(prob_state_and_action_, axis=0)  # |A|
    prob_action_ = tf.where(
        tf.greater(prob_action_, 0.0),
        prob_action_,
        tf.constant(1.0, shape=prob_action_.shape)
    )
    return tf.transpose(prob_state_and_action_ / prob_action_)  # |A| by |S|


def prob_next_state_given_action(transition_model, state_distribution, strat):
    return tf.transpose(
        tf.reduce_sum(
            (
                tf.transpose(transition_model) *
                prob_state_given_action(state_distribution, strat)
            ),
            axis=2
        )
    )


def prob_next_state(transition_model, state_distribution, strat):
    return tf.tensordot(
        transition_model,
        prob_state_and_action(state_distribution, strat),
        axes=((0, 1), (0, 1))
    )
