import tensorflow as tf


def prob_action(state_distribution, strat):
    '''
    Params:
    - state_distribution: Rank-1 Tensor (|S|).
    - strat: Rank-2 Tensor (|S| by |A|).

    Returns:
    - Rank-1 Tensor (|A|). Prob of taking each action.
    '''
    return tf.tensordot(strat, state_distribution, axes=((0, ), (0, )))


def prob_state_and_action(state_distribution, strat=None, num_actions=None):
    '''
    Params:
    - state_distribution: Rank-1 Tensor (|S|).
    - [strat]: (Optional) Rank-2 Tensor (|S| by |A|). Defaults to all ones
        so that every row is the probability of the state under all pure
        strategies. Must supply the number of actions in the `num_actions`
        parameter if `strat` is `None`.
    - [num_actions]: (Optional) The number of actions. Only required when
        `strat` is `None`.

    Returns:
    - Rank-2 Tensor (|S| by |A|). Prob of being in each state and sampling
        each action under strategy `strat`.
    '''
    if strat is None:
        strat = tf.ones((state_distribution.shape[0].value, num_actions))
    return tf.transpose(tf.transpose(strat) * state_distribution)


def prob_state_given_action(state_distribution, strat=None, num_actions=None):
    '''
    Params:
    - state_distribution: Rank-1 Tensor (|S|).
    - [strat]: (Optional) Rank-2 Tensor (|S| by |A|). Defaults to all ones
        so that every row of the output is the probability of the state
        under all pure strategies. Must supply the number of actions in
        the `num_actions` parameter if `strat` is `None`.
    - [num_actions]: (Optional) The number of actions. Only required when
        `strat` is `None`.

    Returns:
    - Rank-2 Tensor (|A| by |S|). Prob of each state given the action was
        sampled under strategy `strat`.
    '''
    # |S| by |A|
    prob_state_and_action_ = prob_state_and_action(
        state_distribution,
        strat=strat,
        num_actions=num_actions
    )  # yapf:disable
    prob_action_ = tf.reduce_sum(prob_state_and_action_, axis=0)  # |A|
    prob_action_ = tf.where(
        tf.greater(prob_action_, 0.0),
        prob_action_,
        tf.constant(1.0, shape=prob_action_.shape)
    )  # yapf:disable
    return tf.transpose(prob_state_and_action_ / prob_action_)  # |A| by |S|


def prob_next_state_given_action(transition_model,
                                 state_distribution,
                                 strat=None):
    return tf.transpose(
        tf.reduce_sum(
            (
                tf.transpose(transition_model) *
                prob_state_given_action(
                    state_distribution,
                    strat=strat,
                    num_actions=transition_model.shape[1].value
                )
            ),
            axis=2
        )
    )  # yapf:disable


def prob_next_state(transition_model, state_distribution, strat=None):
    return tf.tensordot(
        transition_model,
        prob_state_and_action(
            state_distribution,
            strat=strat,
            num_actions=transition_model.shape[1].value
        ),
        axes=((0, 1), (0, 1))
    )  # yapf:disable
