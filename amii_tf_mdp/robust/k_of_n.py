import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from ..pr_mdp import pr_mdp_expected_value, num_states, num_actions


def prob_ith_element_is_in_k_subset(n_weights, k_weights):
    '''
    Linear time algorithm to find the probability that the ith element is
    included given weights that chance uses to sample N and k.

    The naive algorithm for computing this is quadratic time. To derive
    the linear time algorithm, look at the probability that k is less than i
    (the completment of the desired probability) and
    split the sum into the part when N = n for n < i and the part where
    n >= i.
    '''
    n_prob = l1_projection_to_simplex(n_weights)
    w_n_bar = tf.cumsum(k_weights)
    a = n_prob / w_n_bar
    a = tf.where(tf.is_nan(a), tf.zeros_like(a), a)
    w_i_minus_one_bar = tf.cumsum(k_weights, exclusive=True)
    b = w_i_minus_one_bar * tf.cumsum(a, reverse=True)
    return 1.0 - (tf.cumsum(n_prob, exclusive=True) + b)


def prob_ith_element_is_sampled(n_weights, k_weights):
    # TODO Getting the probability that chance selects the ith MDP is not
    # as simple as normalizing the prob of the ith element...
    # TODO This only works when only one k_weight is greater than zero.
    # TODO The math for doing this properly is fairly simple, just need to
    # code it up and test it
    return l1_projection_to_simplex(
        prob_ith_element_is_in_k_subset(n_weights, k_weights)
    )


# rank_weights = prob_ith_element_is_sampled(n_weights, k_weights)
# chance_prob_sequences = [
#     pr_mdp_rollout(horizon, roots[i], transitions[i])
#     for i in range(len(n_weights))
# ]

def rank_to_element_weights(rank_weights, elements):
    # Sorted in ascending order
    ranked_indices = tf.reverse(
        tf.nn.top_k(elements, k=elements.shape[-1].value, sorted=True)[1],
        [0]
    )
    return tf.scatter_nd(
        tf.expand_dims(ranked_indices, dim=1),
        rank_weights,
        [rank_weights.shape[0].value]
    )


def k_of_n_mdp_weights(n_weights, k_weights, evs):
    return rank_to_element_weights(
        prob_ith_element_is_sampled(n_weights, k_weights),
        evs
    )


def k_of_n_pr_mdp_evs(horizon, chance_prob_sequences, reward_models, strat):
    return tf.stack(
        [
            pr_mdp_expected_value(
                horizon,
                num_states(reward_models[i]),
                num_actions(reward_models[i]),
                chance_prob_sequences[i],
                reward_models[i],
                strat
            )
            for i in range(len(reward_models))
        ],
        axis=0
    )


def k_of_n_ev(evs, weights): return tf.tensordot(evs, weights, 1)


def k_of_n_regret_update(
    chance_prob_sequence_list,
    reward_models,
    weights,
    learner
):
    weighted_rewards = [
        chance_prob_sequence_list[i] * reward_models[i] * weights[i]
        for i in range(len(reward_models))
    ]
    inst_regrets = [learner.instantaneous_regrets(r) for r in weighted_rewards]
    regret_update = learner.updated_regrets(
        sum(inst_regrets[1:], inst_regrets[0])
    )
    return regret_update
