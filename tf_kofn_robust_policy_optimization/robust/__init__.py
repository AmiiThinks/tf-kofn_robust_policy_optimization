import tensorflow as tf
from tf_kofn_robust_policy_optimization.utils.tensor import \
    l1_projection_to_simplex


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
        prob_ith_element_is_in_k_subset(n_weights, k_weights))


def rank_to_element_weights(rank_weights, elements):
    rank_weights = tf.squeeze(tf.convert_to_tensor(rank_weights))
    elements = tf.convert_to_tensor(elements)
    _, ranked_indices = tf.nn.top_k(
        -elements, elements.shape[-1].value, sorted=True)
    if len(elements.shape) < 2:
        return tf.manip.scatter_nd(
            tf.expand_dims(ranked_indices, axis=-1), rank_weights,
            rank_weights.shape)
    else:
        return tf.stack([
            tf.manip.scatter_nd(
                tf.expand_dims(ranked_indices[i], axis=-1),
                rank_weights, rank_weights.shape)
            for i in range(ranked_indices.shape[0].value)
        ])


def world_weights(n_weights, k_weights, evs):
    return rank_to_element_weights(
        prob_ith_element_is_sampled(n_weights, k_weights), evs)


def world_utilities(utility_of_world_given_action, strategy, n_weights,
                    k_weights):
    evs = tf.matmul(strategy, utility_of_world_given_action, transpose_a=True)
    p = tf.expand_dims(
        world_weights(n_weights, k_weights, tf.squeeze(evs)), axis=1)
    return tf.matmul(utility_of_world_given_action, p)
