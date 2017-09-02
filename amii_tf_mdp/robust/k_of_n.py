import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex


def prob_ith_element(n_weights, k_weights):
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
    a = n_prob / tf.cumsum(k_weights)
    a = tf.where(tf.is_nan(a), tf.zeros_like(a), a)
    b = tf.cumsum(k_weights, exclusive=True) * tf.cumsum(a, reverse=True)
    return 1.0 - (tf.cumsum(n_prob, exclusive=True) + b)


class KofnGadget(object):
    def __init__(self, n_weights, k_weights, mdps):
        self.i_weights = prob_ith_element(n_weights, k_weights)
        self.mdps = mdps
        self.evs = tf.stack([state.expected_value for state in mdps], axis=0)
        # Sort in descending order
        _, sorted_mdp_indices = tf.nn.top_k(
            self.evs,
            k=self.evs.shape[-1].value,
            sorted=True
        )
        # Sorted in ascending order
        self.sorted_mdp_indices = tf.reverse(sorted_mdp_indices, [0])
        self.mdp_weights = tf.scatter_nd(
            tf.expand_dims(self.sorted_mdp_indices, dim=1),
            self.i_weights,
            [self.max_num_mdps()]
        )
        self.weighted_rewards = [
            mdps[i].sequences * mdps[i].rewards * self.mdp_weights[i]
            for i in range(self.max_num_mdps())
        ]
        self.expected_value = tf.tensordot(self.evs, self.mdp_weights, 1)

    def max_num_mdps(self): return self.i_weights.shape[0].value
