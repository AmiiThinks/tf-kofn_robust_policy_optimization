import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from ..mdp import PrMdpState


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
    def __init__(self, n_weights, k_weights, mdp_sampler):
        '''

        Params:
        - n_weights
        - k_weights
        - mdp_sampler: Function that returns MDP states.
        '''
        self.i_weights = prob_ith_element(n_weights, k_weights)
        self.mdp_sampler = mdp_sampler

    def max_num_mdps(self): return self.i_weights.shape[0].value


class PrKofnGadget(KofnGadget):
    def __call__(self, strat, update):
        sampled_mdps = [self.mdp_sampler() for _ in range(self.max_num_mdps())]
        evs = tf.stack(
            [state.expected_value(strat) for state in sampled_mdps],
            axis=0
        )
        # Sort in descending order
        _, sorted_mdp_indices = tf.nn.top_k(
            evs,
            k=evs.shape[-1].value,
            sorted=True
        )
        # Sorted in ascending order
        sorted_mdp_indices = tf.reverse(sorted_mdp_indices, [0])
        mdp_weights = tf.scatter_nd(
            tf.expand_dims(sorted_mdp_indices, dim=1),
            self.i_weights,
            [self.max_num_mdps()]
        )
        unrolled_weighted_rewards = [
            (
                sampled_mdps[i].unroll() *
                sampled_mdps[i].mdp.rewards *
                mdp_weights[i]
            )
            for i in range(self.max_num_mdps())
        ]
        return (
            update(unrolled_weighted_rewards),
            tf.tensordot(evs, mdp_weights, 1)
        )
