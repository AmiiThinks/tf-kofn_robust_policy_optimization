import tensorflow as tf
from ..utils.tensor import l1_projection_to_simplex


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
    # Sorted in ascending order
    ranked_indices = tf.reverse(
        tf.nn.top_k(elements, k=elements.shape[-1].value, sorted=True)[1], [0])
    return tf.scatter_nd(
        tf.expand_dims(ranked_indices, dim=1), rank_weights,
        [rank_weights.shape[0].value])


def world_weights(n_weights, k_weights, evs):
    return rank_to_element_weights(
        prob_ith_element_is_sampled(n_weights, k_weights), evs)


def world_utilities(utility_of_world_given_action, strategy, n_weights,
                    k_weights):
    evs = tf.matmul(strategy, utility_of_world_given_action, transpose_a=True)
    p = tf.expand_dims(
        world_weights(n_weights, k_weights, tf.squeeze(evs)), axis=1)
    return tf.matmul(utility_of_world_given_action, p)


def kofn_ev(evs, weights):
    return tf.tensordot(evs, weights, 1)


def kofn_regret_update(chance_prob_sequence_list, reward_models, weights,
                       learner):
    weighted_rewards = [
        chance_prob_sequence_list[i] * reward_models[i] * weights[i]
        for i in range(len(reward_models))
    ]
    inst_regrets = [learner.instantaneous_regrets(r) for r in weighted_rewards]
    regret_update = learner.updated_regrets(
        sum(inst_regrets[1:], inst_regrets[0]))
    return regret_update


class DeterministicKofnGameTemplate(object):
    def __init__(self, k, n):
        self.n_weights = [0.0] * n
        self.n_weights[n - 1] = 1.0

        self.k = k
        self.k_weights = [0.0] * n
        self.k_weights[k - 1] = 1.0

        self.prob_ith_element_is_sampled = prob_ith_element_is_sampled(
            self.n_weights, self.k_weights)

    def num_sampled_worlds(self):
        return len(self.n_weights)

    def label(self):
        return '{}-of-{}'.format(self.k, self.num_sampled_worlds())

    def __str__(self):
        return self.label() + ' template'


class ContextualKofnGame(object):
    def __init__(self, mixture_constraint_weights, u, strat):
        n = mixture_constraint_weights.shape[0].value
        assert u.shape[0].value == n

        num_contexts = strat.shape[0].value
        assert u.shape[1].value == num_contexts

        num_actions = strat.shape[1].value
        assert u.shape[2].value == num_actions

        self.mixture_constraint_weights = mixture_constraint_weights
        self.u = u

        self.context_evs = tf.reduce_sum(
            u * tf.tile(tf.expand_dims(strat, axis=0), [n, 1, 1]), axis=2)
        assert self.context_evs.shape[0].value == n
        assert self.context_evs.shape[1].value == num_contexts

        self.evs = tf.reduce_mean(self.context_evs, axis=1, keep_dims=True)

        self.k_weights = tf.expand_dims(
            rank_to_element_weights(self.mixture_constraint_weights,
                                    tf.squeeze(self.evs)),
            axis=1)

        self.root_ev = tf.squeeze(
            tf.matmul(self.evs, self.k_weights, transpose_a=True))

        self.kofn_utility = tf.reduce_sum(
            u * tf.tile(
                tf.expand_dims(self.k_weights, axis=2),
                [1, num_contexts, num_actions]),
            axis=0)
        assert self.kofn_utility.shape[0].value == num_contexts
        assert self.kofn_utility.shape[1].value == num_actions
