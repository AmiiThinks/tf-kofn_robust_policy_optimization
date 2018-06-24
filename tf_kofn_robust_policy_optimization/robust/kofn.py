import tensorflow as tf
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    inst_regrets_op, \
    associated_ops, \
    value_ops, state_successor_policy_evaluation_op
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


class UncertainRewardDiscountedContinuingKofnGame(object):
    def __init__(self,
                 config,
                 root_op,
                 transition_model_op,
                 reward_models_op,
                 gamma,
                 cap_negative_advantages=False):
        if reward_models_op.shape[1].value != config.num_sampled_worlds():
            print(reward_models_op.shape[1].value)
            print(config.num_sampled_worlds())
        assert reward_models_op.shape[1].value == config.num_sampled_worlds()
        self.config = config
        self.root_op = root_op
        self.transition_model_op = transition_model_op
        self.reward_models_op = reward_models_op

        self.advantages_op = tf.Variable(
            tf.zeros([self.num_states(), self.num_actions()]))

        (self.Pi_op, self.action_values_op, self.state_values_op,
         self.evs_op) = associated_ops(
             self.advantages_op,
             root_op,
             transition_model_op,
             reward_models_op,
             gamma=gamma,
             max_num_iterations=int(1e3))

        assert (len(self.evs_op.shape) == 2)
        assert (self.evs_op.shape[0].value == reward_models_op.shape[1].value)
        assert (self.evs_op.shape[1].value == 1)

        self.mdp_weights_op = world_weights(self.config.n_weights,
                                            self.config.k_weights,
                                            tf.squeeze(self.evs_op))
        self.mdp_weights_op = tf.expand_dims(self.mdp_weights_op, axis=1)

        self.ev_op = tf.squeeze(
            tf.matmul(self.evs_op, self.mdp_weights_op, transpose_a=True))

        self.max_num_training_pe_iterations = tf.placeholder(tf.int32)
        (self.training_action_values_op, self.training_state_values_op,
         self.training_evs_op) = value_ops(
             self.Pi_op,
             root_op,
             transition_model_op,
             reward_models_op,
             gamma=gamma,
             threshold=1e-8,
             max_num_iterations=self.max_num_training_pe_iterations)

        self.training_mdp_weights_op = world_weights(self.config.n_weights,
                                                     self.config.k_weights,
                                                     tf.squeeze(self.evs_op))
        self.training_mdp_weights_op = tf.expand_dims(
            self.training_evs_op, axis=1)

        self.r_s_op = tf.reshape(
            inst_regrets_op(self.training_action_values_op, Pi=self.Pi_op)
            @ self.training_mdp_weights_op,
            shape=self.advantages_op.shape)

        next_advantages = self.advantages_op + self.r_s_op
        if cap_negative_advantages:  # RM+
            next_advantages = tf.maximum(0.0, next_advantages)

        self.update_op = tf.assign(self.advantages_op, next_advantages)
        '''
        Probability of each action given each state.

        |States| by |actions| Tensor.
        '''
        self.policy_op = l1_projection_to_simplex(self.advantages_op, axis=1)
        '''
        Discounted state successor distribution.

        |States| by |States| Tensor.
        '''
        self.state_successor_op = state_successor_policy_evaluation_op(
            transition_model_op, self.Pi_op, gamma=gamma)
        '''
        Probability of terminating in each state.

        |States| by 1 Tensor
        '''
        self.state_distribution_op = tf.matmul(
            tf.transpose(self.state_successor_op), root_op)

    def num_states(self):
        return self.root_op.shape[0].value

    def num_actions(self):
        return int(self.transition_model_op.shape[0].value / self.num_states())

    def name(self):
        return self.config.label()

    def __str__(self):
        return self.name()
