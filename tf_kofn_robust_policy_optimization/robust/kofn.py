import tensorflow as tf
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    dual_action_value_policy_evaluation_op
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
    world_idx = -1
    context_idx = 0
    action_idx = 1

    def __init__(self,
                 mixture_constraint_weights,
                 u,
                 strat,
                 context_weights=None):
        self.mixture_constraint_weights = tf.convert_to_tensor(
            mixture_constraint_weights)
        self.u = tf.convert_to_tensor(u)
        self.strat = tf.convert_to_tensor(strat)

        assert self.u.shape[self.world_idx].value == self.num_worlds()
        assert self.u.shape[self.context_idx].value == self.num_contexts()

        assert u.shape[self.action_idx].value == self.num_actions()

        self.policy_weighted_action_values = self.u * tf.expand_dims(
            self.strat, axis=self.world_idx)
        self.context_evs = tf.reduce_sum(
            self.policy_weighted_action_values, axis=self.action_idx)
        assert self.context_evs.shape[
            self.world_idx].value == self.num_worlds()
        assert self.context_evs.shape[
            self.context_idx].value == self.num_contexts()

        if context_weights is None:
            self.evs = tf.reduce_mean(self.context_evs, axis=self.context_idx)
        else:
            self.evs = tf.tensordot(
                self.context_evs, context_weights, axes=self.context_idx)

        self.k_weights = rank_to_element_weights(
            self.mixture_constraint_weights, self.evs)

        self.root_ev = tf.tensordot(self.evs, self.k_weights, axes=0)

        shape = [1] * len(u.shape)
        shape[self.world_idx] = self.num_worlds()
        self.kofn_utility = tf.reduce_sum(
            u * tf.reshape(self.k_weights, shape), axis=self.world_idx)

        assert self.kofn_utility.shape[
            self.context_idx].value == self.num_contexts()
        assert self.kofn_utility.shape[
            self.action_idx].value == self.num_actions()

    def num_contexts(self):
        return self.strat.shape[self.context_idx].value

    def num_actions(self):
        return self.strat.shape[self.action_idx].value

    def num_worlds(self):
        return self.mixture_constraint_weights.shape[0].value


class UncertainRewardDiscountedContinuingKofnGame(object):
    world_idx = ContextualKofnGame.world_idx
    state_idx = ContextualKofnGame.context_idx
    action_idx = ContextualKofnGame.action_idx
    successor_state_idx = action_idx + 1

    def __init__(self,
                 mixture_constraint_weights,
                 root_op,
                 transition_model_op,
                 reward_models_op,
                 policy,
                 gamma=0.9,
                 threshold=1e-15,
                 max_num_iterations=-1):
        self.mixture_constraint_weights = tf.convert_to_tensor(
            mixture_constraint_weights)
        self.root_op = tf.convert_to_tensor(root_op)
        self.transition_model_op = tf.convert_to_tensor(transition_model_op)
        self.reward_models_op = tf.convert_to_tensor(reward_models_op)
        self.policy = tf.convert_to_tensor(policy)

        self.gamma = gamma
        self.threshold = threshold
        self.max_num_iterations = max_num_iterations

        assert (self.reward_models_op.shape[self.world_idx].value ==
                self.num_worlds())

        assert (self.reward_models_op.shape[self.state_idx].value ==
                self.num_states())
        assert (self.transition_model_op.shape[self.state_idx].value ==
                self.num_states())
        assert (self.transition_model_op.shape[self.successor_state_idx].value
                == self.num_states())

        assert (self.reward_models_op.shape[self.action_idx].value ==
                self.num_actions())
        assert (self.transition_model_op.shape[self.action_idx].value ==
                self.num_actions())

        self.action_values = dual_action_value_policy_evaluation_op(
            self.transition_model_op,
            self.policy,
            self.reward_models_op,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)
        assert self.action_values.shape[
            self.state_idx].value == self.num_states()
        assert self.action_values.shape[
            self.action_idx].value == self.num_actions()
        assert self.action_values.shape[
            self.world_idx].value == self.num_worlds()

        self.contextual_kofn_game = ContextualKofnGame(
            self.mixture_constraint_weights, self.action_values, self.policy,
            self.root_op)

    @property
    def state_values(self):
        self.state_values = self.contextual_kofn_game.context_evs

    @property
    def evs(self):
        return self.contextual_kofn_game.evs

    @property
    def k_weights(self):
        return self.contextual_kofn_game.k_weights

    @property
    def root_ev(self):
        return self.contextual_kofn_game.root_ev

    @property
    def kofn_utility(self):
        return self.kofn_utility

    def num_states(self):
        return self.root_op.shape[self.state_idx].value

    def num_actions(self):
        return self.policy.shape[self.action_idx].value

    def num_worlds(self):
        return self.mixture_constraint_weights.shape[0].value
