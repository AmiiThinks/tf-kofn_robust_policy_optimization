import tensorflow as tf
from tf_kofn_robust_policy_optimization.robust import rank_to_element_weights
from tf_kofn_robust_policy_optimization.robust.kofn import \
    UncertainRewardKofnTrainer


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
        assert len(self.context_evs.shape) == 2

        weighted_context_evs = self.context_evs
        if context_weights is not None:
            context_weights = tf.convert_to_tensor(context_weights)
            context_weights = tf.reshape(context_weights,
                                         [self.num_contexts(), 1])
            weighted_context_evs *= context_weights

        self.evs = tf.reduce_mean(weighted_context_evs, axis=self.context_idx)
        assert self.evs.shape[self.world_idx].value == self.num_worlds()

        self.k_weights = rank_to_element_weights(
            self.mixture_constraint_weights, self.evs)

        self.root_ev = tf.reduce_sum(self.evs * self.k_weights)

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


class ContextualKofnTrainer(UncertainRewardKofnTrainer):
    def _eval_game(self, rewards, policy):
        return ContextualKofnGame(self.prob_ith_element_is_sampled, rewards,
                                  policy)
