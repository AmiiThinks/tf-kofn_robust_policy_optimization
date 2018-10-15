import tensorflow as tf
from tf_kofn_robust_policy_optimization.robust import rank_to_element_weights
from tf_kofn_robust_policy_optimization.robust.kofn import \
    UncertainRewardKofnTrainer


class ContextualKofnGame(object):
    batch_idx = 0
    world_idx = -1
    context_idx = 1
    action_idx = 2

    @classmethod
    def environment(cls, mixture_constraint_weights, sample_rewards, **kwargs):
        def play_game(policy):
            return cls(mixture_constraint_weights, sample_rewards(), policy,
                       **kwargs).kofn_utility

        return play_game

    def __init__(self,
                 mixture_constraint_weights,
                 u,
                 strat,
                 context_weights=None):
        self.mixture_constraint_weights = tf.convert_to_tensor(
            mixture_constraint_weights)

        self.u = tf.convert_to_tensor(u)
        if len(self.u.shape) < 4:
            extra_dims = 4 - len(self.u.shape)
            self.u = tf.reshape(
                self.u, [1] * extra_dims + [s.value for s in self.u.shape])

        self.strat = tf.convert_to_tensor(strat)
        if len(self.strat.shape) < 3:
            extra_dims = 3 - len(self.strat.shape)
            self.strat = tf.reshape(
                self.strat,
                [1] * extra_dims + [s.value for s in self.strat.shape])

        assert self.u.shape[self.world_idx].value == self.num_worlds()
        assert self.u.shape[self.context_idx].value == self.num_contexts()
        assert self.u.shape[self.action_idx].value == self.num_actions()

        self.policy_weighted_action_values = (
            self.u * tf.expand_dims(self.strat, axis=self.world_idx))
        self.context_evs = tf.reduce_sum(
            self.policy_weighted_action_values, axis=self.action_idx)
        assert (
            self.context_evs.shape[self.world_idx].value == self.num_worlds())
        assert (self.context_evs.shape[self.context_idx].value ==
                self.num_contexts())
        assert len(self.context_evs.shape) == 3

        if context_weights is not None:
            context_weights = tf.convert_to_tensor(context_weights)
            context_weights = tf.reshape(context_weights,
                                         [1, self.num_contexts(), 1])
            self.evs = tf.reduce_sum(
                self.context_evs * context_weights, axis=self.context_idx)
        else:
            self.evs = tf.reduce_mean(self.context_evs, axis=self.context_idx)
        assert self.evs.shape[self.batch_idx].value == self.batch_size()
        assert self.evs.shape[self.world_idx].value == self.num_worlds()

        self.k_weights = rank_to_element_weights(
            self.mixture_constraint_weights, self.evs)

        self.root_ev = tf.reduce_sum(
            tf.reduce_sum(self.evs * self.k_weights, axis=-1), axis=-1)

        shape = [1] * len(self.u.shape)
        shape[self.batch_idx] = self.batch_size()
        shape[self.world_idx] = self.num_worlds()

        self.kofn_utility = tf.reduce_sum(
            self.u * tf.reshape(self.k_weights, shape), axis=self.world_idx)

        assert (self.kofn_utility.shape[self.context_idx].value ==
                self.num_contexts())
        assert (self.kofn_utility.shape[self.action_idx].value ==
                self.num_actions())

        if self.batch_size() < 2:
            self.u = tf.reshape(
                self.u,
                [self.num_contexts(),
                 self.num_actions(),
                 self.num_worlds()])
            self.evs = tf.squeeze(self.evs)
            self.k_weights = tf.squeeze(self.k_weights)
            self.kofn_utility = tf.reshape(
                self.kofn_utility,
                [self.num_contexts(), self.num_actions()])

    def batch_size(self):
        return max(self.u.shape[self.batch_idx].value,
                   self.strat.shape[self.batch_idx].value)

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
