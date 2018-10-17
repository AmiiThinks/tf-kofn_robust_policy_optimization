import tensorflow as tf
from tf_kofn_robust_policy_optimization import cache
from tf_kofn_robust_policy_optimization.robust.kofn import \
    UncertainRewardKofnTrainer, \
    KofnEvsAndWeights, \
    kofn_action_values


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

        self.batch_size = max(self.u.shape[self.batch_idx].value,
                              self.strat.shape[self.batch_idx].value)

        self.policy_weighted_action_values = (
            self.u * tf.expand_dims(self.strat, axis=self.world_idx))
        self.context_evs = tf.reduce_sum(
            self.policy_weighted_action_values, axis=self.action_idx)
        assert (
            self.context_evs.shape[self.world_idx].value == self.num_worlds())
        assert (self.context_evs.shape[self.context_idx].value ==
                self.num_contexts())
        assert len(self.context_evs.shape) == 3

        self.kofn_evs_and_weights = KofnEvsAndWeights(
            self.context_evs,
            self.mixture_constraint_weights,
            context_weights=context_weights)

        if self.one_in_batch():
            self.u = tf.reshape(
                self.u,
                [self.num_contexts(),
                 self.num_actions(),
                 self.num_worlds()])

    def one_in_batch(self):
        return self.batch_size < 2

    @cache
    def kofn_utility(self):
        kofn_utility = kofn_action_values(self.u, self.k_weights)
        assert (kofn_utility.shape[self.context_idx - self.one_in_batch()]
                .value == self.num_contexts())
        assert (kofn_utility.shape[self.action_idx - self.one_in_batch()].value
                == self.num_actions())
        return kofn_utility

    @cache
    def k_weights(self):
        return (tf.squeeze(self.kofn_evs_and_weights.world_weights)
                if self.one_in_batch() else
                self.kofn_evs_and_weights.world_weights)

    @cache
    def evs(self):
        return (tf.squeeze(self.kofn_evs_and_weights.ev_given_world)
                if self.one_in_batch() else
                self.kofn_evs_and_weights.ev_given_world)

    @cache
    def root_ev(self):
        return (tf.squeeze(self.kofn_evs_and_weights.ev)
                if self.one_in_batch() else self.kofn_evs_and_weights.ev)

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
