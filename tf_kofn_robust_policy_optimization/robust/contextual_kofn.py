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
        u_has_batch = len(self.u.shape) > 3

        self.strat = tf.convert_to_tensor(strat)
        strat_has_batch = len(self.strat.shape) > 2

        result_string = 'sw'
        if u_has_batch:
            u_string = 'bsaw'
            result_string = 'bsw'
        else:
            u_string = 'saw'

        if strat_has_batch:
            strat_string = 'bsa'
            result_string = 'bsw'
        else:
            strat_string = 'sa'

        self.context_evs = tf.einsum('{},{}->{}'.format(
            u_string, strat_string, result_string), self.u, self.strat)

        self.kofn_evs_and_weights = KofnEvsAndWeights(
            self.context_evs,
            self.mixture_constraint_weights,
            context_weights=context_weights)

    def one_in_batch(self):
        return self.batch_size < 2

    @cache
    def kofn_utility(self):
        return kofn_action_values(self.u, self.k_weights)

    @cache
    def k_weights(self):
        return self.kofn_evs_and_weights.world_weights

    @cache
    def evs(self):
        return self.kofn_evs_and_weights.ev_given_world

    @cache
    def root_ev(self):
        return self.kofn_evs_and_weights.ev

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
