import tensorflow as tf
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    dual_action_value_policy_evaluation_op
from tf_kofn_robust_policy_optimization.robust import \
    prob_ith_element_is_sampled
from tf_kofn_robust_policy_optimization.robust.contextual_kofn import \
    ContextualKofnGame


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

    def to_yml(self, indent=0):
        ws = ' ' * indent
        return "\n".join([
            '{}{}: {}'.format(ws, key, value)
            for key, value in [('n', self.num_sampled_worlds()), ('k', self.k)]
        ])

    def num_sampled_worlds(self):
        return len(self.n_weights)

    def label(self):
        return '{}-of-{}'.format(self.k, self.num_sampled_worlds())

    def __str__(self):
        return self.label() + ' template'


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
        return self.contextual_kofn_game.kofn_utility

    def num_states(self):
        return self.root_op.shape[self.state_idx].value

    def num_actions(self):
        return self.policy.shape[self.action_idx].value

    def num_worlds(self):
        return self.mixture_constraint_weights.shape[0].value
