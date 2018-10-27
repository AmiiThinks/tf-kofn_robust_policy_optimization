import tensorflow as tf
from deprecation import deprecated
from tf_kofn_robust_policy_optimization import cache
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    dual_action_value_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op
from tf_kofn_robust_policy_optimization.robust.contextual_kofn import \
    ContextualKofnGame
from tf_kofn_robust_policy_optimization.robust.kofn import \
    UncertainRewardKofnTrainer, \
    KofnEvsAndWeights


class UncertainRewardDiscountedContinuingKofnGame(object):
    world_idx = -1
    state_idx = world_idx + 1
    action_idx = state_idx + 1
    successor_state_idx = action_idx + 1

    @classmethod
    def environment(cls, *args, **kwargs):
        def f(policy):
            return UncertainRewardDiscountedContinuingKofnGame(
                *args, **kwargs).game.kofn_utility

        return f

    def __init__(self,
                 mixture_constraint_weights,
                 root_op,
                 transition_model_op,
                 reward_models_op,
                 policy,
                 gamma=0.9):
        self.mixture_constraint_weights = tf.convert_to_tensor(
            mixture_constraint_weights)
        self.root_op = tf.convert_to_tensor(root_op)
        self.transition_model_op = tf.convert_to_tensor(transition_model_op)
        self.reward_models_op = tf.convert_to_tensor(reward_models_op)
        self.policy = tf.convert_to_tensor(policy)

        self.gamma = gamma

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

    @cache
    def action_values(self):
        action_values = tf.transpose(
            dual_action_value_policy_evaluation_op(
                self.transition_model_op,
                self.policy,
                tf.transpose(self.reward_models_op, [2, 0, 1]),
                gamma=self.gamma), [1, 2, 0])

        assert (action_values.shape[self.state_idx].value == self.num_states())
        assert (
            action_values.shape[self.action_idx].value == self.num_actions())
        assert (action_values.shape[self.world_idx].value == self.num_worlds())

        return action_values

    @cache
    def contextual_kofn_game(self):
        return ContextualKofnGame(self.mixture_constraint_weights,
                                  self.action_values, self.policy,
                                  self.root_op)

    @property
    def state_values(self):
        return self.contextual_kofn_game.context_evs

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


class UncertainRewardDiscountedContinuingKofnEvEnv(object):
    def __init__(self,
                 mixture_constraint_weights,
                 root_probs,
                 transition_model,
                 sample_rewards,
                 gamma=0.9):
        self.mixture_constraint_weights = mixture_constraint_weights
        self.root_probs = root_probs
        self.transition_model = transition_model
        self.sample_rewards = sample_rewards
        self.game = None
        self.gamma = gamma

    def __call__(self, policy):
        state_values = tf.transpose(
            dual_state_value_policy_evaluation_op(
                self.transition_model,
                policy,
                tf.transpose(self.sample_rewards(), [2, 0, 1]),
                gamma=self.gamma))

        kofn_evs_and_weights = KofnEvsAndWeights(
            tf.squeeze(state_values),
            self.mixture_constraint_weights,
            context_weights=self.root_probs)
        return kofn_evs_and_weights.ev


@deprecated(
    details=(
        'Outdated. Use `UncertainRewardDiscountedContinuingKofnGame` directly instead.'
    )
)  # yapf:disable
class UncertainRewardDiscountedContinuingKofnTrainer(
        UncertainRewardKofnTrainer):
    def __init__(self, root_probs, transitions, *args, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_probs = root_probs
        self.transitions = transitions
        self.gamma = gamma

    def _eval_game(self, rewards, policy):
        return UncertainRewardDiscountedContinuingKofnGame(
            self.prob_ith_element_is_sampled,
            self.root_probs,
            self.transitions,
            rewards,
            policy,
            gamma=self.discount)
