import tensorflow as tf
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    dual_action_value_policy_evaluation_op
from tf_kofn_robust_policy_optimization.robust.contextual_kofn import \
    ContextualKofnGame
from tf_kofn_robust_policy_optimization.robust.kofn import \
    UncertainRewardKofnTrainer


class UncertainRewardDiscountedContinuingKofnGame(object):
    world_idx = ContextualKofnGame.world_idx
    state_idx = ContextualKofnGame.context_idx
    action_idx = ContextualKofnGame.action_idx
    successor_state_idx = action_idx + 1

    @classmethod
    def environment(cls, mixture_constraint_weights, root_probs,
                    transition_model, sample_rewards, **kwargs):
        def play_game(policy):
            return cls(
                mixture_constraint_weights, root_probs, transition_model,
                sample_rewards(), policy, **kwargs).kofn_utility

        return play_game

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


class UncertainRewardDiscountedContinuingKofnTrainer(
        UncertainRewardKofnTrainer):
    def __init__(self,
                 root_probs,
                 transitions,
                 *args,
                 gamma=0.99,
                 threshold=1e-10,
                 max_num_iterations=100,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.root_probs = root_probs
        self.transitions = transitions
        self.gamma = gamma
        self.threshold = threshold
        self.max_num_iterations = max_num_iterations

    def _eval_game(self, rewards, policy):
        return UncertainRewardDiscountedContinuingKofnGame(
            self.prob_ith_element_is_sampled,
            self.root_probs,
            self.transitions,
            rewards,
            policy,
            gamma=self.discount,
            threshold=self.threshold,
            max_num_iterations=self.max_num_iterations)
