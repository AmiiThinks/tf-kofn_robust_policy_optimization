import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_kofn_robust_policy_optimization import cache
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    dual_action_value_policy_evaluation_op, \
    state_successor_policy_evaluation_op
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
        return UncertainRewardDiscountedContinuingKofnEnv(*args, **kwargs)

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
        action_values = dual_action_value_policy_evaluation_op(
            self.transition_model_op,
            self.policy,
            self.reward_models_op,
            gamma=self.gamma)

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


class UncertainRewardDiscountedContinuingKofnEnv(object):
    def __init__(self, mixture_constraint_weights, root_probs,
                 transition_model, sample_rewards, **kwargs):
        self.mixture_constraint_weights = mixture_constraint_weights
        self.root_probs = root_probs
        self.transition_model = tf.convert_to_tensor(transition_model)

        num_states = self.transition_model.shape[0].value
        num_actions = self.transition_model.shape[1].value

        self.sample_rewards = sample_rewards
        self.H = ResourceVariable(
            tf.constant(
                1.0 / (num_states * num_actions),
                shape=(num_states * num_actions, num_states * num_actions)))
        self.game = None
        self.kwargs = kwargs

    def __call__(self, policy):
        self.game = UncertainRewardDiscountedContinuingKofnGame(
            self.mixture_constraint_weights,
            self.root_probs,
            self.transition_model,
            self.sample_rewards(),
            policy,
            H_0=self.H,
            **self.kwargs)
        return self.game.kofn_utility

    def update(self):
        return self.H.assign(self.game.state_action_successor_rep)


class UncertainRewardDiscountedContinuingKofnEvEnv(object):
    def __init__(self,
                 mixture_constraint_weights,
                 root_probs,
                 transition_model,
                 sample_rewards,
                 gamma=0.9,
                 **kwargs):
        self.mixture_constraint_weights = mixture_constraint_weights
        self.root_probs = root_probs
        self.transition_model = tf.convert_to_tensor(transition_model)

        num_states = self.transition_model.shape[0].value

        self.sample_rewards = sample_rewards
        self.M = ResourceVariable(
            tf.constant(1.0 / num_states, shape=(num_states, num_states)))
        self.game = None
        self.gamma = gamma
        self.kwargs = kwargs

    def __call__(self, policy):
        self.next_M = state_successor_policy_evaluation_op(
            self.transition_model,
            policy,
            M_0=self.M,
            gamma=self.gamma,
            **self.kwargs)

        r = self.sample_rewards()
        if len(r.shape) > 2:
            M = tf.expand_dims(self.next_M, axis=-1)
            policy = tf.expand_dims(policy, axis=-1)
        weighted_rewards = tf.reduce_sum(r * policy, axis=1, keepdims=True)
        state_values = tf.tensordot(M, weighted_rewards, axes=[[1], [0]])

        if self.gamma < 1:
            state_values = state_values / (1.0 - self.gamma)
        kofn_evs_and_weights = KofnEvsAndWeights(
            tf.squeeze(state_values),
            self.mixture_constraint_weights,
            context_weights=self.root_probs)
        return kofn_evs_and_weights.ev

    def update(self):
        return self.M.assign(self.next_M)


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
