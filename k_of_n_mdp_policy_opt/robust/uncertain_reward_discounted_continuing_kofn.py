import tensorflow as tf
from k_of_n_mdp_policy_opt.discounted_mdp import inst_regrets_op, associated_ops, \
    value_ops, state_successor_policy_evaluation_op
from k_of_n_mdp_policy_opt.utils.tensor import row_normalize_op


class UncertainRewardDiscountedContinuingKofn(object):
    def __init__(self, config, root_op, transition_model_op, reward_models_op,
                 gamma, cap_negative_advantages=False):
        if reward_models_op.shape[1].value != config.num_sampled_mdps():
            print(reward_models_op.shape[1].value)
            print(config.num_sampled_mdps())
        assert reward_models_op.shape[1].value == config.num_sampled_mdps()
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

        self.mdp_weights_op = self.config.mdp_weights_op(self.evs_op)

        self.ev_op = tf.squeeze(
            tf.transpose(self.evs_op) @ self.mdp_weights_op)

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

        self.training_mdp_weights_op = self.config.mdp_weights_op(
            self.training_evs_op)

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
        self.policy_op = row_normalize_op(self.advantages_op)

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
            tf.transpose(self.state_successor_op), root_op
        )

    def num_states(self): return self.root_op.shape[0].value

    def num_actions(self):
        return int(self.transition_model_op.shape[0].value / self.num_states())

    def name(self): return self.config.name()
