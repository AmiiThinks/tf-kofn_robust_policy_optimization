import tensorflow as tf
from .probability_utils import prob_next_state, prob_state_given_action
from .reward_utils import reward_distribution
from .sequence_utils import num_pr_sequences, \
    prob_next_sequence_state_action_and_next_state
from amii_tf_nn.projection import l1_projection_to_simplex


class MdpAnchor(object):
    @staticmethod
    def mean(mdp_anchors):
        z = float(len(mdp_anchors))
        mean_transition_model = mdp_anchors[0].transition_model / z
        mean_rewards = mdp_anchors[0].rewards / z
        for i in range(1, len(mdp_anchors)):
            mean_transition_model = (
                mean_transition_model + mdp_anchors[i].transition_model / z
            )
            mean_rewards = mean_rewards + mdp_anchors[i].rewards / z
        return MdpAnchor(mean_transition_model, mean_rewards)

    # TODO Add initial state distribution.
    def __init__(self, transition_model, rewards):
        '''

        Params:
            - transition_model: rank-3 Tensor (states by actions by states).
              Element i, a, j is the probility of transitioning from the state
              i to the state j after taking action a.
            - rewards: rank-3 Tensor (states by actions by states). Maps
              (state, action, state)-tuples to rewards.
        '''
        transition_model = tf.convert_to_tensor(transition_model)
        rewards = tf.convert_to_tensor(rewards)
        assert(
            transition_model.shape[0].value == transition_model.shape[2].value
        )
        assert(transition_model.shape[0].value == rewards.shape[0].value)
        assert(transition_model.shape[1].value == rewards.shape[1].value)
        assert(transition_model.shape[2].value == rewards.shape[2].value)
        self.transition_model = transition_model
        self.rewards = rewards

    def num_actions(self):
        return self.transition_model.shape[1].value

    def num_states(self):
        return self.rewards.shape[0].value

    def reward_distribution(self, state, strat=None):
        return reward_distribution(self.rewards, state, strat=strat)

    def reward(self, state, next_state, strat=None):
        return tf.tensordot(
            self.reward_distribution(state, strat=strat),
            next_state,
            axes=1
        )

    def num_pr_sequences(self, t):
        return num_pr_sequences(
            t,
            self.num_states(),
            self.num_actions()
        )

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())


class IrUncertainMdp(object):
    '''
    TODO This class has been neglected. It still has useful pieces so I
    don't want to delete it, but don't use it.
    '''
    def __init__(self, mdp, initial_state_distribution=None):
        assert(False)
        self.mdp = mdp
        self.t = 0
        if initial_state_distribution is None:
            initial_state_distribution = l1_projection_to_simplex(
                tf.random_uniform((self.mdp.num_states(),))
            )
        self.state_distribution = tf.Variable(initial_state_distribution)

    def prob_state_given_action(self, strat=None):
        return prob_state_given_action(
            self.state_distribution,
            strat=strat,
            num_actions=self.mdp.num_actions()
        )

    def updated(self, strat=None, **kwargs):
        next_states = prob_next_state(
            self.mdp.transition_model,
            self.state_distribution,
            strat=strat
        )
        reward_distribution_node = self.mdp.reward(
            self.state_distribution,
            next_states,
            strat=strat
        )
        state_distribution_node = tf.assign(
            self.state_distribution,
            next_states,
            **kwargs
        )
        self.t += 1
        return (reward_distribution_node, state_distribution_node)

    def horizon_has_been_reached(self):
        return self.t >= self.mdp.horizon
