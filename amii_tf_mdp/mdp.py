import tensorflow as tf
from .probability_utils import prob_next_state, prob_state_given_action
from .reward_utils import reward_distribution
from amii_tf_nn.projection import l1_projection_to_simplex


class Mdp(object):
    def __init__(self, transition_model, rewards):
        '''

        Params:
            - transition_model: rank-3 Tensor (states by actions by states).
              Element i, a, j is the probility of transitioning from the state
              i to the state j after taking action a.
            - rewards: rank-3 Tensor (states by actions by states). Maps
              (state, action, state)-tuples to rewards.
        '''
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


class FixedHorizonMdp(Mdp):
    @classmethod
    def upgrade(cls, horizon, mdp, *args, **kwargs):
        return cls(
            horizon,
            mdp.transition_model,
            mdp.reward_distribution,
            *args,
            **kwargs
        )

    def __init__(self, horizon, *args, **kwargs):
        super(FixedHorizonMdp, self).__init__(*args, **kwargs)
        assert(horizon > 0)
        self.horizon = horizon


class MdpState(object):
    def __init__(self, mdp, initial_state_distribution=None):
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
