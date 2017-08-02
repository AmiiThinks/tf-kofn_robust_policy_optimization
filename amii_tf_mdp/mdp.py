import tensorflow as tf
from .probability_utils import prob_next_state
from amii_tf_nn.projection import l1_projection_to_simplex


class Mdp(object):
    def __init__(self, transition_model, rewards, horizon):
        '''

        Params:
            - transition_model: rank-3 Tensor (states by actions by states).
              Element i, a, j is the probility of transitioning from the state
              i to the state j after taking action a.
            - rewards: rank-1 Tensor (states). Maps states to rewards.
        '''
        assert(
            transition_model.shape[0].value == transition_model.shape[2].value
        )
        assert(transition_model.shape[0].value == rewards.shape[0].value)
        self.transition_model = transition_model
        self.rewards = rewards
        assert(horizon > 0)
        self.horizon = horizon

    def num_actions(self):
        return self.transition_model.shape[0].value

    def num_states(self):
        return self.rewards.shape[0].value


class MdpState(object):
    def __init__(self, mdp, initial_state_distribution=None):
        self.mdp = mdp
        self.t = 0
        if initial_state_distribution is None:
            initial_state_distribution = l1_projection_to_simplex(
                tf.random_uniform(self.mdp.rewards.shape)
            )
        self.state_distribution = tf.Variable(initial_state_distribution)

    def updated(self, strat=None, **kwargs):
        node = tf.assign(
            self.state_distribution,
            prob_next_state(
                self.mdp.transition_model,
                self.state_distribution, strat
            ),
            **kwargs
        )
        self.t += 1
        return node

    def horizon_has_been_reached(self):
        return self.t >= self.mdp.horizon

    def rewards(self):
        return self.state_distribution * self.mdp.rewards
