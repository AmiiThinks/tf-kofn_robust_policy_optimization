import tensorflow as tf
from .probability_utils import prob_next_state, prob_state_given_action
from .reward_utils import reward_distribution
from .sequence_utils import num_pr_sequences, \
    prob_next_sequence_state_action_and_next_state
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
            mdp.rewards,
            *args,
            **kwargs
        )

    def __init__(self, horizon, *args, **kwargs):
        super(FixedHorizonMdp, self).__init__(*args, **kwargs)
        assert(horizon > 0)
        self.horizon = horizon


class IrMdpState(object):
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


class PrMdpState(object):
    def __init__(self, mdp, initial_state_distribution=None):
        self.mdp = mdp
        if initial_state_distribution is None:
            initial_state_distribution = l1_projection_to_simplex(
                tf.random_uniform((self.mdp.num_states(),))
            )
        self.root = initial_state_distribution
        self.sequences = tf.Variable(
            tf.constant(
                0.0,
                shape=(
                    int(
                        num_pr_sequences(
                            mdp.horizon - 1,
                            mdp.num_states(),
                            mdp.num_actions()
                        ) / mdp.num_states()
                    ),
                    mdp.num_states(),
                    mdp.num_actions(),
                    mdp.num_states()
                )
            )
        )

    def sequences_at_timestep(self, t):
        if t < 1:
            return tf.reshape(self.root, (1, 1, self.root.shape[0].value))
        else:
            n = int(
                num_pr_sequences(
                    t - 2,
                    self.mdp.num_states(),
                    self.mdp.num_actions()
                ) / self.mdp.num_states()
            )
            next_n = int(
                num_pr_sequences(
                    t - 1,
                    self.mdp.num_states(),
                    self.mdp.num_actions()
                ) / self.mdp.num_states()
            )
            return self.sequences[n:next_n, :, :, :]

    def updated_sequences_at_timestep(self, t, strat=None, **kwargs):
        if t < 1:
            prob = prob_next_sequence_state_action_and_next_state(
                self.mdp.transition_model,
                tf.reshape(self.root, (1, 1, self.root.shape[0].value)),
                strat=strat,
                num_actions=self.mdp.num_actions()
            )
            next_n = 0
        else:
            n = int(
                num_pr_sequences(
                    t - 2,
                    self.mdp.num_states(),
                    self.mdp.num_actions()
                ) / self.mdp.num_states()
            )
            next_n = int(
                num_pr_sequences(
                    t - 1,
                    self.mdp.num_states(),
                    self.mdp.num_actions()
                ) / self.mdp.num_states()
            )
            prob = prob_next_sequence_state_action_and_next_state(
                self.mdp.transition_model,
                self.sequences[n:next_n, :, :, :],
                strat=strat,
                num_actions=self.mdp.num_actions()
            )
        return tf.assign(
            self.sequences[next_n:next_n + prob.shape[0].value, :, :, :],
            prob,
            **kwargs
        )

    def expected_value(self, strat):
        last_sequence_prob_update = strat
        for t in range(self.mdp.horizon):
            with tf.control_dependencies([last_sequence_prob_update]):
                last_sequence_prob_update = self.updated_sequences_at_timestep(
                    t,
                    strat=strat[
                        num_pr_sequences(
                            t - 1,
                            self.mdp.num_states(),
                            self.mdp.num_actions()
                        ):num_pr_sequences(
                            t,
                            self.mdp.num_states(),
                            self.mdp.num_actions()
                        ),
                        :
                    ]
                )
        with tf.control_dependencies([last_sequence_prob_update]):
            n = tf.reduce_sum(self.sequences * self.mdp.rewards)
        return n
