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

    def num_pr_sequences(self, t):
        return num_pr_sequences(
            t,
            self.num_states(),
            self.num_actions()
        )

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())


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
    def __init__(
        self,
        mdp,
        initial_state_distribution=None,
        name=None
    ):
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
                    mdp.num_pr_prefixes(mdp.horizon - 1),
                    mdp.num_states(),
                    mdp.num_actions(),
                    mdp.num_states()
                )
            ),
            name=type(self).__name__ if name is None else name
        )

    def sequences_at_timestep(self, t):
        if t < 1:
            return tf.reshape(self.root, (1, 1, self.root.shape[0].value))
        else:
            n = self.mdp.num_pr_prefixes(t - 2)
            next_n = self.mdp.num_pr_prefixes(t - 1)
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
            n = self.mdp.num_pr_prefixes(t - 2)
            next_n = self.mdp.num_pr_prefixes(t - 1)
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

    def unroll(self, strat=None):
        last_sequence_prob_update = None
        strat_for_current_sequences = None
        for t in range(self.mdp.horizon):
            if strat is not None:
                strat_for_current_sequences = strat[
                    self.mdp.num_pr_sequences(t - 1):
                    self.mdp.num_pr_sequences(t),
                    :
                ]
            if last_sequence_prob_update is None:
                last_sequence_prob_update = (
                    self.updated_sequences_at_timestep(
                        t,
                        strat=strat_for_current_sequences
                    )
                )
            else:
                with tf.control_dependencies([last_sequence_prob_update]):
                    last_sequence_prob_update = (
                        self.updated_sequences_at_timestep(
                            t,
                            strat=strat_for_current_sequences
                        )
                    )
        return last_sequence_prob_update

    def expected_value(self, strat):
        with tf.control_dependencies([self.unroll(strat)]):
            n = tf.reduce_sum(self.sequences * self.mdp.rewards)
        return n

    def best_response(self):
        strat = tf.Variable(
            tf.zeros(
                (
                    self.mdp.num_pr_prefixes(self.mdp.horizon - 1),
                    self.mdp.num_states(),
                    self.mdp.num_actions()
                )
            )
        )
        strat.initializer.run()
        current_cf_state_values = None

        with tf.control_dependencies([self.unroll()]):
            action_rewards_weighted_by_chance = tf.squeeze(
                tf.reduce_sum(
                    self.sequences * self.mdp.rewards,
                    axis=3
                )
            )
            for t in range(self.mdp.horizon - 1, -1, -1):
                n = self.mdp.num_pr_prefixes(t - 1)
                next_n = self.mdp.num_pr_prefixes(t)
                current_rewards = action_rewards_weighted_by_chance[
                    n:next_n,
                    :,
                    :
                ]
                num_sequences = next_n - n
                if current_cf_state_values is None:
                    current_cf_action_values = current_rewards
                else:
                    current_cf_action_values = (
                        current_rewards +
                        tf.reshape(
                            tf.reduce_sum(
                                current_cf_state_values,
                                axis=1
                            ),
                            [
                                -1,
                                self.mdp.num_states(),
                                self.mdp.num_actions()
                            ]
                        )
                    )
                br_indices = tf.expand_dims(
                    tf.argmax(
                        tf.reshape(
                            current_cf_action_values,
                            (
                                num_sequences * self.mdp.num_states(),
                                current_cf_action_values.shape[-1].value
                            )
                        ),
                        axis=1
                    ),
                    dim=1
                )
                scatter_indices = tf.concat(
                    [
                        tf.expand_dims(
                            tf.range(
                                br_indices.shape[0].value,
                                dtype=tf.int64
                            ),
                            dim=1
                        ),
                        br_indices
                    ],
                    axis=1
                )
                strat_update = tf.assign(
                    strat[n:next_n, :, :],
                    tf.reshape(
                        tf.scatter_nd(
                            scatter_indices,
                            tf.ones(br_indices.shape[:1]),
                            shape=(
                                num_sequences * self.mdp.num_states(),
                                self.mdp.num_actions()
                            )
                        ),
                        (
                            num_sequences,
                            self.mdp.num_states(),
                            self.mdp.num_actions()
                        )
                    )
                )
                with tf.control_dependencies([strat_update]):
                    current_cf_state_values = tf.expand_dims(
                        # TODO Should be able to do this with tensordot
                        # but it didn't work the first way I tried.
                        tf.reduce_sum(
                            (
                                strat[n:next_n, :, :] *
                                current_cf_action_values
                            ),
                            axis=2
                        ),
                        axis=2
                    )
        br_ev = tf.reduce_sum(current_cf_state_values)
        with tf.control_dependencies([br_ev]):
            final_strat = tf.convert_to_tensor(strat)
        return (final_strat, br_ev)
