import tensorflow as tf
import numpy as np
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


class PrUncertainMdp(object):
    def __init__(self, horizon, num_states, num_actions, name=None):
        name = type(self).__name__ if name is None else name
        self.root = tf.placeholder_with_default(
            l1_projection_to_simplex(
                tf.random_uniform((num_states,))
            ),
            (num_states,),
            name='{}_root'.format(name)
        )
        self.horizon = horizon
        self.transition_model = tf.placeholder(
            tf.float32,
            (num_states, num_actions, num_states),
            name='{}_transition_model'.format(name)
        )
        self.rewards = tf.placeholder(
            tf.float32,
            (num_states, num_actions, num_states),
            name='{}_rewards'.format(name)
        )
        self.sequences = tf.Variable(
            tf.constant(
                0.0,
                shape=(
                    self.num_pr_prefixes(horizon - 1),
                    num_states,
                    num_actions,
                    num_states
                )
            ),
            name='{}_sequences'.format(name)
        )
        default_strat = tf.ones(
            (self.num_pr_sequences(self.horizon - 1), self.num_actions())
        )
        self.strat = tf.placeholder_with_default(
            default_strat,
            default_strat.shape,
            name='{}_strat'.format(name)
        )
        self.unroll = self._unroll()
        self.expected_value = tf.reduce_sum(self.sequences * self.rewards)
        self.best_response, self.br_value = self._best_response()

    def num_actions(self):
        return self.transition_model.shape[1].value

    def num_states(self):
        return self.transition_model.shape[0].value

    def num_pr_sequences(self, t):
        return num_pr_sequences(t, self.num_states(), self.num_actions())

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())

    def _sequences_at_timestep(self, t):
        if t < 1:
            return tf.reshape(self.root, (1, 1, self.root.shape[0].value))
        else:
            n = self.num_pr_prefixes(t - 2)
            next_n = self.num_pr_prefixes(t - 1)
            return self.sequences[n:next_n, :, :, :]

    def _updated_sequences_at_timestep(self, t, **kwargs):
        strat = self.strat[
            self.num_pr_sequences(t - 1):self.num_pr_sequences(t),
            :
        ]
        if t < 1:
            prob = prob_next_sequence_state_action_and_next_state(
                self.transition_model,
                tf.reshape(self.root, (1, 1, self.root.shape[0].value)),
                strat=strat,
                num_actions=self.num_actions()
            )
            next_n = 0
        else:
            n = self.num_pr_prefixes(t - 2)
            next_n = self.num_pr_prefixes(t - 1)
            prob = prob_next_sequence_state_action_and_next_state(
                self.transition_model,
                self.sequences[n:next_n, :, :, :],
                strat=strat,
                num_actions=self.num_actions()
            )
        return tf.assign(
            self.sequences[next_n:next_n + prob.shape[0].value, :, :, :],
            prob,
            **kwargs
        )

    def _unroll(self):
        last_sequence_prob_update = None
        for t in range(self.horizon):
            if last_sequence_prob_update is None:
                last_sequence_prob_update = (
                    self._updated_sequences_at_timestep(t)
                )
            else:
                with tf.control_dependencies([last_sequence_prob_update]):
                    last_sequence_prob_update = (
                        self._updated_sequences_at_timestep(t)
                    )
        return last_sequence_prob_update

    def _best_response(self):
        strat_pieces = []
        current_cf_state_values = None

        action_rewards_weighted_by_chance = tf.squeeze(
            tf.reduce_sum(
                self.sequences * self.rewards,
                axis=3
            )
        )
        for t in range(self.horizon - 1, -1, -1):
            n = self.num_pr_prefixes(t - 1)
            next_n = self.num_pr_prefixes(t)
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
                        current_rewards.shape
                    )
                )
            br_indices = tf.expand_dims(
                tf.argmax(
                    tf.reshape(
                        current_cf_action_values,
                        (
                            num_sequences * self.num_states(),
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

            strat_pieces.append(
                tf.reshape(
                    tf.scatter_nd(
                        scatter_indices,
                        tf.ones(br_indices.shape[:1]),
                        shape=(
                            num_sequences * self.num_states(),
                            self.num_actions()
                        )
                    ),
                    (num_sequences, self.num_states(), self.num_actions())
                )
            )
            with tf.control_dependencies([strat_pieces[-1]]):
                current_cf_state_values = tf.expand_dims(
                    # TODO Should be able to do this with tensordot
                    # but it didn't work the first way I tried.
                    tf.reduce_sum(
                        strat_pieces[-1] * current_cf_action_values,
                        axis=2
                    ),
                    axis=2
                )
        br_ev = tf.reduce_sum(current_cf_state_values)
        strat_pieces.reverse()
        strat = tf.concat(strat_pieces, axis=0)
        return (strat, br_ev)
