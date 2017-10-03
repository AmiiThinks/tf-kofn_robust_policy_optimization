import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from .utils.sequence import num_pr_sequences, \
    prob_next_sequence_state_action_and_next_state
from .utils.tf_node import UnboundTfNode


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

        default_strat = tf.ones(
            (self.num_pr_sequences(self.horizon - 1), self.num_actions())
        )
        self.strat = tf.placeholder_with_default(
            default_strat,
            default_strat.shape,
            name='{}_strat'.format(name)
        )

        def sequence_feed_dict_generator(
            transition_model,
            root=None,
            strat=None
        ):
            d = {self.transition_model: transition_model}
            if root is not None: d[self.root] = root
            if strat is not None: d[self.strat] = strat
            return d
        self.unbound_sequences = UnboundTfNode(
            self._unroll(),
            sequence_feed_dict_generator
        )

        def expected_value_feed_dict_generator(
            transition_model,
            rewards,
            strat,
            root=None,
        ):
            d = sequence_feed_dict_generator(
                transition_model,
                root=root,
                strat=strat
            )
            d[self.rewards] = rewards
            return d

        self.unbound_expected_value = UnboundTfNode(
            self._expected_value(),
            expected_value_feed_dict_generator
        )
        best_response, br_value = self._best_response()

        def best_resonse_feed_dict_generator(
            transition_model,
            rewards,
            root=None
        ):
            d = sequence_feed_dict_generator(transition_model, root=root)
            d[self.rewards] = rewards
            return d

        self.unbound_best_response = UnboundTfNode(
            best_response,
            best_resonse_feed_dict_generator
        )
        self.unbound_br_value = UnboundTfNode(
            br_value,
            best_resonse_feed_dict_generator
        )

    def bound_sequences_node(self, transition_model, root=None, strat=None):
        return self.unbound_sequences(transition_model, root=root, strat=strat)

    def bound_expected_value_node(
        self,
        transition_model,
        rewards,
        strat,
        root=None
    ):
        return self.unbound_expected_value(
            transition_model,
            rewards,
            strat,
            root=root
        )

    def bound_best_response_node(self, transition_model, rewards, root=None):
        return self.unbound_best_response(transition_model, rewards, root=root)

    def bound_br_value_node(self, transition_model, rewards, root=None):
        return self.unbound_br_value(transition_model, rewards, root=root)

    def num_actions(self):
        return self.transition_model.shape[1].value

    def num_states(self):
        return self.transition_model.shape[0].value

    def num_pr_sequences(self, t):
        return num_pr_sequences(t, self.num_states(), self.num_actions())

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())

    def _unroll(self):
        sequence_blocks = [
            prob_next_sequence_state_action_and_next_state(
                self.transition_model,
                tf.reshape(self.root, (1, 1, self.root.shape[0].value)),
                num_actions=self.num_actions()
            )
        ]
        for t in range(1, self.horizon):
            sequence_blocks.append(
                prob_next_sequence_state_action_and_next_state(
                    self.transition_model,
                    sequence_blocks[-1],
                    num_actions=self.num_actions()
                )
            )
        return tf.concat(sequence_blocks, axis=0)

    def _expected_value(self):
        current_cf_state_values = None
        strat = tf.reshape(
            self.strat,
            [
                self.num_pr_prefixes(self.horizon - 1),
                self.num_states(),
                self.num_actions()
            ]
        )
        action_rewards_weighted_by_chance = tf.squeeze(
            tf.reduce_sum(
                self.unbound_sequences.component * self.rewards,
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
        return tf.reduce_sum(current_cf_state_values)

    def _best_response(self):
        strat_pieces = []
        current_cf_state_values = None

        action_rewards_weighted_by_chance = tf.squeeze(
            tf.reduce_sum(
                self.unbound_sequences.component * self.rewards,
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
