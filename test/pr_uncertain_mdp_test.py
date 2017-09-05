import tensorflow as tf
import numpy as np
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp
from amii_tf_mdp.sequence_utils import num_pr_sequences
from amii_tf_nn.projection import l1_projection_to_simplex


class PrUncertainMdpTest(tf.test.TestCase):
    def test_update(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        tf.transpose(
                            tf.random_normal(
                                (num_states, num_actions, num_states)
                            )
                        )
                    )
                )
            )
            x_root = sess.run(
                l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
            )

            patient = PrUncertainMdp(horizon, num_states, num_actions)
            n = patient.bound_sequences_node(transition_model, root=x_root)

            x_update_1 = [
                [
                    [
                        [0.15077846, 0., 0.01588821],
                        [0.02819305, 0.13847362, 0.]
                    ],
                    [
                        [0.04897123, 0.28436211, 0.],
                        [0., 0.33333334, 0.]
                    ],
                    [
                        [0., 0.33023661, 0.16976337],
                        [0.5, 0., 0.]
                    ]
                ]
            ]

            self.assertAllClose(
                tf.constant(x_update_1).eval(),
                n.run(sess)[:1, :, :, :]
            )
            # TODO Check that the rest of the sequences were updated properly.

    def test_expected_value(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        tf.zeros((num_states, num_actions, num_states))
                    )
                )
            )
            rewards = np.ones((num_states, num_actions, num_states))
            x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))

            patient = PrUncertainMdp(horizon, num_states, num_actions)

            uniform_random_strat = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        tf.zeros(
                            (
                                num_actions,
                                num_pr_sequences(
                                    horizon - 1,
                                    num_states,
                                    num_actions
                                )
                            )
                        )
                    )
                )
            )

            n = patient.bound_expected_value_node(
                transition_model,
                rewards,
                uniform_random_strat,
                root=x_root
            )

            self.assertAlmostEqual(2.0, n.run(sess), places=6)

    def test_expected_value_can_be_run_repeatedly(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        np.random.normal(
                            size=(
                                num_states,
                                num_actions,
                                num_states
                            )
                        )
                    )
                )
            )
            rewards = sess.run(
                np.random.normal(
                    loc=-1.0,
                    scale=1.0,
                    size=(num_states, num_actions, num_states)
                ) *
                tf.transpose(
                    l1_projection_to_simplex(
                        np.random.normal(
                            size=(
                                num_states,
                                num_actions,
                                num_states
                            ),
                            scale=5.0
                        )
                    )
                )
            )
            x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))
            uniform_random_strat = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        tf.zeros(
                            (
                                num_actions,
                                num_pr_sequences(
                                    horizon - 1,
                                    num_states,
                                    num_actions
                                )
                            )
                        )
                    )
                )
            )

            x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))

            patient = PrUncertainMdp(horizon, num_states, num_actions)
            n = patient.bound_expected_value_node(
                transition_model,
                rewards,
                uniform_random_strat,
                root=x_root
            )

            self.assertAlmostEqual(-0.62523419, n.run(sess))
            self.assertAlmostEqual(-0.62523419, n.run(sess))

    def test_best_response(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = sess.run(
                tf.transpose(
                    l1_projection_to_simplex(
                        np.random.normal(
                            size=(
                                num_states,
                                num_actions,
                                num_states
                            )
                        )
                    )
                )
            )
            rewards = sess.run(
                np.random.normal(
                    loc=-1.0,
                    scale=1.0,
                    size=(num_states, num_actions, num_states)
                ) *
                tf.transpose(
                    l1_projection_to_simplex(
                        np.random.normal(
                            size=(
                                num_states,
                                num_actions,
                                num_states
                            ),
                            scale=5.0
                        )
                    )
                )
            )
            x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))

            patient = PrUncertainMdp(horizon, num_states, num_actions)

            n = patient.bound_br_value_node(
                transition_model,
                rewards,
                root=x_root
            )
            self.assertAlmostEqual(-0.35074115, n.run(sess))

    # def test_root_counterfactual_value_is_ev(self):
    #     with self.test_session() as sess:
    #         horizon = 2
    #         num_states = 3
    #         num_actions = 2
    #
    #         transition_model = sess.run(
    #             tf.transpose(
    #                 l1_projection_to_simplex(
    #                     tf.zeros((num_states, num_actions, num_states))
    #                 )
    #             )
    #         )
    #         rewards = np.ones((num_states, num_actions, num_states))
    #         uniform_random_strat = tf.transpose(
    #             l1_projection_to_simplex(
    #                 tf.zeros(
    #                     (
    #                         num_actions,
    #                         num_pr_sequences(
    #                             horizon - 1,
    #                             num_states,
    #                             num_actions
    #                         )
    #                     )
    #                 )
    #             )
    #         )
    #         uniform_random_strat = tf.reshape(
    #             uniform_random_strat,
    #             (-1, num_states, num_actions)
    #         )
    #         x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))
    #
    #         patient = PrUncertainMdp(horizon, num_states, num_actions)
    #         sess.run(tf.global_variables_initializer())
    #
    #         sess.run(
    #             patient.unroll,
    #             feed_dict={
    #                 patient.root: x_root,
    #                 patient.transition_model: transition_model
    #             }
    #         )
    #
    #         action_rewards_weighted_by_chance = tf.squeeze(
    #             tf.reduce_sum(patient.sequences * rewards, axis=3)
    #         )
    #         current_cf_state_values = None
    #         for t in range(horizon - 1, -1, -1):
    #             n = int(
    #                 num_pr_sequences(
    #                     t - 1,
    #                     num_states,
    #                     num_actions
    #                 ) / num_states
    #             )
    #             next_n = int(
    #                 num_pr_sequences(
    #                     t,
    #                     num_states,
    #                     num_actions
    #                 ) / num_states
    #             )
    #             if current_cf_state_values is None:
    #                 current_cf_action_values = (
    #                     action_rewards_weighted_by_chance[n:next_n, :, :]
    #                 )
    #             else:
    #                 current_cf_action_values = (
    #                     action_rewards_weighted_by_chance[n:next_n, :, :] +
    #                     tf.reshape(
    #                         tf.reduce_sum(current_cf_state_values, axis=1),
    #                         [-1, num_states, num_actions]
    #                     )
    #                 )
    #
    #             current_cf_state_values = tf.expand_dims(
    #                 tf.reduce_sum(
    #                     (
    #                         uniform_random_strat[n:next_n, :, :] *
    #                         current_cf_action_values
    #                     ),
    #                     axis=2
    #                 ),
    #                 axis=2
    #             )
    #         self.assertAlmostEqual(
    #             2.0,
    #             tf.reduce_sum(current_cf_state_values).eval(),
    #             places=6
    #         )


if __name__ == '__main__':
    tf.test.main()
