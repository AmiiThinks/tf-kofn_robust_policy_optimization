import tensorflow as tf
import numpy as np
from amii_tf_mdp.mdp import PrMdpState, FixedHorizonMdp
from amii_tf_mdp.sequence_utils import num_pr_sequences
from amii_tf_nn.projection import l1_projection_to_simplex


class PrMdpStateTest(tf.test.TestCase):
    def test_initializer(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
                l1_projection_to_simplex(
                    tf.transpose(
                        tf.random_normal(
                            (num_states, num_actions, num_states)
                        )
                    )
                )
            )
            rewards = tf.random_normal((num_states, num_actions, num_states))

            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            patient = PrMdpState(mdp)
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(
                tf.constant([0.61768991, 0.21680082, 0.16550928]).eval(),
                patient.root.eval()
            )
            self.assertAllEqual(
                tf.zeros(
                    (
                        1 + num_states * num_actions,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences.eval()
            )

            x_root = l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
            patient = PrMdpState(mdp, x_root)
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(x_root.eval(), patient.root.eval())
            self.assertAllEqual(
                tf.zeros(
                    (
                        1 + num_states * num_actions,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences.eval()
            )

    def test_update(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
                l1_projection_to_simplex(
                    tf.transpose(
                        tf.random_normal(
                            (num_states, num_actions, num_states)
                        )
                    )
                )
            )
            rewards = tf.random_normal(
                (num_states, num_actions, num_states)
            )

            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            x_root = l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
            patient = PrMdpState(mdp, x_root)
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(x_root.eval(), patient.root.eval())
            self.assertAllEqual(
                tf.zeros(
                    (
                        1 + num_states * num_actions,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences.eval()
            )

            patient.updated_sequences_at_timestep(0).eval()

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
                tf.concat(
                    (
                        x_update_1,
                        tf.zeros(
                            (
                                num_states * num_actions,
                                num_states,
                                num_actions,
                                num_states
                            )
                        )
                    ),
                    axis=0
                ).eval(),
                patient.sequences.eval()
            )

            patient.updated_sequences_at_timestep(1).eval()

            self.assertAllClose(
                tf.constant(x_update_1).eval(),
                patient.sequences[:1, :, :, :].eval()
            )
            # TODO Check that the rest of the sequences were updated properly.

    def test_get(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
                l1_projection_to_simplex(
                    tf.transpose(
                        tf.random_normal(
                            (num_states, num_actions, num_states)
                        )
                    )
                )
            )
            rewards = tf.random_normal(
                (num_states, num_actions, num_states)
            )

            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            x_root = l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
            patient = PrMdpState(mdp, x_root)
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(x_root.eval(), patient.root.eval())
            self.assertAllEqual(
                tf.zeros(
                    (
                        1 + num_states * num_actions,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences.eval()
            )
            self.assertAllEqual(
                tf.reshape(x_root, (1, 1, num_states)).eval(),
                patient.sequences_at_timestep(0).eval()
            )
            self.assertAllEqual(
                tf.zeros(
                    (
                        1,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences_at_timestep(1).eval()
            )
            self.assertAllEqual(
                tf.zeros(
                    (
                        num_states * num_actions,
                        num_states,
                        num_actions,
                        num_states
                    )
                ).eval(),
                patient.sequences_at_timestep(2).eval()
            )

            patient.updated_sequences_at_timestep(0).eval()

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
                patient.sequences_at_timestep(1).eval()
            )

            patient.updated_sequences_at_timestep(1).eval()

            self.assertAllClose(
                tf.constant(x_update_1).eval(),
                patient.sequences_at_timestep(1).eval()
            )
            # TODO Check that the rest of the sequences were updated properly.

    def test_expected_value(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
                l1_projection_to_simplex(
                    tf.zeros((num_states, num_actions, num_states))
                )
            )
            rewards = tf.ones((num_states, num_actions, num_states))

            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            x_root = l1_projection_to_simplex(tf.constant([1, 1, 1.0]))
            patient = PrMdpState(mdp, x_root)

            uniform_random_strat = tf.transpose(
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
            sess.run(tf.global_variables_initializer())
            self.assertAlmostEqual(
                2.0,
                patient.expected_value(uniform_random_strat).eval(),
                places=6
            )

    def test_expected_value_can_be_run_repeatedly(self):
        with self.test_session():
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
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
            rewards = (
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
            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            x_root = l1_projection_to_simplex(
                tf.constant([1, 1, 1.0])
            )
            uniform_random_strat = tf.transpose(
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

            patient = PrMdpState(mdp, x_root)
            self.assertAllEqual(x_root, patient.root)

            patient.sequences.initializer.run()
            ev = patient.expected_value(uniform_random_strat)
            self.assertAlmostEqual(-0.62523419, ev.eval())
            self.assertAlmostEqual(-0.62523419, ev.eval())

    def test_root_counterfactual_value_is_ev(self):
        with self.test_session() as sess:
            horizon = 2
            num_states = 3
            num_actions = 2

            transition_model = tf.transpose(
                l1_projection_to_simplex(
                    tf.zeros((num_states, num_actions, num_states))
                )
            )
            rewards = tf.ones((num_states, num_actions, num_states))

            mdp = FixedHorizonMdp(horizon, transition_model, rewards)
            x_root = l1_projection_to_simplex(tf.constant([1, 1, 1.0]))
            patient = PrMdpState(mdp, x_root)

            uniform_random_strat = tf.transpose(
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

            sess.run(tf.global_variables_initializer())
            sess.run(patient.unroll())
            uniform_random_strat = tf.reshape(
                uniform_random_strat,
                (-1, mdp.num_states(), mdp.num_actions())
            )

            action_rewards_weighted_by_chance = tf.squeeze(
                tf.reduce_sum(patient.sequences * mdp.rewards, axis=3)
            )
            current_cf_state_values = None
            for t in range(mdp.horizon - 1, -1, -1):
                n = int(
                    num_pr_sequences(
                        t - 1,
                        mdp.num_states(),
                        mdp.num_actions()
                    ) / mdp.num_states()
                )
                next_n = int(
                    num_pr_sequences(
                        t,
                        mdp.num_states(),
                        mdp.num_actions()
                    ) / mdp.num_states()
                )
                if current_cf_state_values is None:
                    current_cf_action_values = (
                        action_rewards_weighted_by_chance[n:next_n, :, :]
                    )
                else:
                    current_cf_action_values = (
                        action_rewards_weighted_by_chance[n:next_n, :, :] +
                        tf.reshape(
                            tf.reduce_sum(current_cf_state_values, axis=1),
                            [-1, mdp.num_states(), mdp.num_actions()]
                        )
                    )

                current_cf_state_values = tf.expand_dims(
                    tf.reduce_sum(
                        (
                            uniform_random_strat[n:next_n, :, :] *
                            current_cf_action_values
                        ),
                        axis=2
                    ),
                    axis=2
                )
            self.assertAlmostEqual(
                2.0,
                tf.reduce_sum(current_cf_state_values).eval(),
                places=6
            )


if __name__ == '__main__':
    tf.test.main()
