import tensorflow as tf
import numpy as np
from k_of_n_mdp_policy_opt.pr_mdp import pr_mdp_rollout, pr_mdp_expected_value, \
    pr_mdp_optimal_policy_and_value
from k_of_n_mdp_policy_opt.utils.sequence import num_pr_sequences
from amii_tf_nn.projection import l1_projection_to_simplex


class PrMdpTest(tf.test.TestCase):
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
            root = l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
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
                sess.run(
                    pr_mdp_rollout(horizon, root, transition_model)
                )[:1, :, :, :]
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
            rewards = np.ones((num_states, num_actions, num_states))
            root = l1_projection_to_simplex(tf.ones([3]))
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
            self.assertAlmostEqual(
                2.0,
                sess.run(
                    pr_mdp_expected_value(
                        horizon,
                        num_states,
                        num_actions,
                        pr_mdp_rollout(horizon, root, transition_model),
                        rewards,
                        uniform_random_strat
                    )
                ),
                places=6
            )

    def test_expected_value_2(self):
        with self.test_session() as sess:
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
            root = l1_projection_to_simplex(tf.ones([3]))
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
            ev = pr_mdp_expected_value(
                horizon,
                num_states,
                num_actions,
                pr_mdp_rollout(horizon, root, transition_model),
                rewards,
                uniform_random_strat
            )
            self.assertAlmostEqual(-0.62523419, sess.run(ev))
            self.assertAlmostEqual(-0.62523419, sess.run(ev))

    def test_best_response(self):
        with self.test_session() as sess:
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
            root = l1_projection_to_simplex(tf.ones([3]))
            br_strat, br_val = pr_mdp_optimal_policy_and_value(
                horizon,
                num_states,
                num_actions,
                pr_mdp_rollout(horizon, root, transition_model),
                rewards
            )
            self.assertAlmostEqual(-0.35074115, sess.run(br_val))


if __name__ == '__main__':
    tf.test.main()
