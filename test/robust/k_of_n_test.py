import tensorflow as tf
import amii_tf_mdp.robust.k_of_n as patient
from amii_tf_mdp.pr_mdp import pr_mdp_rollout, pr_mdp_evs


class KOfNTest(tf.test.TestCase):
    def test_prob_ith_element_is_in_k_subset(self):
        with self.test_session():
            n_weights = [1, 0.0]
            k_weights = [1, 0.0]
            self.assertAllEqual(
                [1, 0.0],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [0.0, 1.0]
            k_weights = [0.0, 1.0]
            self.assertAllEqual(
                [1.0, 1.0],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [0.0, 1.0]
            k_weights = [1.0, 1.0]
            self.assertAllEqual(
                [1.0, 0.5],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 1.0]
            k_weights = [1.0, 1.0]
            self.assertAllEqual(
                [1.0, 0.25],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 2.0]
            k_weights = [1.0, 1.0]
            self.assertAllClose(
                [1.0, 1 / 3.0],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 2.0]
            k_weights = [2.0, 1.0]
            self.assertAllClose(
                [1.0, 2 / 9.0],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 2.0, 1.0]
            k_weights = [2.0, 1.0, 0.0]
            self.assertAllClose(
                [1.0, 3 / 12.0, 0],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 2.0, 1.0]
            k_weights = [2.0, 1.0, 0.1]
            self.assertAllClose(
                [
                    1.0,
                    2 / 4.0 * 1 / 3.0 + 1 / 4.0 * 1.1 / 3.1,
                    1 / 4.0 * 0.1 / 3.1
                ],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

            n_weights = [1.0, 2.0, 1.0, 3.0]
            k_weights = [2.0, 1.0, 0.1, 0.7]
            self.assertAllClose(
                [
                    1.0,
                    (
                        2 / 7.0 * 1 / 3.0 +
                        1 / 7.0 * 1.1 / 3.1 +
                        3 / 7.0 * 1.8 / 3.8
                    ),
                    1 / 7.0 * 0.1 / 3.1 + 3 / 7.0 * 0.8 / 3.8,
                    3 / 7.0 * 0.7 / 3.8
                ],
                patient.prob_ith_element_is_in_k_subset(
                    n_weights,
                    k_weights
                ).eval()
            )

    def test_bind(self):
        with self.test_session() as sess:
            horizon = 2
            num_mdps = 2
            roots = [tf.constant([0.1, 0.9])] * num_mdps
            transition_models = [
                tf.constant(
                    [
                        [
                            [0.1, 0.9],
                            [0.2, 0.8]
                        ],
                        [
                            [0.3, 0.7],
                            [0.4, 0.6]
                        ]
                    ]
                )
            ] * num_mdps
            reward_models = [
                tf.constant(
                    [
                        [
                            [1.0, 9.0],
                            [2.0, 8.0]
                        ],
                        [
                            [3.0, 7.0],
                            [4.0, 6.0]
                        ]
                    ]
                )
            ] * num_mdps

            chance_prob_sequences = [
                pr_mdp_rollout(horizon, roots[i], transition_models[i])
                for i in range(len(roots))
            ]

            n_weights = [0.0] * (num_mdps - 1) + [1.0]
            k_weights = [1.0] + [0.0] * (num_mdps - 1)
            strat = [
                # First timestep
                [0.1, 0.9],
                [0.2, 0.8],
                # Second timestep
                [0.3, 0.7],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.8, 0.2],
                [0.9, 0.1],
                [1.0, 0]
            ]

            evs = pr_mdp_evs(
                horizon,
                chance_prob_sequences,
                reward_models,
                strat
            )
            mdp_weights = patient.k_of_n_mdp_weights(
                n_weights,
                k_weights,
                evs
            )
            k_of_n_ev = patient.k_of_n_ev(evs, mdp_weights)

            self.assertAllClose([12.0361805, 12.0361805], sess.run(evs))
            self.assertAllClose([0.0, 1.0], sess.run(mdp_weights))
            self.assertNear(12.0361805, sess.run(k_of_n_ev), 1e-8)


if __name__ == '__main__':
    tf.test.main()
