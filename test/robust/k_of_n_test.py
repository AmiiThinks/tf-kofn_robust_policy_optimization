import tensorflow as tf
import amii_tf_mdp.robust.k_of_n as patient
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp


class KOfNTest(tf.test.TestCase):
    def test_prob_ith_element(self):
        with self.test_session():
            n_weights = [1, 0.0]
            k_weights = [1, 0.0]
            self.assertAllEqual(
                [1, 0.0],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [0.0, 1.0]
            k_weights = [0.0, 1.0]
            self.assertAllEqual(
                [1.0, 1.0],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [0.0, 1.0]
            k_weights = [1.0, 1.0]
            self.assertAllEqual(
                [1.0, 0.5],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [1.0, 1.0]
            k_weights = [1.0, 1.0]
            self.assertAllEqual(
                [1.0, 0.25],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [1.0, 2.0]
            k_weights = [1.0, 1.0]
            self.assertAllClose(
                [1.0, 1 / 3.0],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [1.0, 2.0]
            k_weights = [2.0, 1.0]
            self.assertAllClose(
                [1.0, 2 / 9.0],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [1.0, 2.0, 1.0]
            k_weights = [2.0, 1.0, 0.0]
            self.assertAllClose(
                [1.0, 3 / 12.0, 0],
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

            n_weights = [1.0, 2.0, 1.0]
            k_weights = [2.0, 1.0, 0.1]
            self.assertAllClose(
                [
                    1.0,
                    2 / 4.0 * 1 / 3.0 + 1 / 4.0 * 1.1 / 3.1,
                    1 / 4.0 * 0.1 / 3.1
                ],
                patient.prob_ith_element(n_weights, k_weights).eval()
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
                patient.prob_ith_element(n_weights, k_weights).eval()
            )

    def test_init(self):
        with self.test_session():
            patient.KofnGadget(
                [1.0] * 2,
                [1.0] * 2,
                lambda: PrUncertainMdp(2, 2, 2)
            )

    def test_bind(self):
        with self.test_session():
            patient.KofnGadget(
                [1.0] * 2,
                [1.0] * 2,
                lambda: PrUncertainMdp(2, 2, 2)
            ).bind(
                [
                    [
                        [
                            [
                                [0.1, 0.9],
                                [0.2, 0.8]
                            ],
                            [
                                [0.3, 0.7],
                                [0.4, 0.6]
                            ]
                        ],
                        [
                            [
                                [1.0, 9.0],
                                [2.0, 8.0]
                            ],
                            [
                                [3.0, 7.0],
                                [4.0, 6.0]
                            ]
                        ],
                        [0.1, 0.9],
                        [
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
                        ],
                    ]
                ] * 2
            )


if __name__ == '__main__':
    tf.test.main()
