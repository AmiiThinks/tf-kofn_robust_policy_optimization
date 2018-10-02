import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
import tf_kofn_robust_policy_optimization.robust as patient


class RobustTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_deterministic_kofn_weights(self):
        n = 5
        for k in range(1, n):
            x = np.zeros([n])
            x[:k] = 1.0 / k
            self.assertAllClose(x, patient.deterministic_kofn_weights(k, n))

    def test_rank_to_element_weights(self):
        weights = patient.rank_to_element_weights([0.1, 0.7, 0.2],
                                                  [-1.0, -2.0, -3.0])
        self.assertAllClose([0.2, 0.7, 0.1], weights)

    def test_prob_ith_element_is_in_k_subset(self):
        n_weights = [1, 0.0]
        k_weights = [1, 0.0]
        self.assertAllEqual(
            [1, 0.0],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [0.0, 1.0]
        k_weights = [0.0, 1.0]
        self.assertAllEqual(
            [1.0, 1.0],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [0.0, 1.0]
        k_weights = [1.0, 1.0]
        self.assertAllEqual(
            [1.0, 0.5],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [1.0, 1.0]
        k_weights = [1.0, 1.0]
        self.assertAllEqual(
            [1.0, 0.25],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [1.0, 2.0]
        k_weights = [1.0, 1.0]
        self.assertAllClose(
            [1.0, 1 / 3.0],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [1.0, 2.0]
        k_weights = [2.0, 1.0]
        self.assertAllClose(
            [1.0, 2 / 9.0],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [1.0, 2.0, 1.0]
        k_weights = [2.0, 1.0, 0.0]
        self.assertAllClose(
            [1.0, 3 / 12.0, 0],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

        n_weights = [1.0, 2.0, 1.0]
        k_weights = [2.0, 1.0, 0.1]
        self.assertAllClose(
            [
                1.0,
                2 / 4.0 * 1 / 3.0 + 1 / 4.0 * 1.1 / 3.1,
                1 / 4.0 * 0.1 / 3.1
            ],
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable

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
            patient.prob_ith_element_is_in_k_subset(n_weights, k_weights)
        )  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
