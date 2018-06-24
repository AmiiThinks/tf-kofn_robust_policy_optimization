import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import tf_kofn_robust_policy_optimization.utils.tensor as patient


class TensorUtilsTest(tf.test.TestCase):
    def test_l1_projection_to_simplex_no_negative(self):
        self.assertAllClose(
            patient.l1_projection_to_simplex(tf.constant([2.0, 8.0, 0.0])),
            [0.2, 0.8, 0.0])

    def test_l1_projection_to_simplex_with_negative(self):
        self.assertAllClose(
            patient.l1_projection_to_simplex(
                tf.constant([2.0, 8.0, -5.0])), [0.2, 0.8, 0.0])

    def test_l1_projection_to_simplex_multiple_rows(self):
        v = patient.l1_projection_to_simplex(
            tf.transpose(tf.constant([[2.0, 8.0, -5.0], [9.5, 0.4, 0.1]])))

        self.assertAllClose(
            tf.transpose(v), [[0.2, 0.8, 0.0], [0.95, 0.04, 0.01]])

    def test_l1_projection_to_simplex_multiple_rows_axis_1(self):
        v = patient.l1_projection_to_simplex(
            tf.constant([[2.0, 8.0, -5.0], [9.5, 0.4, 0.1]]), axis=1)

        self.assertAllClose(v, [[0.2, 0.8, 0.0], [0.95, 0.04, 0.01]])

    def test_num_pr_sequences_at_timestep(self):
        self.assertAllEqual(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1]
            ],
            patient.block_ones(2, 3)
        )  # yapf:disable

    def test_matrix_to_block_matrix_op(self):
        self.assertAllEqual(
            [
                [1.0, 2, 3, 0, 0, 0],
                [0, 0, 0, 4, 5, 6]
            ],
            patient.matrix_to_block_matrix_op(
                tf.constant([[1.0, 2, 3], [4, 5, 6]])
            )
        )  # yapf:disable

    def test_ind_max(self):
        self.assertAllEqual(
            [
                [1.0, 0, 0],
                [0, 0, 1.0]
            ],
            patient.ind_max_op([[4.0, 2, 3], [4, 5, 6]], axis=1)
        )  # yapf:disable
        self.assertAllEqual(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            patient.ind_max_op(
                [
                    [-2.7822149, -2.3662598],
                    [-2.475302, -2.4006093],
                    [-2.5785851, -3.7109096]
                ],
                axis=1
            )
        )  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
