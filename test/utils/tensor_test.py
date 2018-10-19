import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import tf_kofn_robust_policy_optimization.utils.tensor as patient


class TensorUtilsTest(tf.test.TestCase):
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


if __name__ == '__main__':
    tf.test.main()
