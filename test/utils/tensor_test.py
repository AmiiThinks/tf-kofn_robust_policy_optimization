import tensorflow as tf
import k_of_n_mdp_policy_opt.utils.tensor as patient


class TensorUtilsTest(tf.test.TestCase):
    def test_num_pr_sequences_at_timestep(self):
        self.assertAllEqual(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1]
            ],
            patient.block_ones(2, 3)
        )

    def test_matrix_to_block_matrix_op(self):
        with self.test_session():
            self.assertAllEqual(
                [
                    [1.0, 2, 3, 0, 0, 0],
                    [0, 0, 0, 4, 5, 6]
                ],
                patient.matrix_to_block_matrix_op(
                    tf.constant([[1.0, 2, 3], [4, 5, 6]])
                ).eval()
            )

    def test_ind_max(self):
        with self.test_session():
            self.assertAllEqual(
                [
                    [1.0, 0, 0],
                    [0, 0, 1.0]
                ],
                patient.ind_max_op(
                    tf.constant([[4.0, 2, 3], [4, 5, 6]])
                ).eval()
            )
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
                    ]
                ).eval()
            )


if __name__ == '__main__':
    tf.test.main()
