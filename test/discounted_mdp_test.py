import tensorflow as tf
from k_of_n_mdp_policy_opt.discounted_mdp import root_value_op, \
    state_action_successor_policy_evaluation_op, \
    primal_action_value_policy_evaluation_op, \
    dual_action_value_policy_evaluation_op, \
    regret_matching_op, \
    generalized_policy_iteration_op, \
    state_successor_policy_evaluation_op
from k_of_n_mdp_policy_opt.utils.tensor import row_normalize_op, \
    matrix_to_block_matrix_op
from k_of_n_mdp_policy_opt.utils.random import reset_random_state


class DiscountedMdpTest(tf.test.TestCase):
    def test_row_normalize_op(self):
        with self.test_session():
            root = tf.reshape(
                row_normalize_op(tf.constant([1, 2, 3.0])), [3, 1])
            v = tf.reshape(tf.constant([1.0, 2, 3]), [3, 1])
            self.assertAlmostEqual(2.3333335, root_value_op(root, v).eval())

    def test_state_action_successor_policy_evaluation_op(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        with self.test_session() as sess:
            P = sess.run(
                row_normalize_op(
                    tf.random_normal(
                        shape=[num_states * num_actions, num_states])))
            Pi = matrix_to_block_matrix_op(
                row_normalize_op(tf.ones([num_states, num_actions])))

            x_successor_matrix = [[
                0.2285012, 0.12850118, 0.30992848, 0.30992848, 0.01157004,
                0.01157003
            ], [
                0.0663096, 0.1663096, 0.38033253, 0.38033253, 0.00335755,
                0.00335755
            ], [
                0.06453303, 0.06453302, 0.48219907, 0.38219905, 0.0032676,
                0.0032676
            ], [
                0.05279975, 0.05279974, 0.39452648, 0.49452642, 0.00267349,
                0.00267349
            ], [
                0.06508669, 0.06508669, 0.36358333, 0.3635833, 0.12132971,
                0.0213297
            ], [
                0.13266486, 0.13266484, 0.31061745, 0.31061745, 0.00671741,
                0.10671742
            ]]
            self.assertAllClose(x_successor_matrix,
                                state_action_successor_policy_evaluation_op(
                                    P, Pi, gamma).eval())

    def test_dual_and_primal_policy_evaluation_agree(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = -1
        with self.test_session() as sess:
            reset_random_state(10)
            P = sess.run(
                row_normalize_op(
                    tf.random_normal(
                        shape=[num_states * num_actions, num_states])))
            r = sess.run(tf.random_normal(shape=[num_states * num_actions, 1]))
            Pi = matrix_to_block_matrix_op(
                row_normalize_op(tf.ones([num_states, num_actions])))
            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    P,
                    Pi,
                    r,
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations).eval(),
                dual_action_value_policy_evaluation_op(
                    P,
                    Pi,
                    r,
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations).eval())

    def test_regret_matching_value(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = -1
        with self.test_session() as sess:
            reset_random_state(101)
            P = tf.constant(
                sess.run(
                    row_normalize_op(
                        tf.random_normal(
                            shape=[num_states * num_actions, num_states]))))
            r = tf.constant(
                sess.run(
                    tf.random_normal(shape=[num_states * num_actions, 1])))
            regret_1_op = regret_matching_op(
                P, r, t=10, max_num_pe_iterations=lambda _: 1)
            Pi = matrix_to_block_matrix_op(row_normalize_op(regret_1_op))
            q_op = primal_action_value_policy_evaluation_op(
                P,
                Pi,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations)
            self.assertAlmostEqual(3.001621,
                                   tf.squeeze(
                                       root_value_op(
                                           row_normalize_op(
                                               tf.ones([num_states, 1])),
                                           Pi @ q_op)).eval())

            regret_5_op = regret_matching_op(
                P, r, t=10, max_num_pe_iterations=lambda _: 5)

            Pi = matrix_to_block_matrix_op(row_normalize_op(regret_5_op))
            q_op = primal_action_value_policy_evaluation_op(
                P,
                Pi,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations)
            root_op = row_normalize_op(tf.ones([num_states, 1]))
            ev = tf.squeeze(root_value_op(root_op, Pi @ q_op))
            self.assertAlmostEqual(3.031004, ev.eval())

    def test_gpi_value(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = -1
        with self.test_session() as sess:
            reset_random_state(101)
            P = tf.constant(
                sess.run(
                    row_normalize_op(
                        tf.random_normal(
                            shape=[num_states * num_actions, num_states]))))
            r = tf.constant(
                sess.run(
                    tf.random_normal(shape=[num_states * num_actions, 1])))
            Pi_1_op = matrix_to_block_matrix_op(
                generalized_policy_iteration_op(
                    P, r, gamma=gamma, t=10,
                    max_num_pe_iterations=lambda _: 1))
            q_op = primal_action_value_policy_evaluation_op(
                P,
                Pi_1_op,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations)
            self.assertAlmostEqual(3.001621,
                                   tf.squeeze(
                                       root_value_op(
                                           row_normalize_op(
                                               tf.ones([num_states, 1])),
                                           Pi_1_op @ q_op)).eval())

            Pi_5_op = matrix_to_block_matrix_op(
                generalized_policy_iteration_op(
                    P, r, gamma=gamma, t=10,
                    max_num_pe_iterations=lambda _: 5))
            q_op = primal_action_value_policy_evaluation_op(
                P,
                Pi_5_op,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations)
            self.assertAlmostEqual(3.031004,
                                   tf.squeeze(
                                       root_value_op(
                                           row_normalize_op(
                                               tf.ones([num_states, 1])),
                                           Pi_5_op @ q_op)).eval())

    def test_recover_state_distribution_from_state_action_distribution(self):
        num_states = 3
        num_actions = 2
        gamma = 0.9
        with self.test_session() as sess:
            reset_random_state(101)

            policy_op = row_normalize_op(
                [
                    [1.0, 2.0],
                    [3.0, 0.0],
                    [5.0, 6.0]
                ]
            )
            indices_op = tf.stack(
                (
                    tf.range(num_states, dtype=tf.int64),
                    tf.argmax(policy_op, axis=1)
                ),
                axis=1
            )
            non_zero_probs_op = tf.gather_nd(policy_op, indices_op)

            self.assertAllClose(
                [2 / 3.0, 1.0, 0.54545456],
                non_zero_probs_op.eval()
            )

            Pi_op = matrix_to_block_matrix_op(policy_op)
            P = sess.run(
                row_normalize_op(
                    tf.random_normal(
                        shape=[num_states * num_actions, num_states])))

            # This only works when H is very close to the true state-action
            # distribution.
            H_op = state_action_successor_policy_evaluation_op(
                P, Pi_op, gamma)

            A_op = tf.matmul(Pi_op, H_op)

            columns_to_gather_op = (
                indices_op[:, 1] + num_actions * indices_op[:, 0]
            )
            thin_A_op = tf.transpose(
                tf.gather(
                    tf.transpose(A_op),
                    columns_to_gather_op
                )
            )

            M_op = thin_A_op / non_zero_probs_op
            sum_M_op = tf.reduce_sum(M_op, axis=1)
            self.assertAllClose(
                tf.ones_like(sum_M_op).eval(),
                sum_M_op.eval()
            )
            self.assertAllClose(
                A_op.eval(),
                tf.matmul(M_op, Pi_op).eval()
            )
            self.assertAllClose(
                M_op.eval(),
                state_successor_policy_evaluation_op(P, Pi_op, gamma).eval()
            )


if __name__ == '__main__':
    tf.test.main()
