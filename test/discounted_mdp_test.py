import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.discounted_mdp import root_value_op, \
    state_action_successor_policy_evaluation_op, \
    primal_action_value_policy_evaluation_op, \
    dual_action_value_policy_evaluation_op, \
    regret_matching_op, \
    generalized_policy_iteration_op, \
    state_successor_policy_evaluation_op
from tf_kofn_robust_policy_optimization.utils.tensor import \
    normalized, \
    l1_projection_to_simplex, \
    matrix_to_block_matrix_op


class DiscountedMdpTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(10)
        tf.set_random_seed(10)

    def test_row_normalize_op(self):
        root = tf.reshape(normalized(tf.constant([1, 2, 3.0])), [3, 1])
        v = tf.reshape(tf.constant([1.0, 2, 3]), [3, 1])
        self.assertAllClose([[2.3333335]], root_value_op(root, v))

    def test_state_action_successor_policy_evaluation_op(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2

        P = normalized(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1)
        Pi = matrix_to_block_matrix_op(
            normalized(tf.ones([num_states, num_actions]), axis=1))

        x_successor_matrix = [[
            0.2148275077342987, 0.11482749879360199, 0.16744479537010193,
            0.16744479537010193, 0.16772767901420593, 0.16772769391536713
        ], [
            0.09174314141273499, 0.1917431354522705, 0.14561119675636292,
            0.14561119675636292, 0.21264560520648956, 0.21264559030532837
        ], [
            0.10726028680801392, 0.10726027935743332, 0.3093593120574951,
            0.2093593180179596, 0.13338038325309753, 0.13338038325309753
        ], [
            0.11482749879360199, 0.11482749879360199, 0.16744479537010193,
            0.26744478940963745, 0.16772767901420593, 0.16772769391536713
        ], [
            0.11470848321914673, 0.11470848321914673, 0.17305435240268707,
            0.17305435240268707, 0.2622370719909668, 0.16223706305027008
        ], [
            0.12214899063110352, 0.12214899063110352, 0.1533845216035843,
            0.1533845216035843, 0.17446643114089966, 0.2744664251804352
        ]]

        patient = state_action_successor_policy_evaluation_op(
            P, Pi, gamma, max_num_iterations=100)

        self.assertAllClose(x_successor_matrix, patient)

    def test_dual_and_primal_policy_evaluation_agree(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = -1

        P = l1_projection_to_simplex(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1)

        r = tf.random_normal(shape=[num_states * num_actions, 1])

        Pi = matrix_to_block_matrix_op(
            normalized(tf.ones([num_states, num_actions]), axis=1))

        self.assertAllClose(
            primal_action_value_policy_evaluation_op(
                P,
                Pi,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations),
            dual_action_value_policy_evaluation_op(
                P,
                Pi,
                r,
                gamma=gamma,
                threshold=threshold,
                max_num_iterations=max_num_iterations))

    def test_regret_matching_value(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = 100

        P = l1_projection_to_simplex(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1)

        r = tf.random_normal(shape=[num_states * num_actions, 1])

        regret_1_op = regret_matching_op(
            P, r, t=10, max_num_pe_iterations=lambda _: 1)

        Pi = matrix_to_block_matrix_op(
            l1_projection_to_simplex(regret_1_op, axis=1))

        q_op = primal_action_value_policy_evaluation_op(
            P,
            Pi,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)
        self.assertAllClose(
            -2.354442,
            tf.squeeze(
                root_value_op(
                    normalized(tf.ones([num_states, 1])),
                    Pi @ q_op
                )
            )
        )  # yapf:disable

        regret_5_op = regret_matching_op(
            P, r, t=10, max_num_pe_iterations=lambda _: 5)

        Pi = matrix_to_block_matrix_op(
            l1_projection_to_simplex(regret_5_op, axis=1))
        q_op = primal_action_value_policy_evaluation_op(
            P,
            Pi,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)
        root_op = normalized(tf.ones([num_states, 1]))
        ev = tf.squeeze(root_value_op(root_op, Pi @ q_op))
        self.assertAllClose(-2.35444, ev)

    def test_gpi_value(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15,
        max_num_iterations = 100

        P = l1_projection_to_simplex(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1)

        r = tf.random_normal(shape=[num_states * num_actions, 1])

        Pi_1_op = matrix_to_block_matrix_op(
            generalized_policy_iteration_op(
                P, r, gamma=gamma, t=10, max_num_pe_iterations=lambda _: 1))

        q_op = primal_action_value_policy_evaluation_op(
            P,
            Pi_1_op,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)

        mu = normalized(tf.ones([num_states, 1]))

        self.assertAllClose(
            -2.354461,
            tf.squeeze(root_value_op(mu, Pi_1_op @ q_op)))  # yapf:disable

        Pi_5_op = matrix_to_block_matrix_op(
            generalized_policy_iteration_op(
                P, r, gamma=gamma, t=10, max_num_pe_iterations=lambda _: 5))
        q_op = primal_action_value_policy_evaluation_op(
            P,
            Pi_5_op,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)
        self.assertAllClose(
            -2.354438,
            tf.squeeze(root_value_op(mu, Pi_5_op @ q_op)))  # yapf:disable

    def test_recover_state_distribution_from_state_action_distribution(self):
        num_states = 3
        num_actions = 2
        gamma = 0.9

        policy_op = normalized([[1.0, 2.0], [3.0, 0.0], [5.0, 6.0]], axis=1)

        indices_op = tf.stack(
            (tf.range(num_states, dtype=tf.int64), tf.argmax(
                policy_op, axis=1)),
            axis=1)

        non_zero_probs_op = tf.gather_nd(policy_op, indices_op)

        self.assertAllClose([2 / 3.0, 1.0, 0.54545456], non_zero_probs_op)

        Pi_op = matrix_to_block_matrix_op(policy_op)

        P = l1_projection_to_simplex(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1
        )  # yapf:disable

        # This only works when H is very close to the true state-action
        # distribution.
        H_op = state_action_successor_policy_evaluation_op(P, Pi_op, gamma)

        A_op = tf.matmul(Pi_op, H_op)

        columns_to_gather_op = (
            indices_op[:, 1] + num_actions * indices_op[:, 0])
        thin_A_op = tf.transpose(
            tf.gather(tf.transpose(A_op), columns_to_gather_op))

        M_op = thin_A_op / non_zero_probs_op
        sum_M_op = tf.reduce_sum(M_op, axis=1)
        self.assertAllClose(tf.ones_like(sum_M_op), sum_M_op)
        self.assertAllClose(A_op, tf.matmul(M_op, Pi_op))
        self.assertAllClose(M_op,
                            state_successor_policy_evaluation_op(
                                P, Pi_op, gamma))


if __name__ == '__main__':
    tf.test.main()
