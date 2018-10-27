import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    root_value_op, \
    state_action_successor_policy_evaluation_op, \
    primal_action_value_policy_evaluation_op, \
    dual_action_value_policy_evaluation_op, \
    generalized_policy_iteration_op, \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op
from tf_contextual_prediction_with_expert_advice import \
    normalized, \
    l1_projection_to_simplex
from tf_kofn_robust_policy_optimization.utils.tensor import \
    matrix_to_block_matrix_op


class DiscountedMdpTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(10)
        tf.set_random_seed(10)

    def test_row_normalize_op(self):
        root = normalized([1, 2, 3.0])
        v = [1.0, 2, 3]
        self.assertAllClose(2.3333335, root_value_op(root, v))

    def test_state_action_successor_policy_evaluation_op(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2

        P = normalized(
            tf.random_normal(shape=[num_states * num_actions, num_states]),
            axis=1)

        Pi = normalized(tf.ones([num_states, num_actions]), axis=1)

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
            tf.reshape(P, [num_states, num_actions, num_states]),
            Pi,
            gamma,
            max_num_iterations=100)

        self.assertAllClose(x_successor_matrix, patient, rtol=1e-5, atol=1e-5)

    def test_dual_and_primal_policy_evaluation_agree(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15
        max_num_iterations = -1

        transitions = tf.reshape(
            l1_projection_to_simplex(
                tf.random_normal(shape=[num_states * num_actions, num_states]),
                axis=1
            ),
            [num_states, num_actions, num_states]
        )  # yapf:disable

        r = tf.random_normal(shape=[num_states, num_actions])

        policy = normalized(tf.ones([num_states, num_actions]), axis=1)

        with self.subTest('single reward function'):
            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r,
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations),
                dual_action_value_policy_evaluation_op(
                    transitions, policy, r, gamma=gamma))

        with self.subTest('two reward functions'):
            r_both = tf.stack(
                [r, tf.random_normal(shape=[num_states, num_actions])],
                axis=-1)

            patient = dual_action_value_policy_evaluation_op(
                transitions, policy, r_both, gamma=gamma)

            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r_both[:, :, 0],
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations), patient[:, :, 0])
            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r_both[:, :, 1],
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations), patient[:, :, 1])

    def test_gpi_value(self):
        gamma = 0.9
        num_states = 3
        num_actions = 2
        threshold = 1e-15
        max_num_iterations = 100

        transitions = tf.reshape(
            l1_projection_to_simplex(
                tf.random_normal(shape=[num_states * num_actions, num_states]),
                axis=1
            ),
            [num_states, num_actions, num_states]
        )  # yapf:disable

        r = tf.random_normal(shape=[num_states, num_actions])

        policy_1_op = generalized_policy_iteration_op(
            transitions,
            r,
            gamma=gamma,
            t=10,
            max_num_pe_iterations=lambda _: 1)

        q_op = primal_action_value_policy_evaluation_op(
            transitions,
            policy_1_op,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)

        mu = normalized(tf.ones([num_states]))

        v = tf.reduce_sum(policy_1_op * q_op, axis=-1)
        self.assertAllClose(-2.354447, root_value_op(mu, v))

        policy_5_op = generalized_policy_iteration_op(
            transitions,
            r,
            gamma=gamma,
            t=10,
            max_num_pe_iterations=lambda _: 5)
        q_op = primal_action_value_policy_evaluation_op(
            transitions,
            policy_5_op,
            r,
            gamma=gamma,
            threshold=threshold,
            max_num_iterations=max_num_iterations)

        v = tf.reduce_sum(policy_5_op * q_op, axis=-1)
        self.assertAllClose(-2.354447, root_value_op(mu, v))

        dual_state_values = dual_state_value_policy_evaluation_op(
            transitions, policy_5_op, r, gamma=gamma)

        self.assertAllClose(
            -2.354438,
            root_value_op(mu, dual_state_values),
            rtol=1e-04,
            atol=1e-04)

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

        transitions = tf.reshape(
            l1_projection_to_simplex(
                tf.random_normal(shape=[num_states * num_actions, num_states]),
                axis=1
            ),
            [num_states, num_actions, num_states]
        )  # yapf:disable

        # This only works when H is very close to the true state-action
        # distribution.
        H_op = state_action_successor_policy_evaluation_op(
            transitions, policy_op, gamma)

        A_op = tf.matmul(Pi_op, H_op)

        columns_to_gather_op = (
            indices_op[:, 1] + num_actions * indices_op[:, 0])
        thin_A_op = tf.transpose(
            tf.gather(tf.transpose(A_op), columns_to_gather_op))

        M_op = thin_A_op / non_zero_probs_op
        sum_M_op = tf.reduce_sum(M_op, axis=1)
        self.assertAllClose(tf.ones_like(sum_M_op), sum_M_op)
        self.assertAllClose(A_op, tf.matmul(M_op, Pi_op))
        self.assertAllClose(
            M_op, (1.0 - gamma) * state_successor_policy_evaluation_op(
                transitions, policy_op, gamma))


if __name__ == '__main__':
    tf.test.main()
