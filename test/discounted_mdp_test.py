import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    primal_action_value_policy_evaluation_op, \
    dual_action_value_policy_evaluation_op, \
    generalized_policy_iteration_op, \
    dual_state_value_policy_evaluation_op, \
    state_successor_policy_evaluation_op
from tf_contextual_prediction_with_expert_advice import \
    normalized, \
    l1_projection_to_simplex


class DiscountedMdpTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(10)
        tf.set_random_seed(10)

    def test_situation_dependent_discount(self):
        num_states = 3
        num_actions = 2
        transitions = tf.reshape(
            l1_projection_to_simplex(
                tf.random_normal(shape=[num_states * num_actions, num_states]),
                axis=1
            ),
            [num_states, num_actions, num_states]
        )  # yapf:disable
        policy = normalized(tf.zeros([num_states, num_actions]), axis=1)
        rewards = [[1.0, 2], [3, 4], [5, 6]]

        self.assertAllClose(
            dual_state_value_policy_evaluation_op(transitions, policy, rewards,
                                                  0.99),
            [355.64996, 356.69995, 358.75452])

        discounts = np.full([num_states, 1], 0.99, dtype='float32')
        discounts[-1] = 0

        self.assertAllClose(
            dual_state_value_policy_evaluation_op(transitions, policy, rewards,
                                                  discounts),
            [13.126714, 17.98831, 5.5])

        discounts = np.full([num_states, num_actions], 0.99, dtype='float32')
        discounts[-1, -1] = 0

        self.assertAllClose(
            dual_state_value_policy_evaluation_op(transitions, policy, rewards,
                                                  discounts),
            [25.490976, 30.214983, 18.251637])

        self.assertAllClose(
            dual_action_value_policy_evaluation_op(transitions, policy,
                                                   rewards, discounts),
            [[30.912834, 20.06912], [32.023956, 28.406008], [30.503275, 6.]])

    def test_state_successor_policy_evaluation_op(self):
        num_states = 3
        num_actions = 2
        transitions = tf.reshape(
            l1_projection_to_simplex(
                tf.random_normal(shape=[num_states * num_actions, num_states]),
                axis=1
            ),
            [num_states, num_actions, num_states]
        )  # yapf:disable
        policy = l1_projection_to_simplex(
            tf.random_normal([num_states, num_actions]), axis=1)

        def run_test(gamma):
            patient = state_successor_policy_evaluation_op(
                transitions, policy, gamma)

            with self.subTest('row normalized'):
                c = tf.reduce_sum(patient, axis=-1)
                if gamma < 1:
                    c = (1.0 - gamma) * c
                self.assertAllClose(
                    tf.ones([num_states]), c, rtol=1e-4, atol=1e-4)
            with self.subTest('values without batch dimension'):
                threshold = 1e-10
                max_num_iterations = -1
                r = tf.random_normal(shape=[num_states, num_actions])

                q = primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r,
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations)

                v = tf.reduce_sum(
                    patient * tf.expand_dims(
                        tf.reduce_sum(r * policy, axis=-1), 0),
                    axis=-1)

                self.assertAllClose(
                    tf.reduce_sum(q * policy, axis=-1),
                    v,
                    rtol=1e-3,
                    atol=1e-3)

                self.assertAllClose(
                    dual_state_value_policy_evaluation_op(
                        transitions, policy, r, gamma), v)
            with self.subTest('values with batch dimension'):
                threshold = 1e-10
                max_num_iterations = -1
                batch_size = 2
                r = tf.random_normal(
                    shape=[batch_size, num_states, num_actions])

                v = tf.reduce_sum(
                    tf.expand_dims(patient, 0) * tf.expand_dims(
                        tf.reduce_sum(r * tf.expand_dims(policy, 0), axis=-1),
                        1),
                    axis=-1)

                self.assertAllClose(
                    dual_state_value_policy_evaluation_op(
                        transitions, policy, r, gamma), v)

                for i in range(batch_size):
                    q = primal_action_value_policy_evaluation_op(
                        transitions,
                        policy,
                        r[i],
                        gamma=gamma,
                        threshold=threshold,
                        max_num_iterations=max_num_iterations)

                    self.assertAllClose(
                        tf.reduce_sum(q * policy, axis=-1),
                        v[i],
                        rtol=1e-2,
                        atol=1e-2)

        for gamma in [0.9, 0.99, 0.999]:
            with self.subTest('with discount={}'.format(gamma)):
                run_test(gamma)

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
                [r, tf.random_normal(shape=[num_states, num_actions])], axis=0)

            patient = dual_action_value_policy_evaluation_op(
                transitions, policy, r_both, gamma=gamma)

            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r_both[0],
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations), patient[0])
            self.assertAllClose(
                primal_action_value_policy_evaluation_op(
                    transitions,
                    policy,
                    r_both[1],
                    gamma=gamma,
                    threshold=threshold,
                    max_num_iterations=max_num_iterations), patient[1])

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
        self.assertAllClose(-2.354447, tf.reduce_sum(mu * v))

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
        self.assertAllClose(-2.354447, tf.reduce_sum(mu * v))

        dual_state_values = dual_state_value_policy_evaluation_op(
            transitions, policy_5_op, r, gamma=gamma)

        self.assertAllClose(
            -2.354438,
            tf.reduce_sum(mu * dual_state_values),
            rtol=1e-4,
            atol=1e-4)


if __name__ == '__main__':
    tf.test.main()
