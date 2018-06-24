import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import tf_kofn_robust_policy_optimization.utils.probability as patient


class ProbabilityUtilsTest(tf.test.TestCase):
    def test_prob_action(self):
        state_dist = tf.constant([0.5, 0.5])
        strat = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5]
            ]
        )  # yapf:disable
        self.assertAllClose([0.5, 0.25, 0.25],
                            patient.prob_action(state_dist, strat))

        state_dist = tf.constant([1.0, 0.0])
        self.assertAllClose([1.0, 0.0, 0.0],
                            patient.prob_action(state_dist, strat))

        state_dist = tf.constant([0.0, 1.0])
        self.assertAllClose([0.0, 0.5, 0.5],
                            patient.prob_action(state_dist, strat))

        state_dist = tf.constant([0.55000001, 0.44999999])
        self.assertAllClose([0.55000001, 0.22499999, 0.22499999],
                            patient.prob_action(state_dist, strat))

    def test_prob_state_and_action(self):
        state_dist = tf.constant([0.5, 0.5])
        strat = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5]
            ]
        )  # yapf:disable
        self.assertAllClose(
            [[0.5, 0., 0.],
             [0., 0.25, 0.25]],
            patient.prob_state_and_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([1.0, 0.0])
        self.assertAllClose(
            [[1., 0., 0.],
             [0., 0., 0.]],
            patient.prob_state_and_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([0.0, 1.0])
        self.assertAllClose(
            [[0., 0., 0.],
             [0., 0.5, 0.5]],
            patient.prob_state_and_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([0.55000001, 0.44999999])
        self.assertAllClose(
            [[0.55000001, 0., 0.],
             [0., 0.22499999, 0.22499999]],
            patient.prob_state_and_action(state_dist, strat)
        )  # yapf:disable

    def test_prob_state_given_action(self):
        state_dist = tf.constant([0.5, 0.5])
        strat = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5]
            ]
        )  # yapf:disable
        self.assertAllClose(
            [[1., 0.],
             [0., 1.],
             [0., 1.]],
            patient.prob_state_given_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([1.0, 0.0])
        self.assertAllClose(
            [[1., 0.],
             [0., 0.],
             [0., 0.]],
            patient.prob_state_given_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([0.0, 1.0])
        self.assertAllClose(
            [[0., 0.],
             [0., 1.],
             [0., 1.]],
            patient.prob_state_given_action(state_dist, strat)
        )  # yapf:disable

        state_dist = tf.constant([0.55000001, 0.44999999])
        self.assertAllClose(
            [[1., 0.],
             [0., 1.],
             [0., 1.]],
            patient.prob_state_given_action(state_dist, strat)
        )  # yapf:disable

    def test_prob_next_state_given_action(self):
        T = tf.constant(
            [
                [
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6]
                ],
                [
                    [0.1, 0.9],
                    [0.5, 0.5],
                    [0.6, 0.4]
                ]
            ]
        )  # yapf:disable

        state_dist = tf.constant([0.5, 0.5])
        strat = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5]
            ]
        )  # yapf:disable

        self.assertAllClose(
            [[0.2, 0.80000001],
             [0.5, 0.5],
             [0.60000002, 0.40000001]],
            patient.prob_next_state_given_action(
                T,
                state_dist,
                strat
            )
        )  # yapf:disable

        T = tf.constant(
            [
                [
                    [1.0, 0.0],
                    [0.3, 0.7],
                    [0.4, 0.6]
                ],
                [
                    [0.1, 0.9],
                    [0.5, 0.5],
                    [0.6, 0.4]
                ]
            ]
        )  # yapf:disable

        state_dist = tf.constant([1.0, 0.0])
        self.assertAllClose(
            [[1., 0.],
             [0., 0.],
             [0., 0.]],
            patient.prob_next_state_given_action(
                T,
                state_dist,
                strat
            )
        )  # yapf:disable

        state_dist = tf.constant([0.0, 1.0])
        self.assertAllClose(
            [[0., 0.],
             [0.5, 0.5],
             [0.60000002, 0.40000001]],
            patient.prob_next_state_given_action(
                T,
                state_dist,
                strat
            )
        )  # yapf:disable

        state_dist = tf.constant([0.55000001, 0.44999999])
        self.assertAllClose(
            [[1., 0.],
             [0.5, 0.5],
             [0.60000002, 0.40000001]],
            patient.prob_next_state_given_action(
                T,
                state_dist,
                strat
            )
        )  # yapf:disable

    def test_prob_next_state(self):
        T = tf.constant(
            [
                [
                    [0.2, 0.8],  # P(s' | s = 1, a = 1)
                    [0.3, 0.7],  # P(s' | s = 1, a = 2)
                    [0.4, 0.6]   # P(s' | s = 1, a = 3)
                ],
                [
                    [0.1, 0.9],  # P(s' | s = 2, a = 1)
                    [0.5, 0.5],  # P(s' | s = 2, a = 2)
                    [0.6, 0.4]   # P(s' | s = 2, a = 3)
                ]
            ]
        )  # yapf:disable
        state_dist = tf.constant([0.5, 0.5])
        strat = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5]
            ]
        )  # yapf:disable
        self.assertAllClose([0.375, 0.625],
                            patient.prob_next_state(T, state_dist, strat))

        T = tf.constant(
            [
                [
                    [1.0, 0.0],
                    [0.3, 0.7],
                    [0.4, 0.6]
                ],
                [
                    [0.1, 0.9],
                    [0.5, 0.5],
                    [0.6, 0.4]
                ]
            ]
        )  # yapf:disable
        state_dist = tf.constant([1.0, 0.0])
        self.assertAllClose([1., 0.],
                            patient.prob_next_state(T, state_dist, strat))

        state_dist = tf.constant([0.0, 1.0])
        self.assertAllClose([0.55000001, 0.44999999],
                            patient.prob_next_state(T, state_dist, strat))

        state_dist = tf.constant([0.55000001, 0.44999999])
        self.assertAllClose([0.79750001, 0.20249999],
                            patient.prob_next_state(T, state_dist, strat))


if __name__ == '__main__':
    tf.test.main()
