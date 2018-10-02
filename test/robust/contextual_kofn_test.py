import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.robust.kofn import \
    DeterministicKofnGameTemplate
from tf_kofn_robust_policy_optimization.robust.contextual_kofn import \
    ContextualKofnGame


class ContextualKofnTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

    def test_1_of_5_game_utilties(self):
        n = 5
        num_actions = 3

        utility_of_instance_given_action = np.random.normal(
            size=[num_actions, 1, n]).astype('float32')

        self.assertAllClose([[
            0.49671414494514465, -0.13826429843902588, 0.6476885676383972,
            1.5230298042297363, -0.2341533750295639
        ], [
            -0.23413695394992828, 1.5792127847671509, 0.7674347162246704,
            -0.4694743752479553, 0.5425600409507751
        ], [
            -0.4634176790714264, -0.4657297432422638, 0.241962268948555,
            -1.9132802486419678, -1.7249178886413574
        ]], utility_of_instance_given_action[:, 0, :])

        strategy = np.random.uniform(size=[num_actions, 1]).astype('float32')
        strategy /= tf.reduce_sum(strategy, axis=0, keepdims=True)

        with self.test_session():
            game_template = DeterministicKofnGameTemplate(1, n)
            utilities = tf.reduce_mean(
                ContextualKofnGame(
                    game_template.prob_ith_element_is_sampled,
                    tf.transpose(utility_of_instance_given_action, [1, 0, 2]),
                    tf.transpose(strategy)).kofn_utility,
                axis=0)

            self.assertAllClose([[0.5863516330718994], [0.13367994129657745],
                                 [0.2799684405326843]], strategy)

            # This makes sense since the opponent can choose the worst
            # column from utility_of_instance_given_action.
            self.assertAllClose(
                [-0.2341533750295639, 0.5425600409507751, -1.7249178886413574],
                utilities)

    def test_2_of_5_game_utilties(self):
        n = 5
        num_actions = 3

        utility_of_instance_given_action = np.random.normal(
            size=[num_actions, 1, n]).astype('float32')

        self.assertAllClose([[
            0.49671414494514465, -0.13826429843902588, 0.6476885676383972,
            1.5230298042297363, -0.2341533750295639
        ], [
            -0.23413695394992828, 1.5792127847671509, 0.7674347162246704,
            -0.4694743752479553, 0.5425600409507751
        ], [
            -0.4634176790714264, -0.4657297432422638, 0.241962268948555,
            -1.9132802486419678, -1.7249178886413574
        ]], utility_of_instance_given_action[:, 0, :])

        strategy = np.random.uniform(size=[num_actions, 1]).astype('float32')
        strategy /= tf.reduce_sum(strategy, axis=0, keepdims=True)

        game_template = DeterministicKofnGameTemplate(2, n)

        utility_of_instance_given_action = tf.constant(
            utility_of_instance_given_action, name='normal_random_utils')
        game = ContextualKofnGame(
            game_template.prob_ith_element_is_sampled,
            tf.transpose(utility_of_instance_given_action, [1, 0, 2]),
            tf.transpose(strategy))

        utilities = tf.reduce_mean(game.kofn_utility, axis=0)

        with self.test_session():
            self.assertAllClose(
                [0.13, -3.5e-4, 0.55, 0.29, -0.55],
                game.evs,
                rtol=1e-2,
                atol=1e-2)

            self.assertAllClose([0, 0.5, 0, 0, 0.5], game.k_weights)

            self.assertAllClose([[0.5863516330718994], [0.13367994129657745],
                                 [0.2799684405326843]], strategy)

            # This makes sense since the opponent must mix between the worst
            # two columns from utility_of_instance_given_action.
            self.assertAllClose([-0.186209, 1.060886, -1.095324], utilities)

    def test_batched_strategies(self):
        n = 5
        num_actions = 3

        utility_of_instance_given_action = np.random.normal(
            size=[1, num_actions, n]).astype('float32')

        strategies = np.random.uniform(
            size=[2, 1, num_actions]).astype('float32')
        strategies /= tf.reduce_sum(strategies, axis=-1, keepdims=True)

        with self.test_session():
            game_template = DeterministicKofnGameTemplate(1, n)
            kofn_utility = ContextualKofnGame(
                game_template.prob_ith_element_is_sampled,
                utility_of_instance_given_action, strategies).kofn_utility

            assert len(kofn_utility.shape) == 3
            assert kofn_utility.shape[0].value == 2
            assert kofn_utility.shape[1].value == 1
            assert kofn_utility.shape[2].value == num_actions

            utilities = tf.reduce_mean(kofn_utility, axis=1)

            self.assertAllClose(
                [
                    [
                        -0.2341533750295639,
                        0.5425600409507751,
                        -1.7249178886413574
                    ],
                    [
                        -0.2341533750295639,
                        0.5425600409507751,
                        -1.7249178886413574
                    ]
                ],
                utilities
            )  # yapf:disable

    def test_batched_utilities(self):
        n = 5
        num_actions = 3

        utility_of_instance_given_action = np.random.normal(
            size=[2, 1, num_actions, n]).astype('float32')

        strategy = np.random.uniform(size=[1, num_actions]).astype('float32')
        strategy /= tf.reduce_sum(strategy, axis=-1, keepdims=True)

        with self.test_session():
            game_template = DeterministicKofnGameTemplate(1, n)
            kofn_utility = ContextualKofnGame(
                game_template.prob_ith_element_is_sampled,
                utility_of_instance_given_action, strategy).kofn_utility

            assert len(kofn_utility.shape) == 3
            assert kofn_utility.shape[0].value == 2
            assert kofn_utility.shape[1].value == 1
            assert kofn_utility.shape[2].value == num_actions

            utilities = tf.reduce_mean(kofn_utility, axis=1)

            self.assertAllClose(
                [
                    [
                        -0.2341533750295639,
                        0.5425600409507751,
                        -1.7249178886413574
                    ],
                    [
                        -0.9080241,
                        -1.4247482,
                        -0.6006387
                    ]
                ],
                utilities
            )  # yapf:disable

    def test_batched_strategies_and_utilities(self):
        n = 5
        num_actions = 3

        utility_of_instance_given_action = np.random.normal(
            size=[2, 1, num_actions, n]).astype('float32')

        strategy = np.random.uniform(size=[2, 1, num_actions]).astype('float32')
        strategy /= tf.reduce_sum(strategy, axis=-1, keepdims=True)

        with self.test_session():
            game_template = DeterministicKofnGameTemplate(1, n)
            kofn_utility = ContextualKofnGame(
                game_template.prob_ith_element_is_sampled,
                utility_of_instance_given_action, strategy).kofn_utility

            assert len(kofn_utility.shape) == 3
            assert kofn_utility.shape[0].value == 2
            assert kofn_utility.shape[1].value == 1
            assert kofn_utility.shape[2].value == num_actions

            utilities = tf.reduce_mean(kofn_utility, axis=1)

            self.assertAllClose(
                [
                    [
                        -0.2341533750295639,
                        0.5425600409507751,
                        -1.7249178886413574
                    ],
                    [
                        -1.0128311,
                        -0.2257763,
                        -1.1509936
                    ]
                ],
                utilities
            )  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
