import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
import tf_kofn_robust_policy_optimization.robust.kofn as patient
from tf_kofn_robust_policy_optimization.pr_mdp import \
    pr_mdp_rollout, pr_mdp_evs


class KofnTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

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

    def test_bind(self):
        horizon = 2
        num_mdps = 2
        roots = [tf.constant([0.1, 0.9])] * num_mdps
        transition_models = [
            tf.constant(
                [
                    [
                        [0.1, 0.9],
                        [0.2, 0.8]
                    ],
                    [
                        [0.3, 0.7],
                        [0.4, 0.6]
                    ]
                ]
            )
        ] * num_mdps  # yapf:disable

        reward_models = [
            tf.constant(
                [
                    [
                        [1.0, 9.0],
                        [2.0, 8.0]
                    ],
                    [
                        [3.0, 7.0],
                        [4.0, 6.0]
                    ]
                ]
            )
        ] * num_mdps  # yapf:disable

        chance_prob_sequences = [
            pr_mdp_rollout(horizon, roots[i], transition_models[i])
            for i in range(len(roots))
        ]

        n_weights = [0.0] * (num_mdps - 1) + [1.0]
        k_weights = [1.0] + [0.0] * (num_mdps - 1)
        strat = [
            # First timestep
            [0.1, 0.9],
            [0.2, 0.8],
            # Second timestep
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
            [1.0, 0]
        ]  # yapf:disable

        evs = pr_mdp_evs(horizon, chance_prob_sequences, reward_models, strat)
        mdp_weights = patient.world_weights(n_weights, k_weights, evs)
        kofn_ev = patient.kofn_ev(evs, mdp_weights)

        self.assertAllClose([12.0361805, 12.0361805], evs)
        self.assertAllClose([1.0, 0.0], mdp_weights)
        self.assertAllClose(12.0361805, kofn_ev)

    def test_determinstic_k_of_n_game_template_creation(self):
        n = 5
        k = 2
        game_template = patient.DeterministicKofnGameTemplate(k, n)
        self.assertEqual(n, game_template.num_sampled_worlds())
        self.assertEqual('{}-of-{} template'.format(k, n), str(game_template))

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
            game_template = patient.DeterministicKofnGameTemplate(1, n)
            utilities = tf.reduce_mean(
                patient.ContextualKofnGame(
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

        game_template = patient.DeterministicKofnGameTemplate(2, n)

        utility_of_instance_given_action = tf.constant(
            utility_of_instance_given_action, name='normal_random_utils')
        game = patient.ContextualKofnGame(
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


if __name__ == '__main__':
    tf.test.main()
