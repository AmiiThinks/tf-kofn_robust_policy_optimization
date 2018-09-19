import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.robust import world_weights
import tf_kofn_robust_policy_optimization.robust.kofn as patient
from tf_kofn_robust_policy_optimization.pr_mdp import \
    pr_mdp_rollout, pr_mdp_evs


class KofnTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)
        np.random.seed(42)

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
        mdp_weights = world_weights(n_weights, k_weights, evs)
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

    def test_determinstic_k_of_n_game_template_yml(self):
        n = 5
        k = 2
        game_template = patient.DeterministicKofnGameTemplate(k, n)

        for i in range(8):
            self.assertEqual(
                ' ' * i + "n: {}\n".format(n) + ' ' * i + "k: {}".format(k),
                game_template.to_yml(i))


if __name__ == '__main__':
    tf.test.main()
