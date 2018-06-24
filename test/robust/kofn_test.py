import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import tf_kofn_robust_policy_optimization.robust.kofn as patient
from tf_kofn_robust_policy_optimization.pr_mdp import pr_mdp_rollout, pr_mdp_evs


class KOfNTest(tf.test.TestCase):
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
        self.assertAllClose([0.0, 1.0], mdp_weights)
        self.assertAllClose(12.0361805, kofn_ev)


if __name__ == '__main__':
    tf.test.main()
