import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_kofn_robust_policy_optimization.pr_mdp import \
    pr_mdp_rollout, \
    pr_mdp_expected_value, \
    pr_mdp_optimal_policy_and_value
from tf_kofn_robust_policy_optimization.utils.sequence import num_pr_sequences
from tf_contextual_prediction_with_expert_advice import \
    l1_projection_to_simplex


class PrMdpTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_update(self):
        horizon = 2
        num_states = 3
        num_actions = 2

        transition_model = tf.transpose(
            l1_projection_to_simplex(
                np.random.normal(
                    size=(num_states, num_actions, num_states)
                ).T
            ))  # yapf:disable
        root = l1_projection_to_simplex(tf.constant([1, 2, 3.0]))
        x_update_1 = np.array(
            [
                [
                    [
                        [0.07233965396881104, 0.0, 0.09432701766490936],
                        [0.1666666716337204, 0.0, 0.0]
                    ],
                    [
                        [0.22432182729244232, 0.10901150107383728, 0.0],
                        [0.3333333432674408, 0.0, 0.0]
                    ],
                    [
                        [0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.5]
                    ]
                ]
            ]
        )  # yapf:disable

        with self.test_session():
            patient = pr_mdp_rollout(horizon, root,
                                     transition_model)[:1, :, :, :]

            self.assertAllClose(x_update_1, patient)
            # TODO Check that the rest of the sequences were updated properly.

    def test_expected_value(self):
        horizon = 2
        num_states = 3
        num_actions = 2

        transition_model = tf.transpose(
            l1_projection_to_simplex(
                tf.zeros((num_states, num_actions, num_states))))
        rewards = np.ones((num_states, num_actions, num_states))
        root = l1_projection_to_simplex(tf.ones([3]))
        uniform_random_strat = tf.transpose(
            l1_projection_to_simplex(
                tf.zeros(
                    (
                        num_actions,
                        num_pr_sequences(
                            horizon - 1,
                            num_states,
                            num_actions
                        )
                    ))))  # yapf:disable
        self.assertAllClose(
            2.0,
            pr_mdp_expected_value(
                horizon,
                num_states,
                num_actions,
                pr_mdp_rollout(horizon, root, transition_model),
                rewards,
                uniform_random_strat
            ))  # yapf:disable

    def test_expected_value_2(self):
        horizon = 2
        num_states = 3
        num_actions = 2

        transition_model = tf.transpose(
            l1_projection_to_simplex(
                np.random.normal(
                    size=(
                        num_states,
                        num_actions,
                        num_states
                    ))))  # yapf:disable
        rewards = (
            np.random.normal(
                loc=-1.0,
                scale=1.0,
                size=(num_states, num_actions, num_states)
            ) *
            tf.transpose(
                l1_projection_to_simplex(
                    np.random.normal(
                        size=(
                            num_states,
                            num_actions,
                            num_states
                        ),
                        scale=5.0
                    )
                )
            )
        )  # yapf:disable
        root = l1_projection_to_simplex(tf.ones([3]))
        uniform_random_strat = tf.transpose(
            l1_projection_to_simplex(
                tf.zeros(
                    (
                        num_actions,
                        num_pr_sequences(
                            horizon - 1,
                            num_states,
                            num_actions
                        )))))  # yapf:disable
        ev = pr_mdp_expected_value(
            horizon,
            num_states,
            num_actions,
            pr_mdp_rollout(horizon, root, transition_model),
            rewards,
            uniform_random_strat)  # yapf:disable
        self.assertAllClose(-0.75842464, ev)
        self.assertAllClose(-0.75842464, ev)

    def test_best_response(self):
        horizon = 2
        num_states = 3
        num_actions = 2

        transition_model = tf.transpose(
            l1_projection_to_simplex(
                np.random.normal(
                    size=(
                        num_states,
                        num_actions,
                        num_states
                    ))))  # yapf:disable
        rewards = (
            np.random.normal(
                loc=-1.0,
                scale=1.0,
                size=(num_states, num_actions, num_states)
            ) *
            tf.transpose(
                l1_projection_to_simplex(
                    np.random.normal(
                        size=(
                            num_states,
                            num_actions,
                            num_states
                        ),
                        scale=5.0
                    )
                )
            )
        )  # yapf:disable
        root = l1_projection_to_simplex(tf.ones([3]))
        br_strat, br_val = pr_mdp_optimal_policy_and_value(
            horizon,
            num_states,
            num_actions,
            pr_mdp_rollout(horizon, root, transition_model),
            rewards)  # yapf:disable

        self.assertAllClose(-0.24534382, br_val)


if __name__ == '__main__':
    tf.test.main()
