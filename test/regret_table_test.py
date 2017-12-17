import tensorflow as tf
import numpy as np
from amii_tf_nn.projection import l1_projection_to_simplex
from k_of_n_mdp_policy_opt.regret_table import RegretTable, \
    PrRegretMatchingPlus, PrRegretTable
from k_of_n_mdp_policy_opt.pr_mdp import pr_mdp_rollout, pr_mdp_expected_value, \
    pr_mdp_optimal_policy_and_value


class RegretTableTest(tf.test.TestCase):
    def test_new_ir_table(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            patient = RegretTable.new_ir_table(
                horizon, num_states, num_actions
            )
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(
                tf.zeros((horizon * num_states, num_actions)).eval(),
                tf.convert_to_tensor(patient).eval()
            )

    def test_sequences_at_timestep_ir(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            regrets = RegretTable.new_ir_table(
                horizon, num_states, num_actions
            )
            self.assertEqual(horizon * num_states, regrets.shape[0].value)
            self.assertEqual(num_actions, regrets.shape[1].value)
            sess.run(tf.global_variables_initializer())

            for t in range(horizon):
                patient = RegretTable.sequences_at_timestep_ir(
                    regrets,
                    t,
                    num_states
                )
                self.assertEqual(num_states, patient.shape[0].value)
                self.assertEqual(num_actions, patient.shape[1].value)
                self.assertAllEqual(
                    tf.zeros((num_states, num_actions)).eval(),
                    tf.convert_to_tensor(patient).eval()
                )

    def test_updated_regrets_at_timestep_ir(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            regrets = RegretTable.new_ir_table(
                horizon, num_states, num_actions
            )
            self.assertEqual(
                horizon * num_states,
                regrets.shape[0].value
            )
            self.assertEqual(num_actions, regrets.shape[1].value)
            sess.run(tf.global_variables_initializer())

            updated_t = 1
            updated_regrets = (
                RegretTable.updated_regrets_at_timestep_ir(
                    regrets,
                    updated_t,
                    num_states,
                    tf.ones((num_states, num_actions))
                )
            )
            self.assertEqual(
                horizon * num_states,
                updated_regrets.shape[0].value
            )
            sess.run(updated_regrets)

            for t in range(horizon):
                patient = RegretTable.sequences_at_timestep_ir(
                    regrets,
                    t,
                    num_states
                )
                self.assertEqual(num_states, patient.shape[0].value)
                self.assertEqual(num_actions, patient.shape[1].value)
                if t == updated_t:
                    self.assertAllEqual(
                        tf.ones((num_states, num_actions)).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )
                else:
                    self.assertAllEqual(
                        tf.zeros((num_states, num_actions)).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )

    def test_new_pr_table(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            patient = RegretTable.new_pr_table(
                horizon, num_states, num_actions
            )
            sess.run(tf.global_variables_initializer())
            num_sequences = (
                num_states * (
                    (num_states * num_actions)**horizon - 1
                ) / (
                    num_states * num_actions - 1
                )
            )
            self.assertAllEqual(
                tf.zeros((num_sequences, num_actions)).eval(),
                tf.convert_to_tensor(patient).eval()
            )

    def test_sequences_at_timestep_pr(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            regrets = RegretTable.new_pr_table(
                horizon, num_states, num_actions
            )

            def num_sequences(t):
                if t < 0: return 0
                return num_states * (
                    (num_states * num_actions)**(t + 1) - 1
                ) / (
                    num_states * num_actions - 1
                )

            self.assertEqual(
                num_sequences(horizon - 1),
                regrets.shape[0].value
            )
            self.assertEqual(num_actions, regrets.shape[1].value)

            sess.run(tf.global_variables_initializer())

            for t in range(horizon):
                patient = RegretTable.sequences_at_timestep_pr(
                    regrets,
                    t,
                    num_states,
                    num_actions
                )
                self.assertEqual(
                    num_sequences(t) - num_sequences(t - 1),
                    patient.shape[0].value
                )
                self.assertEqual(num_actions, patient.shape[1].value)
                self.assertAllEqual(
                    tf.zeros(
                        (
                            num_sequences(t) - num_sequences(t - 1),
                            num_actions
                        )
                    ).eval(),
                    tf.convert_to_tensor(patient).eval()
                )

    def test_updated_regrets_at_timestep_pr(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            regrets = RegretTable.new_pr_table(
                horizon, num_states, num_actions
            )

            def num_sequences(t):
                if t < 0: return 0
                return num_states * (
                    (num_states * num_actions)**(t + 1) - 1
                ) / (
                    num_states * num_actions - 1
                )

            self.assertEqual(
                num_sequences(horizon - 1),
                regrets.shape[0].value
            )
            self.assertEqual(num_actions, regrets.shape[1].value)

            updated_t = 2
            updated_regrets = (
                RegretTable.updated_regrets_at_timestep_pr(
                    regrets,
                    updated_t,
                    num_states,
                    num_actions,
                    tf.ones(
                        (
                            num_sequences(updated_t) -
                            num_sequences(updated_t - 1),
                            num_actions
                        )
                    )
                )
            )
            self.assertEqual(
                num_sequences(horizon - 1),
                updated_regrets.shape[0].value
            )

            sess.run(tf.global_variables_initializer())
            sess.run(updated_regrets)

            for t in range(horizon):
                patient = RegretTable.sequences_at_timestep_pr(
                    regrets,
                    t,
                    num_states,
                    num_actions
                )
                self.assertEqual(
                    num_sequences(t) - num_sequences(t - 1),
                    patient.shape[0].value
                )
                self.assertEqual(num_actions, patient.shape[1].value)

                if t == updated_t:
                    self.assertAllEqual(
                        tf.ones(
                            (
                                num_sequences(updated_t) -
                                num_sequences(updated_t - 1),
                                num_actions
                            )
                        ).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )
                else:
                    self.assertAllEqual(
                        tf.zeros(
                            (
                                num_sequences(t) - num_sequences(t - 1),
                                num_actions
                            )
                        ).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )

    def test_cfr_update(self):
        with self.test_session() as sess:
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
                        )
                    )
                )
            )
            rewards = (
                np.random.uniform(
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
            )
            root = l1_projection_to_simplex(tf.ones([3]))
            seq_chance_probs_node = pr_mdp_rollout(
                horizon,
                root,
                transition_model
            )
            weighted_rewards_node = seq_chance_probs_node * rewards

            for patient in [
                PrRegretTable(horizon, num_states, num_actions),
                PrRegretMatchingPlus(horizon, num_states, num_actions)
            ]:
                patient.regrets.initializer.run()

                ev_node = pr_mdp_expected_value(
                    horizon,
                    num_states,
                    num_actions,
                    seq_chance_probs_node,
                    rewards,
                    patient.strat
                )
                x_ev = 0.78666484
                self.assertAlmostEqual(x_ev, sess.run(ev_node), places=6)

                patient_cfr_update = patient.updated_regrets(
                    patient.instantaneous_regrets(
                        weighted_rewards_node
                    )
                )
                sess.run(patient_cfr_update)
                self.assertGreater(sess.run(ev_node), x_ev)

                sess.run(patient_cfr_update)
                self.assertGreater(sess.run(ev_node), x_ev)

                self.assertAlmostEqual(
                    sess.run(
                        pr_mdp_optimal_policy_and_value(
                            horizon,
                            num_states,
                            num_actions,
                            seq_chance_probs_node,
                            rewards
                        )[1]
                    ),
                    sess.run(ev_node),
                    places=6
                )


if __name__ == '__main__':
    tf.test.main()
