import tensorflow as tf
import numpy as np
from amii_tf_nn.projection import l1_projection_to_simplex
from amii_tf_mdp.regret_table import RegretTable, \
    PrRegretMatchingPlus, PrRegretTable
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp
from amii_tf_mdp.tf_node import BoundTfNode


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

            transition_model = sess.run(
                tf.transpose(
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
            )
            rewards = sess.run(
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
            x_root = sess.run(l1_projection_to_simplex(tf.ones([3])))

            state = PrUncertainMdp(horizon, num_states, num_actions)
            sess.run(tf.global_variables_initializer())

            weighted_rewards_node = state.bound_sequences_node(
                transition_model,
                root=x_root
            )
            weighted_rewards_node.components *= rewards

            for patient in [
                PrRegretTable.from_mdp(state),
                PrRegretMatchingPlus.from_mdp(state)
            ]:
                patient.regrets.initializer.run()

                ev_node = state.bound_expected_value_node(
                    transition_model,
                    rewards,
                    root=x_root,
                    strat=sess.run(patient.strat)
                )
                x_ev = ev_node.run(sess)
                self.assertAlmostEqual(0.78666484, x_ev)

                patient_cfr_update = BoundTfNode(
                    [
                        patient.updated_regrets(
                            patient.instantaneous_regrets(
                                weighted_rewards_node.components
                            )
                        )
                    ],
                    weighted_rewards_node.feed_dict
                )
                patient_cfr_update.run(sess)

                next_ev_node = state.bound_expected_value_node(
                    transition_model,
                    rewards,
                    root=x_root,
                    strat=sess.run(patient.strat)
                )
                self.assertGreater(next_ev_node.run(sess), x_ev)

                patient_cfr_update.run(sess)

                next_ev_node = state.bound_expected_value_node(
                    transition_model,
                    rewards,
                    root=x_root,
                    strat=sess.run(patient.strat)
                )
                next_ev = next_ev_node.run(sess)
                self.assertGreater(next_ev, x_ev)

                self.assertAlmostEqual(
                    state.bound_br_value_node(
                        transition_model,
                        rewards,
                        root=x_root
                    ).run(sess),
                    next_ev,
                    places=6
                )


if __name__ == '__main__':
    tf.test.main()
