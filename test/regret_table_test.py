import tensorflow as tf
from amii_tf_mdp.regret_table import RegretTable


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

    def test_regrets_at_timestep_ir(self):
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
                patient = RegretTable.regrets_at_timestep_ir(
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
            self.assertEqual(horizon * num_states, regrets.shape[0].value)
            self.assertEqual(num_actions, regrets.shape[1].value)
            sess.run(tf.global_variables_initializer())

            updated_t = 1
            regrets = RegretTable.updated_regrets_at_timestep_ir(
                regrets,
                updated_t,
                num_states,
                tf.ones((num_states, num_actions))
            )
            self.assertEqual(horizon * num_states, regrets.shape[0].value)

            for t in range(horizon):
                patient = RegretTable.regrets_at_timestep_ir(
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
                    (num_states * num_actions)**(horizon + 1) - 1
                ) / (
                    num_states * num_actions - 1
                )
            )
            self.assertAllEqual(
                tf.zeros((num_sequences, num_actions)).eval(),
                tf.convert_to_tensor(patient).eval()
            )

    def test_regrets_at_timestep_pr(self):
        with self.test_session() as sess:
            horizon = 3
            num_states = 9
            num_actions = 10
            regrets = RegretTable.new_pr_table(
                horizon, num_states, num_actions
            )

            def num_sequences(t):
                return num_states * (
                    (num_states * num_actions)**(t + 1) - 1
                ) / (
                    num_states * num_actions - 1
                )

            self.assertEqual(num_sequences(horizon), regrets.shape[0].value)
            self.assertEqual(num_actions, regrets.shape[1].value)

            sess.run(tf.global_variables_initializer())

            for t in range(horizon):
                patient = RegretTable.regrets_at_timestep_pr(
                    regrets,
                    t,
                    num_states,
                    num_actions
                )
                self.assertEqual(
                    num_sequences(t + 1) - num_sequences(t),
                    patient.shape[0].value
                )
                self.assertEqual(num_actions, patient.shape[1].value)
                self.assertAllEqual(
                    tf.zeros(
                        (
                            num_sequences(t + 1) - num_sequences(t),
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
                return num_states * (
                    (num_states * num_actions)**(t + 1) - 1
                ) / (
                    num_states * num_actions - 1
                )

            self.assertEqual(num_sequences(horizon), regrets.shape[0].value)
            self.assertEqual(num_actions, regrets.shape[1].value)

            updated_t = 2
            regrets = RegretTable.updated_regrets_at_timestep_pr(
                regrets,
                updated_t,
                num_states,
                num_actions,
                tf.ones(
                    (
                        num_sequences(updated_t + 1) -
                        num_sequences(updated_t),
                        num_actions
                    )
                )
            )
            self.assertEqual(num_sequences(horizon), regrets.shape[0].value)

            sess.run(tf.global_variables_initializer())

            for t in range(horizon):
                patient = RegretTable.regrets_at_timestep_pr(
                    regrets,
                    t,
                    num_states,
                    num_actions
                )
                self.assertEqual(
                    num_sequences(t + 1) - num_sequences(t),
                    patient.shape[0].value
                )
                self.assertEqual(num_actions, patient.shape[1].value)

                if t == updated_t:
                    self.assertAllEqual(
                        tf.ones(
                            (
                                num_sequences(updated_t + 1) -
                                num_sequences(updated_t),
                                num_actions
                            )
                        ).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )
                else:
                    self.assertAllEqual(
                        tf.zeros(
                            (
                                num_sequences(t + 1) - num_sequences(t),
                                num_actions
                            )
                        ).eval(),
                        tf.convert_to_tensor(patient).eval()
                    )


if __name__ == '__main__':
    tf.test.main()
