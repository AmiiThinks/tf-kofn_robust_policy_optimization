import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from tf_contextual_prediction_with_expert_advice import \
    l1_projection_to_simplex
from tf_kofn_robust_policy_optimization.regret_table import RegretTable, \
    PrRegretMatchingPlus, PrRegretTable
from tf_kofn_robust_policy_optimization.pr_mdp import pr_mdp_rollout, pr_mdp_expected_value, \
    pr_mdp_optimal_policy_and_value


class RegretTableTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.set_random_seed(42)

    def test_new_ir_table(self):
        horizon = 3
        num_states = 9
        num_actions = 10

        patient = RegretTable.new_ir_table(horizon, num_states, num_actions)

        self.assertAllEqual(
            tf.zeros((horizon * num_states, num_actions)),
            tf.convert_to_tensor(patient))

    def test_sequences_at_timestep_ir(self):
        horizon = 3
        num_states = 9
        num_actions = 10
        regrets = RegretTable.new_ir_table(horizon, num_states, num_actions)

        self.assertEqual(horizon * num_states, regrets.shape[0].value)
        self.assertEqual(num_actions, regrets.shape[1].value)

        for t in range(horizon):
            patient = RegretTable.sequences_at_timestep_ir(
                regrets, t, num_states)
            self.assertEqual(num_states, patient.shape[0].value)
            self.assertEqual(num_actions, patient.shape[1].value)
            self.assertAllEqual(
                tf.zeros((num_states, num_actions)),
                tf.convert_to_tensor(patient))

    def test_update_regrets_at_timestep_ir(self):
        horizon = 3
        num_states = 9
        num_actions = 10

        regrets = RegretTable.new_ir_table(horizon, num_states, num_actions)

        self.assertEqual(horizon * num_states, regrets.shape[0].value)
        self.assertEqual(num_actions, regrets.shape[1].value)

        updated_t = 1
        updated_regrets = RegretTable.update_regrets_at_timestep_ir(
            regrets,
            updated_t,
            num_states,
            tf.ones((num_states, num_actions)))  # yapf:disable
        self.assertEqual(horizon * num_states, updated_regrets.shape[0].value)

        for t in range(horizon):
            patient = RegretTable.sequences_at_timestep_ir(
                regrets, t, num_states)
            self.assertEqual(num_states, patient.shape[0].value)
            self.assertEqual(num_actions, patient.shape[1].value)
            if t == updated_t:
                self.assertAllEqual(
                    tf.ones((num_states, num_actions)),
                    tf.convert_to_tensor(patient))
            else:
                self.assertAllEqual(
                    tf.zeros((num_states, num_actions)),
                    tf.convert_to_tensor(patient))

    def test_new_pr_table(self):
        horizon = 3
        num_states = 9
        num_actions = 10

        patient = RegretTable.new_pr_table(horizon, num_states, num_actions)

        num_sequences = (num_states *
                         ((num_states * num_actions)**horizon - 1) /
                         (num_states * num_actions - 1))

        self.assertAllEqual(
            tf.zeros((num_sequences, num_actions)),
            tf.convert_to_tensor(patient))

    def test_sequences_at_timestep_pr(self):
        horizon = 3
        num_states = 9
        num_actions = 10
        regrets = RegretTable.new_pr_table(horizon, num_states, num_actions)

        def num_sequences(t):
            if t < 0: return 0
            return (
                num_states * ((num_states * num_actions)**(t + 1) - 1)
                / (num_states * num_actions - 1)
            )  # yapf:disable

        self.assertEqual(num_sequences(horizon - 1), regrets.shape[0].value)
        self.assertEqual(num_actions, regrets.shape[1].value)

        for t in range(horizon):
            patient = RegretTable.sequences_at_timestep_pr(
                regrets, t, num_states, num_actions)
            self.assertEqual(
                num_sequences(t) - num_sequences(t - 1),
                patient.shape[0].value)
            self.assertEqual(num_actions, patient.shape[1].value)

            x = tf.zeros((num_sequences(t) - num_sequences(t - 1),
                          num_actions))
            self.assertAllEqual(x, tf.convert_to_tensor(patient))

    def test_update_regrets_at_timestep_pr(self):
        horizon = 3
        num_states = 9
        num_actions = 10

        regrets = RegretTable.new_pr_table(horizon, num_states, num_actions)

        def num_sequences(t):
            if t < 0: return 0
            num_state_action_pairs = num_states * num_actions
            return (
                num_states * (num_state_action_pairs**(t + 1) - 1)
                / (num_state_action_pairs - 1)
            )  # yapf:disable

        self.assertEqual(num_sequences(horizon - 1), regrets.shape[0].value)
        self.assertEqual(num_actions, regrets.shape[1].value)

        updated_t = 2
        updated_regrets = (RegretTable.update_regrets_at_timestep_pr(
            regrets, updated_t, num_states, num_actions,
            tf.ones((num_sequences(updated_t) - num_sequences(updated_t - 1),
                     num_actions))))
        self.assertEqual(
            num_sequences(horizon - 1), updated_regrets.shape[0].value)

        for t in range(horizon):
            patient = RegretTable.sequences_at_timestep_pr(
                regrets, t, num_states, num_actions)
            self.assertEqual(
                num_sequences(t) - num_sequences(t - 1),
                patient.shape[0].value)
            self.assertEqual(num_actions, patient.shape[1].value)

            if t == updated_t:
                self.assertAllEqual(
                    tf.ones((num_sequences(updated_t) -
                             num_sequences(updated_t - 1), num_actions)),
                    tf.convert_to_tensor(patient))
            else:
                self.assertAllEqual(
                    tf.zeros((num_sequences(t) - num_sequences(t - 1),
                              num_actions)), tf.convert_to_tensor(patient))

    def test_cfr_update(self):
        horizon = 2
        num_states = 3
        num_actions = 2

        transition_model = l1_projection_to_simplex(
            np.random.normal(size=(num_states, num_actions, num_states)),
            axis=2)
        rewards = (
            np.random.uniform(size=(num_states, num_actions, num_states))
            * l1_projection_to_simplex(
                np.random.normal(
                    size=(num_states, num_actions, num_states), scale=5.0)
            )
        )  # yapf:disable
        root = l1_projection_to_simplex(tf.ones([3]))
        seq_chance_probs_node = pr_mdp_rollout(horizon, root, transition_model)
        weighted_rewards_node = seq_chance_probs_node * rewards

        for patient in [
                PrRegretTable(horizon, num_states, num_actions),
                PrRegretMatchingPlus(horizon, num_states, num_actions)
        ]:
            ev_node = pr_mdp_expected_value(horizon, num_states, num_actions,
                                            seq_chance_probs_node, rewards,
                                            patient.strat())
            x_ev = 0.263307
            self.assertAllClose(x_ev, ev_node)

            patient.update_regrets(
                patient.instantaneous_regrets(weighted_rewards_node))

            ev_node = pr_mdp_expected_value(horizon, num_states, num_actions,
                                            seq_chance_probs_node, rewards,
                                            patient.strat())

            self.assertGreater(ev_node, x_ev)

            patient.update_regrets(
                patient.instantaneous_regrets(weighted_rewards_node))

            ev_node = pr_mdp_expected_value(horizon, num_states, num_actions,
                                            seq_chance_probs_node, rewards,
                                            patient.strat())

            self.assertGreater(ev_node, x_ev)

            self.assertAllClose(
                pr_mdp_optimal_policy_and_value(
                    horizon,
                    num_states,
                    num_actions,
                    seq_chance_probs_node,
                    rewards
                )[1],
                ev_node)  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
