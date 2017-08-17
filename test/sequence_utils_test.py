import tensorflow as tf
import amii_tf_mdp.sequence_utils as patient


class SequenceUtilsTest(tf.test.TestCase):
    def test_num_pr_sequences_at_timestep(self):
        self.assertEqual(
            patient.num_pr_sequences(0, 3, 2),
            patient.num_pr_sequences_at_timestep(0, 3, 2)
        )
        self.assertEqual(
            (
                patient.num_pr_sequences(1, 3, 2) -
                patient.num_pr_sequences(0, 3, 2)
            ),
            patient.num_pr_sequences_at_timestep(1, 3, 2)
        )
        self.assertEqual(
            (
                patient.num_pr_sequences(2, 3, 2) -
                patient.num_pr_sequences(1, 3, 2)
            ),
            patient.num_pr_sequences_at_timestep(2, 3, 2)
        )

    def test_prob_sequence_state_and_action(self):
        with self.test_session():
            prob_sequence_and_state = tf.constant([[0.5, 0.4, 0.1]])
            num_actions = 2
            self.assertAllClose(
                [
                    [[0.5, 0.5],
                     [0.4, 0.4],
                     [0.1, 0.1]]
                ],
                patient.prob_sequence_state_and_action(
                    prob_sequence_and_state,
                    num_actions=num_actions
                ).eval()
            )

            self.assertAllClose(
                [
                    [[1.1, 1.1],
                     [2.1, 2.1],
                     [3.1, 3.1]],
                    [[1.2, 1.2],
                     [2.2, 2.2],
                     [3.2, 3.2]],
                    [[1.3, 1.3],
                     [2.3, 2.3],
                     [3.3, 3.3]]
                ],
                patient.prob_sequence_state_and_action(
                    tf.constant(
                        [[1.1, 2.1, 3.1],
                         [1.2, 2.2, 3.2],
                         [1.3, 2.3, 3.3]]
                    ),
                    num_actions=num_actions
                ).eval()
            )

    def test_prob_next_sequence_and_next_state(self):
        with self.test_session():
            prob_sequence_and_state = tf.constant([[0.5, 0.4, 0.1]])
            num_actions = 2
            transition_model = [
                [
                    [4.11, 4.12, 4.13],
                    [5.11, 5.12, 5.13]
                ],
                [
                    [4.21, 4.22, 4.23],
                    [5.21, 5.22, 5.23]
                ],
                [
                    [4.31, 4.32, 4.33],
                    [5.31, 5.32, 5.33]
                ]
            ]
            x_prob_next_sequence_and_next_state = [
                [2.05500007, 2.05999994, 2.06500006],
                [2.55500007, 2.55999994, 2.56500006],
                [1.68400002, 1.68799996, 1.69200003],
                [2.08400011, 2.08800006, 2.09200001],
                [0.43099999, 0.43200001, 0.433],
                [0.53100002, 0.53200001, 0.53299999]
            ]

            self.assertAllClose(
                x_prob_next_sequence_and_next_state,
                patient.prob_next_sequence_and_next_state(
                    transition_model,
                    prob_sequence_and_state,
                    num_actions=num_actions
                ).eval()
            )

            prob_sequence_and_state = tf.constant(
                [[1.1, 2.1, 3.1],
                 [1.2, 2.2, 3.2],
                 [1.3, 2.3, 3.3]]
            )
            x_prob_next_sequence_and_next_state = [
                [4.52100039, 4.53200006, 4.54300022],
                [5.62100029, 5.63199997, 5.64300013],
                [8.8409996, 8.86199951, 8.88299942],
                [10.94099998, 10.96199894, 10.9829998 ],
                [13.36099911, 13.3920002, 13.42299938],
                [16.46099854, 16.49200058, 16.52299881],
                [4.93200016, 4.94400024, 4.95600033],
                [6.13200045, 6.14400005, 6.15600061],
                [9.26200008, 9.28399944, 9.30600071],
                [11.46199989, 11.48400021, 11.50600052],
                [13.79199982, 13.82400036, 13.85599995],
                [16.99200058, 17.02400017, 17.05599976],
                [5.34299994, 5.35599947, 5.36899996],
                [6.64300013, 6.65599966, 6.66899967],
                [9.68299961, 9.70599937, 9.72900009],
                [11.9829998, 12.00599957, 12.02899933],
                [14.22299957, 14.25600052, 14.28899956],
                [17.52299881, 17.55599976, 17.58899879]
            ]

            self.assertAllClose(
                x_prob_next_sequence_and_next_state,
                patient.prob_next_sequence_and_next_state(
                    transition_model,
                    prob_sequence_and_state,
                    num_actions=num_actions
                ).eval()
            )


if __name__ == '__main__':
    tf.test.main()
