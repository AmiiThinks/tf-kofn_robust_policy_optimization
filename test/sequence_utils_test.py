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
            prob_sequence_action_and_state = tf.constant([[[[0.5, 0.4, 0.1]]]])
            num_actions = 2
            self.assertAllClose(
                [
                    [[0.5, 0.5],
                     [0.4, 0.4],
                     [0.1, 0.1]]
                ],
                patient.prob_sequence_state_and_action(
                    prob_sequence_action_and_state,
                    num_actions=num_actions
                ).eval()
            )

            self.assertAllClose(
                [
                    [
                        [1.11, 1.11],
                        [1.12, 1.12],
                        [1.13, 1.13]
                    ],
                    [
                        [1.21, 1.21],
                        [1.22, 1.22],
                        [1.23, 1.23]
                    ],
                    [
                        [2.11, 2.11],
                        [2.12, 2.12],
                        [2.13, 2.13]
                    ],
                    [
                        [2.21, 2.21],
                        [2.22, 2.22],
                        [2.23, 2.23]
                    ],
                    [
                        [3.11, 3.11],
                        [3.12, 3.12],
                        [3.13, 3.13]
                    ],
                    [
                        [3.21, 3.21],
                        [3.22, 3.22],
                        [3.23, 3.23]
                    ]
                ],
                patient.prob_sequence_state_and_action(
                    tf.constant(
                        [
                            [
                                [
                                    [1.11, 1.12, 1.13],
                                    [1.21, 1.22, 1.23]
                                ],
                                [
                                    [2.11, 2.12, 2.13],
                                    [2.21, 2.22, 2.23]
                                ],
                                [
                                    [3.11, 3.12, 3.13],
                                    [3.21, 3.22, 3.23]
                                ]
                            ]
                        ]
                    ),
                    num_actions=num_actions
                ).eval()
            )

    def test_prob_next_sequence_state_action_and_next_state(self):
        with self.test_session():
            prob_sequence_action_and_state = tf.constant([[[[0.5, 0.4, 0.1]]]])
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
            x_prob_next_sequence_state_action_and_next_state = [
                [
                    [
                        [2.05500007, 2.05999994, 2.06500006],
                        [2.55500007, 2.55999994, 2.56500006]
                    ],
                    [
                        [1.68400002, 1.68799996, 1.69200003],
                        [2.08400011, 2.08800006, 2.09200001]
                    ],
                    [
                        [0.43099999, 0.43200001, 0.433],
                        [0.53100002, 0.53200001, 0.53299999]
                    ]
                ]
            ]

            self.assertAllClose(
                x_prob_next_sequence_state_action_and_next_state,
                patient.prob_next_sequence_state_action_and_next_state(
                    transition_model,
                    prob_sequence_action_and_state,
                    num_actions=num_actions
                ).eval()
            )

            prob_sequence_action_and_state = tf.constant(
                [
                    [
                        [1.11, 1.12],
                        [1.21, 1.22]
                    ],
                    [
                        [2.11, 2.12],
                        [2.21, 2.22]
                    ]
                ]
            )
            transition_model = [
                [
                    [4.11, 4.12],
                    [5.11, 5.12]
                ],
                [
                    [4.21, 4.22],
                    [5.21, 5.22]
                ]
            ]
            x_prob_next_sequence_state_action_and_next_state = [
                [
                    [
                        [4.56210041, 4.57319975],
                        [5.67210007, 5.68319988]
                    ],
                    [
                        [4.71519995, 4.7263999],
                        [5.83519983, 5.84639978]
                    ]
                ],
                [
                    [
                        [4.97310019, 4.98519993],
                        [6.18310022, 6.19519997]
                    ],
                    [
                        [5.13619995, 5.14839983],
                        [6.35620022, 6.3684001]
                    ]
                ],
                [
                    [
                        [8.67210007, 8.69319916],
                        [10.78209972, 10.80319881]
                    ],
                    [
                        [8.92519951, 8.94639874],
                        [11.04519939, 11.06639862]
                    ]
                ],
                [
                    [
                        [9.08310032, 9.10519981],
                        [11.29310036, 11.31519985]
                    ],
                    [
                        [9.34619999, 9.36839962],
                        [11.56620026, 11.58839989]
                    ]
                ]
            ]

            self.assertAllClose(
                x_prob_next_sequence_state_action_and_next_state,
                patient.prob_next_sequence_state_action_and_next_state(
                    transition_model,
                    prob_sequence_action_and_state,
                    num_actions=num_actions
                ).eval()
            )


if __name__ == '__main__':
    tf.test.main()
