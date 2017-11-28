import tensorflow as tf
from amii_tf_mdp.environments.gridworld import Gridworld


class GridworldTest(tf.test.TestCase):
    def test_state_op(self):
        with self.test_session():
            num_rows = 2
            num_columns = 3
            patient = Gridworld(num_rows, num_columns)
            self.assertAllEqual(
                [
                    [1, 0, 0],
                    [0, 0, 0]
                ],
                tf.reshape(
                    patient.indicator_state_op(0, 0),
                    [num_rows, num_columns]
                ).eval()
            )
            self.assertAllEqual(
                [
                    [0, 1, 0],
                    [0, 0, 0]
                ],
                tf.reshape(
                    patient.indicator_state_op(0, 1),
                    [num_rows, num_columns]
                ).eval()
            )
            self.assertAllEqual(
                [
                    [0, 0, 0],
                    [1, 0, 0]
                ],
                tf.reshape(
                    patient.indicator_state_op(1, 0),
                    [num_rows, num_columns]
                ).eval()
            )
            self.assertAllEqual(
                [
                    [0, 0, 0],
                    [0, 0, 1]
                ],
                tf.reshape(
                    patient.indicator_state_op(1, 2),
                    [num_rows, num_columns]
                ).eval()
            )

    def test_cardinal_transition_model_op(self):
        with self.test_session():
            num_rows = 2
            num_columns = 3
            num_actions = 4

            patient = Gridworld(
                num_rows,
                num_columns
            )

            self.assertAllEqual(
                ['North', 'East', 'South', 'West'],
                Gridworld.cardinal_direction_names()
            )
            state_action_state_model = patient.cardinal_transition_model_op()

            self.assertAllEqual(
                [
                    [  # North
                        [1, 0, 0],
                        [0, 0, 0]
                    ],
                    [  # East
                        [0, 1, 0],
                        [0, 0, 0]
                    ],
                    [  # South
                        [0, 0, 0],
                        [1, 0, 0]
                    ],
                    [  # West
                        [1, 0, 0],
                        [0, 0, 0]
                    ]
                ],
                tf.reshape(
                    state_action_state_model[0, :, :],
                    [num_actions, num_rows, num_columns]
                ).eval()
            )

            self.assertAllEqual(
                [
                    [  # North
                        [0, 1, 0],
                        [0, 0, 0]
                    ],
                    [  # East
                        [0, 0, 1],
                        [0, 0, 0]
                    ],
                    [  # South
                        [0, 0, 0],
                        [0, 1, 0]
                    ],
                    [  # West
                        [1, 0, 0],
                        [0, 0, 0]
                    ]
                ],
                tf.reshape(
                    state_action_state_model[1, :, :],
                    [num_actions, num_rows, num_columns]
                ).eval()
            )

            self.assertAllEqual(
                [
                    [  # North
                        [1, 0, 0],
                        [0, 0, 0]
                    ],
                    [  # East
                        [0, 0, 0],
                        [0, 1, 0]
                    ],
                    [  # South
                        [0, 0, 0],
                        [1, 0, 0]
                    ],
                    [  # West
                        [0, 0, 0],
                        [1, 0, 0]
                    ]
                ],
                tf.reshape(
                    state_action_state_model[3, :, :],
                    [num_actions, num_rows, num_columns]
                ).eval()
            )
            self.assertAllEqual(
                [
                    [  # North
                        [0, 0, 1],
                        [0, 0, 0]
                    ],
                    [  # East
                        [0, 0, 0],
                        [0, 0, 1]
                    ],
                    [  # South
                        [0, 0, 0],
                        [0, 0, 1]
                    ],
                    [  # West
                        [0, 0, 0],
                        [0, 1, 0]
                    ]
                ],
                tf.reshape(
                    state_action_state_model[5, :, :],
                    [num_actions, num_rows, num_columns]
                ).eval()
            )


if __name__ == '__main__':
    tf.test.main()
