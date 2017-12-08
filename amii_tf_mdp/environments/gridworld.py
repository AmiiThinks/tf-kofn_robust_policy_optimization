import tensorflow as tf


class Gridworld(object):
    @staticmethod
    def cardinal_direction_names():
        return ['North', 'East', 'South', 'West']

    @staticmethod
    def cardinal_direction_transformations():
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @staticmethod
    def num_cardinal_directions(): return 4

    def __init__(self, num_rows, num_columns):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self._cardinal_transition_model_ops = {}

    def indicator_state_op(self, *state):
        return tf.reshape(
            tf.scatter_nd(
                [state],
                [1],
                shape=(self.num_rows, self.num_columns)
            ),
            [self.num_rows * self.num_columns, 1]
        )

    def cardinal_transition_model_op(self, sink=None):
        if sink not in self._cardinal_transition_model_ops:
            self._cardinal_transition_model_ops[sink] = tf.transpose(
                tf.reshape(
                    tf.transpose(
                        self._cardinal_grid_transition_model_op(sink),
                        [2, 0, 1, 3, 4]
                    ),
                    [
                        self.num_cardinal_directions(),
                        self.num_rows * self.num_columns,
                        self.num_rows * self.num_columns
                    ]
                ),
                [1, 0, 2]
            )
        return self._cardinal_transition_model_ops[sink]

    def cardinal_reward_model_op(
        self,
        row_column_indices,
        destination_rewards,
        sink=None
    ):
        state_action_state_model_op = self.cardinal_transition_model_op(sink)
        if sink is not None:
            # TODO This is slow
            sink_state = tf.where(
                tf.greater(tf.squeeze(self.indicator_state_op(*sink)), 0)
            )[0, 0]
            state_action_state_model_op = (
                state_action_state_model_op -
                tf.scatter_nd(
                    [
                        (sink_state, a, s)
                        for a in range(self.num_cardinal_directions())
                        for s in range(self.num_rows * self.num_columns)
                    ],
                    tf.reshape(
                        state_action_state_model_op[sink_state, :, :],
                        [
                            self.num_cardinal_directions() *
                            self.num_rows *
                            self.num_columns
                        ]
                    ),
                    shape=state_action_state_model_op.shape
                )
            )
        state_actions_to_state_model_op = tf.reshape(
            state_action_state_model_op,
            (
                (
                    self.num_rows *
                    self.num_columns *
                    self.num_cardinal_directions()
                ),
                self.num_rows * self.num_columns
            )
        )
        state_rewards = tf.reshape(
            tf.scatter_nd(
                row_column_indices,
                destination_rewards,
                shape=(self.num_rows, self.num_columns)
            ),
            (self.num_rows * self.num_columns, 1)
        )
        return state_actions_to_state_model_op @ state_rewards

    def _cardinal_grid_transition_model_op(self, sink=None):
        indices = []
        movement = tf.constant(
            self.__class__.cardinal_direction_transformations()
        )
        row_movement = movement[:, 0]
        column_movement = movement[:, 1]
        num_actions = row_movement.shape[0].value
        for row in range(self.num_rows):
            for column in range(self.num_columns):
                if (
                    sink is not None and
                    sink[0] == row and
                    sink[1] == column
                ):
                    successors = [
                        tf.constant(row, shape=[num_actions]),
                        tf.constant(column, shape=[num_actions]),
                        tf.range(num_actions),
                        tf.constant(sink[0], shape=[num_actions]),
                        tf.constant(sink[1], shape=[num_actions])
                    ]
                else:
                    successors = [
                        tf.constant(row, shape=[num_actions]),
                        tf.constant(column, shape=[num_actions]),
                        tf.range(num_actions),
                        tf.minimum(
                            self.num_rows - 1,
                            tf.maximum(0, row + row_movement)
                        ),
                        tf.minimum(
                            self.num_columns - 1,
                            tf.maximum(0, column + column_movement)
                        )
                    ]
                indices.append(tf.stack(successors, axis=1))
        intuitive_shape = (
            self.num_rows,
            self.num_columns,
            num_actions,
            self.num_rows,
            self.num_columns
        )
        indices = tf.reshape(
            tf.stack(indices, axis=0),
            [len(indices) * num_actions, len(intuitive_shape)]
        )
        return tf.scatter_nd(
            indices,
            tf.ones([indices.shape[0]]),
            shape=intuitive_shape
        )
