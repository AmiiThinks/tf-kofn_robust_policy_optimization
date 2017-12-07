import tensorflow as tf
import numpy as np


class Gridworld(object):
    @staticmethod
    def cardinal_direction_names():
        return ['North', 'East', 'South', 'West']

    @staticmethod
    def cardinal_direction_transformations():
        return tf.constant([(-1, 0), (0, 1), (1, 0), (0, -1)])

    @staticmethod
    def cardinal_direction_transformations_np():
        return np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])

    def __init__(self, num_rows, num_columns):
        self.num_rows = num_rows
        self.num_columns = num_columns

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
        indices = []
        movement = self.__class__.cardinal_direction_transformations()
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
        return tf.transpose(
            tf.reshape(
                tf.transpose(
                    tf.scatter_nd(
                        indices,
                        tf.ones([indices.shape[0]]),
                        shape=intuitive_shape
                    ),
                    [2, 0, 1, 3, 4]
                ),
                [
                    num_actions,
                    self.num_rows * self.num_columns,
                    self.num_rows * self.num_columns
                ]
            ),
            [1, 0, 2]
        )

    # def cardinal_reward_model_op(
    #     self,
    #     goal,
    #     unknown_reward_positions,
    #     unknown_reward_means
    # ):
    #     movement = self.__class__.cardinal_direction_transformations()
    #     row_movement = movement[:, 0]
    #     column_movement = movement[:, 1]
    #     num_actions = movement.shape[0].value
    #     unknown_reward_positions_dict = {p for p in unknown_reward_positions}
    #     indices = []
    #     for row in range(self.num_rows):
    #         for column in range(self.num_columns):
    #             if not(goal[0] == row and goal[1] == column):
    #                 # TODO
    #                 np.concatenate(
    #                     (
    #                         np.minimum(
    #                             self.num_rows - 1,
    #                             np.maximum(0, row + row_movement)
    #                         ),
    #                         np.minimum(
    #                             self.num_columns - 1,
    #                             np.maximum(0, column + column_movement)
    #                         )
    #                     ),
    #                     axis=1
    #                 )
    #             indices.append(tf.stack(successors, axis=1))
