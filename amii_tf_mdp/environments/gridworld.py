import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Gridworld(object):
    @staticmethod
    def cardinal_direction_names():
        return ['North', 'East', 'South', 'West']

    @staticmethod
    def cardinal_direction_transformations():
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @staticmethod
    def num_cardinal_directions(): return 4

    class Painter(object):
        MIN_ALPHA = 0.2

        @classmethod
        def weight_to_alpha(cls, weight):
            return (weight + cls.MIN_ALPHA) / (1 - cls.MIN_ALPHA)


        class Tiles(object):
            '''
            Based on the color scheme used by
            <AI safety gridworld>(https://github.com/deepmind/ai-safety-gridworlds)
            '''
            AGENT_BLUE = [0.0, 0.70588235, 1.0]
            WALL_GREY = [0.59607843, 0.59607843, 0.59607843]
            GRID_GREY = [0.85882353, 0.85882353, 0.85882353]
            UNCERTAINTY_YELLOW = [1.0, 1.0, 0.10980392]
            GOAL_GREEN = [0.0, 0.82352941, 0.19607843]
            PATH_WHITE = [1.0, 1.0, 1.0]
            SOURCE_ORANGE = [1.0, 130 / 255.0, 0.0]

            @classmethod
            def grid_rgb_img(
                cls,
                num_rows,
                num_columns,
                source=(-2, -2),
                goal=(1, -2)
            ):
                grid = np.ones([num_rows + 2, num_columns + 2, 3])
                grid[:, 0] = cls.WALL_GREY
                grid[:, -1] = cls.WALL_GREY
                grid[0, :] = cls.WALL_GREY
                grid[-1, :] = cls.WALL_GREY
                grid[source] = cls.SOURCE_ORANGE
                grid[goal] = cls.GOAL_GREEN
                grid[2, 2:-1] = cls.UNCERTAINTY_YELLOW
                return grid

        class PolicyWeights(object):
            NORTH_OFFSET = np.array(
                [(-0.5, -0.5), (0, -0.5 - 0.5/3), (0.5, -0.5)]
            )
            EAST_OFFSET = np.array([(0.5, -0.5), (0.5 + 0.5/3, 0), (0.5, 0.5)])
            SOUTH_OFFSET = np.array(
                [(-0.5, 0.5), (0, 0.5 + 0.5/3), (0.5, 0.5)]
            )
            WEST_OFFSET = np.array(
                [(-0.5, -0.5), (-0.5 - 0.5/3, 0), (-0.5, 0.5)]
            )
            DIRECTION_OFFSETS = [
                NORTH_OFFSET, EAST_OFFSET, SOUTH_OFFSET, WEST_OFFSET
            ]

            @classmethod
            def policy_patches(
                cls,
                row_column_policy,
                color='black',
                sink=None
            ):
                patches = []
                for (row, column, direction), weight in np.ndenumerate(
                    row_column_policy
                ):
                    if (
                        (
                            sink is None or
                            sink[0] != row or
                            sink[1] != column
                        ) and
                        weight > 0.0
                    ):
                        pos = np.array((column, row)) + 1
                        patches.append(
                            mpl.patches.Polygon(
                                pos +
                                cls.DIRECTION_OFFSETS[direction],
                                closed=True,
                                alpha=Gridworld.Painter.weight_to_alpha(
                                    weight),
                                facecolor=color
                            )
                        )
                return patches

        @staticmethod
        def add_top_left_corner_label(
            row,
            column,
            label,
            plt=plt,
            fontsize=7
        ):
            return plt.text(
                column - 0.4,
                row - 0.4,
                label,
                ha='left',
                va='top',
                fontsize=fontsize
            )

        @staticmethod
        def add_center_label(row, column, label, plt=plt, fontsize=5):
            return plt.text(
                column,
                row,
                label,
                ha='center',
                va='center',
                fontsize=fontsize
            )

        @classmethod
        def distribution_overlay_patches(
            cls,
            discounted_row_column_distribution
        ):
            patches = []
            for (row, column), weight in np.ndenumerate(
                discounted_row_column_distribution
            ):
                if weight > 0:
                    pos = np.array((column, row)) + 1
                    patches.append(
                        mpl.patches.Rectangle(
                            pos - np.array((1.0 / 3, 1.0 / 3)),
                            width=2.0 / 3,
                            height=2.0 / 3,
                            fill=True,
                            facecolor=cls.Tiles.AGENT_BLUE,
                            alpha=cls.weight_to_alpha(weight)
                        )
                    )
            return patches

        @staticmethod
        def draw(
            row_column_policy,
            discounted_row_column_distribution=None,
            plt=plt
        ):
            grid_img = Gridworld.Painter.Tiles.grid_rgb_img(
                row_column_policy.shape[0],
                row_column_policy.shape[1]
            )

            plt.rc('grid', linestyle='-', color=Gridworld.Painter.Tiles.GRID_GREY)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.grid(True)

            # Remove ticks and tick labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False

            ax.set_xticks(np.arange(0.5, num_columns + 1.5))
            ax.set_yticks(np.arange(0.5, num_rows + 1.5))

            plt.imshow(grid_img)

            for patch in Gridworld.Painter.distribution_overlay_patches(
                discounted_row_column_distribution
            ):
                ax.add_patch(patch)

            for patch in Gridworld.Painter.PolicyWeights.policy_patches(
                row_column_policy,
                sink=(0, row_column_policy.shape[1] - 1)
            ):
                ax.add_patch(patch)

            Gridworld.Painter.add_top_left_corner_label(
                row_column_policy.shape[0],
                row_column_policy.shape[1],
                r'$S$'
            )
            Gridworld.Painter.add_top_left_corner_label(
                1, row_column_policy.shape[1], r'$G$')
            Gridworld.Painter.add_center_label(
                1, row_column_policy.shape[1], '0.1')

            for c in range(1, row_column_policy.shape[1]):
                Gridworld.Painter.add_center_label(
                    2, c + 1, r"$\sigma={}$".format(0.1))
            return fig, ax

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
        state_rewards_op = tf.reshape(
            tf.scatter_nd(
                row_column_indices,
                destination_rewards,
                shape=(self.num_rows, self.num_columns)
            ),
            (self.num_rows * self.num_columns, 1)
        )
        state_action_rewards_op = tf.matmul(
            state_actions_to_state_model_op, state_rewards_op)
        if sink is not None:
            temp = tf.reshape(
                state_action_rewards_op,
                [
                    self.num_rows,
                    self.num_columns,
                    self.num_cardinal_directions()
                ]
            )
            state_action_rewards_op = tf.reshape(
                temp - tf.scatter_nd(
                    [
                        (sink[0], sink[1], a)
                        for a in range(self.num_cardinal_directions())
                    ],
                    temp[sink[0], sink[1], :],
                    shape=temp.shape
                ),
                [
                    self.num_rows *
                    self.num_columns *
                    self.num_cardinal_directions(),
                    1
                ]
            )
        return state_action_rewards_op

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


if __name__ == '__main__':
    num_rows = 3
    num_columns = 10

    k_200_policy = np.array([[ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 0.25      ,  0.25      ,  0.25      ,  0.25      ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.70766139,  0.        ,  0.        ,  0.29233864],
       [ 0.41930115,  0.        ,  0.        ,  0.58069885],
       [ 0.30641049,  0.        ,  0.        ,  0.69358951],
       [ 0.25235799,  0.        ,  0.        ,  0.74764198],
       [ 0.26399311,  0.        ,  0.        ,  0.73600686],
       [ 0.21423712,  0.        ,  0.        ,  0.78576285],
       [ 0.15432464,  0.        ,  0.        ,  0.84567541],
       [ 0.15158875,  0.        ,  0.        ,  0.8484112 ]],)
    row_column_policy = k_200_policy.reshape([num_rows, num_columns, 4])

    weights = np.maximum(
        0.0,
        np.random.normal(size=row_column_policy.shape[:2])
    )
    fig, ax = Gridworld.Painter.draw(
        row_column_policy,
        weights / weights.sum()
    )

    plt.show()
