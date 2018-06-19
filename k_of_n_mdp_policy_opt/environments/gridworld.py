import tensorflow as tf
import numpy as np


class Gridworld(object):
    @staticmethod
    def cardinal_direction_names():
        return ['North', 'East', 'South', 'West']

    @staticmethod
    def cardinal_direction_transformations():
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]

    @staticmethod
    def num_cardinal_directions():
        return 4

    class Painter(object):
        MIN_ALPHA = 0.1

        @classmethod
        def weight_to_alpha(cls, weight):
            return (weight + cls.MIN_ALPHA) / (1 + cls.MIN_ALPHA)

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
            def grid_rgb_img(cls,
                             num_rows,
                             num_columns,
                             source=(-2, -2),
                             goal=(1, -2)):
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
                [(-0.5, -0.5), (0, -0.5 - 0.5 / 3), (0.5, -0.5)]
            )  # yapf:disable
            EAST_OFFSET = np.array(
                [(0.5, -0.5), (0.5 + 0.5 / 3, 0), (0.5, 0.5)])  # yapf:disable
            SOUTH_OFFSET = np.array(
                [(-0.5, 0.5), (0, 0.5 + 0.5 / 3), (0.5, 0.5)]
            )  # yapf:disable
            WEST_OFFSET = np.array(
                [(-0.5, -0.5), (-0.5 - 0.5 / 3, 0), (-0.5, 0.5)]
            )  # yapf:disable
            DIRECTION_OFFSETS = [
                NORTH_OFFSET, EAST_OFFSET, SOUTH_OFFSET, WEST_OFFSET
            ]  # yapf:disable

            @classmethod
            def policy_patches(cls,
                               row_column_policy,
                               mpl,
                               color='black',
                               sink=None,
                               threshold=1e-15):
                patches = []
                for (row, column, direction), weight in np.ndenumerate(
                    row_column_policy
                ):  # yapf:disable
                    if ((sink is None or sink[0] != row or sink[1] != column)
                            and weight > threshold):
                        pos = np.array((column, row)) + 1
                        patches.append(
                            mpl.patches.Polygon(
                                pos + cls.DIRECTION_OFFSETS[direction],
                                closed=True,
                                alpha=Gridworld.Painter.weight_to_alpha(
                                    weight),
                                facecolor=color))
                return patches

        @staticmethod
        def add_top_left_corner_label(row, column, label, plt, fontsize=7):
            return plt.text(
                column - 0.4,
                row - 0.4,
                label,
                ha='left',
                va='top',
                fontsize=fontsize)

        @staticmethod
        def add_center_label(row, column, label, plt, fontsize=5):
            return plt.text(
                column,
                row,
                label,
                ha='center',
                va='center',
                fontsize=fontsize)

        @classmethod
        def distribution_overlay_patches(cls,
                                         discounted_row_column_distribution,
                                         mpl,
                                         threshold=1e-15):
            patches = []
            for (row, column), weight in np.ndenumerate(
                    discounted_row_column_distribution):
                if weight > threshold:
                    pos = np.array((column, row)) + 1
                    patches.append(
                        mpl.patches.Rectangle(
                            pos - np.array((1.0 / 3, 1.0 / 3)),
                            width=2.0 / 3,
                            height=2.0 / 3,
                            fill=True,
                            facecolor=cls.Tiles.AGENT_BLUE,
                            alpha=cls.weight_to_alpha(weight)))
            return patches

        @staticmethod
        def draw(row_column_policy,
                 mpl,
                 plt,
                 discounted_row_column_distribution=None,
                 threshold=1e-15,
                 source=None,
                 goal=None,
                 uncertainty=None,
                 ax=None):
            num_rows = row_column_policy.shape[0]
            num_columns = row_column_policy.shape[1]

            grid_img = Gridworld.Painter.Tiles.grid_rgb_img(
                num_rows, num_columns)

            plt.rc(
                'grid', linestyle='-', color=Gridworld.Painter.Tiles.GRID_GREY)
            if ax is None:
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
                    discounted_row_column_distribution, threshold=threshold):
                ax.add_patch(patch)

            for patch in Gridworld.Painter.PolicyWeights.policy_patches(
                    row_column_policy,
                    mpl,
                    sink=(0, row_column_policy.shape[1] - 1),
                    threshold=threshold):
                ax.add_patch(patch)

            if source is not None:
                Gridworld.Painter.add_top_left_corner_label(
                    source[0] + 1, source[1] + 1, r'$S$')

            if goal is not None:
                Gridworld.Painter.add_top_left_corner_label(
                    goal[0] + 1, goal[1] + 1, r'$G$')
                Gridworld.Painter.add_center_label(goal[0] + 1, goal[1] + 1,
                                                   '{}'.format(goal[2]))

            if uncertainty is not None:
                for r, c, label in uncertainty:
                    Gridworld.Painter.add_center_label(r + 1, c + 1, label)
            return ax

    def __init__(self, num_rows, num_columns):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self._cardinal_transition_model_ops = {}

    def indicator_state_op(self, *state):
        return tf.reshape(
            tf.scatter_nd(
                [state], [1], shape=(self.num_rows, self.num_columns)),
            [self.num_rows * self.num_columns, 1])

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
                [1, 0, 2])  # yapf:disable
        return self._cardinal_transition_model_ops[sink]

    def cardinal_reward_model_op(self,
                                 row_column_indices,
                                 destination_rewards,
                                 sink=None):
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
        )  # yapf:disable
        state_rewards_op = tf.reshape(
            tf.scatter_nd(
                row_column_indices,
                destination_rewards,
                shape=(self.num_rows, self.num_columns)
            ),
            (self.num_rows * self.num_columns, 1)
        )  # yapf:disable
        state_action_rewards_op = tf.matmul(state_actions_to_state_model_op,
                                            state_rewards_op)
        if sink is not None:
            temp = tf.reshape(
                state_action_rewards_op,
                [
                    self.num_rows,
                    self.num_columns,
                    self.num_cardinal_directions()
                ]
            )  # yapf:disable
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
            )  # yapf:disable
        return state_action_rewards_op

    def _cardinal_grid_transition_model_op(self, sink=None):
        indices = []
        movement = tf.constant(
            self.__class__.cardinal_direction_transformations())
        row_movement = movement[:, 0]
        column_movement = movement[:, 1]
        num_actions = row_movement.shape[0].value
        for row in range(self.num_rows):
            for column in range(self.num_columns):
                if (sink is not None and sink[0] == row and sink[1] == column):
                    successors = [
                        tf.constant(row, shape=[num_actions]),
                        tf.constant(column, shape=[num_actions]),
                        tf.range(num_actions),
                        tf.constant(sink[0], shape=[num_actions]),
                        tf.constant(sink[1], shape=[num_actions])
                    ]  # yapf:disable
                else:
                    successors = [
                        tf.constant(row, shape=[num_actions]),
                        tf.constant(column, shape=[num_actions]),
                        tf.range(num_actions),
                        tf.minimum(self.num_rows - 1,
                                   tf.maximum(0, row + row_movement)),
                        tf.minimum(self.num_columns - 1,
                                   tf.maximum(0, column + column_movement))
                    ]
                indices.append(tf.stack(successors, axis=1))
        intuitive_shape = (self.num_rows, self.num_columns, num_actions,
                           self.num_rows, self.num_columns)
        indices = tf.reshape(
            tf.stack(indices, axis=0),
            [len(indices) * num_actions, len(intuitive_shape)])  # yapf:disable
        return tf.scatter_nd(
            indices, tf.ones([indices.shape[0]]), shape=intuitive_shape)


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    num_rows = 3
    num_columns = 10

    k_200_policy = np.array(
        [[0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.        ,  1.        ,  0.        ,  0.        ],
         [0.25      ,  0.25      ,  0.25      ,  0.25      ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [1.        ,  0.        ,  0.        ,  0.        ],
         [0.74676859,  0.        ,  0.        ,  0.25323141],
         [0.53854144,  0.        ,  0.        ,  0.46145853],
         [0.3968643 ,  0.        ,  0.        ,  0.6031357 ],
         [0.35593784,  0.        ,  0.        ,  0.64406216],
         [0.32623151,  0.        ,  0.        ,  0.67376852],
         [0.23172346,  0.        ,  0.        ,  0.76827651],
         [0.24368413,  0.        ,  0.        ,  0.75631583],
         [0.15257818,  0.        ,  0.        ,  0.84742177]])  # yapf:disable
    row_column_policy = k_200_policy.reshape([num_rows, num_columns, 4])

    uncertainty = []
    for c in range(1, row_column_policy.shape[1]):
        uncertainty.append((1, c, r"$\sigma={}$".format(0.1)))

    fig = plt.figure()
    ax = fig.add_subplot(121)
    Gridworld.Painter.draw(
        row_column_policy,
        mpl,
        plt,
        discounted_row_column_distribution=np.array([[
            1.40720793e-17, 5.25108597e-04, 2.19317898e-03, 4.96153440e-03,
            8.52138456e-03, 1.39449192e-02, 2.20358949e-02, 2.95764264e-02,
            4.16728668e-02, 4.98644024e-01
        ], [
            1.40720793e-17, 5.83454152e-04, 1.91175693e-03, 3.31963645e-03,
            4.50667180e-03, 6.97296951e-03, 1.05394088e-02, 1.08268056e-02,
            1.67267621e-02, 1.37320356e-02
        ], [
            1.40720793e-17, 6.48282294e-04, 2.84448825e-03, 6.84902770e-03,
            1.26174437e-02, 2.17671245e-02, 3.58961523e-02, 5.19143939e-02,
            7.62679577e-02, 1.00000001e-01
        ]]),
        threshold=1e-10,
        source=(row_column_policy.shape[0] - 1,
                row_column_policy.shape[1] - 1),
        goal=(0, row_column_policy.shape[1] - 1, 0.1),
        uncertainty=uncertainty,
        ax=ax)
    ax = fig.add_subplot(122)
    Gridworld.Painter.draw(
        row_column_policy,
        mpl,
        plt,
        discounted_row_column_distribution=np.array([[
            1.40720793e-17, 5.25108597e-04, 2.19317898e-03, 4.96153440e-03,
            8.52138456e-03, 1.39449192e-02, 2.20358949e-02, 2.95764264e-02,
            4.16728668e-02, 4.98644024e-01
        ], [
            1.40720793e-17, 5.83454152e-04, 1.91175693e-03, 3.31963645e-03,
            4.50667180e-03, 6.97296951e-03, 1.05394088e-02, 1.08268056e-02,
            1.67267621e-02, 1.37320356e-02
        ], [
            1.40720793e-17, 6.48282294e-04, 2.84448825e-03, 6.84902770e-03,
            1.26174437e-02, 2.17671245e-02, 3.58961523e-02, 5.19143939e-02,
            7.62679577e-02, 1.00000001e-01
        ]]),
        threshold=1e-10,
        source=(row_column_policy.shape[0] - 1,
                row_column_policy.shape[1] - 1),
        goal=(0, row_column_policy.shape[1] - 1, 0.1),
        uncertainty=uncertainty,
        ax=ax)

    plt.show()
