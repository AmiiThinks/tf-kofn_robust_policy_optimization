import tensorflow as tf
import amii_tf_mdp.reward_utils as patient


class RewardUtilsTest(tf.test.TestCase):
    def test_reward_distribution(self):
        with self.test_session():
            rewards = tf.constant(
                [
                    [
                        [0.2, 0.8],  # R(s' | s = 1, a = 1)
                        [0.3, 0.7],  # R(s' | s = 1, a = 2)
                        [0.4, 0.6]   # R(s' | s = 1, a = 3)
                    ],
                    [
                        [0.1, 0.9],  # R(s' | s = 2, a = 1)
                        [0.5, 0.5],  # R(s' | s = 2, a = 2)
                        [0.6, 0.4]   # R(s' | s = 2, a = 3)
                    ]
                ]
            )
            state_dist = tf.constant([0.5, 0.5])
            strat = tf.constant(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5]
                ]
            )
            self.assertAllClose(
                [0.375, 0.625],
                patient.reward_distribution(
                    rewards,
                    state_dist,
                    strat
                ).eval()
            )


if __name__ == '__main__':
    tf.test.main()
