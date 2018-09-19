import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_kofn_robust_policy_optimization.robust.uncertain_reward_discounted_continuing_kofn import \
    UncertainRewardDiscountedContinuingKofnGame
from tf_kofn_robust_policy_optimization.utils.tensor import normalized


class UncertainRewardDiscountedContinuingKofnGameTest(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(42)

    def test_init(self):
        num_worlds = 3
        num_states = 4
        num_actions = 2

        mixture_constraint_weights = normalized(
            tf.random_uniform([num_worlds]))

        self.assertAllClose(
            [0.4556628465652466, 0.30238017439842224, 0.24195699393749237],
            mixture_constraint_weights)

        root_probs = normalized(tf.random_uniform([num_states]))
        transition_model = normalized(
            tf.random_uniform([num_states, num_actions, num_states]), axis=2)
        reward_models = tf.random_normal(
            shape=[num_states, num_actions, num_worlds])
        policy = normalized(tf.random_uniform([num_states, num_actions]))
        discount = 0.99
        max_num_iterations = 100

        patient = UncertainRewardDiscountedContinuingKofnGame(
            mixture_constraint_weights,
            root_probs,
            transition_model,
            reward_models,
            policy,
            gamma=discount,
            max_num_iterations=max_num_iterations)

        self.assertAllClose(
            [0.05193137004971504, 0.08981878310441971, -0.07024738192558289],
            patient.evs)

        self.assertAllClose(
            [0.30238017439842224, 0.24195699393749237, 0.4556628465652466],
            patient.k_weights)

        self.assertAllClose(0.005426, patient.root_ev)

        self.assertAllClose(
            [
                [0.20841458439826965, -0.8067079186439514],
                [-0.7348495125770569, 0.4023403525352478],
                [0.6527014970779419, -0.31257206201553345],
                [0.2731754183769226, -0.992716372013092]
            ],
            patient.kofn_utility
        )  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
