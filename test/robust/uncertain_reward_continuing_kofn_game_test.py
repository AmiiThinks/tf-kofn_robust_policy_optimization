import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tf_kofn_robust_policy_optimization.robust.uncertain_reward_discounted_continuing_kofn import \
    UncertainRewardDiscountedContinuingKofnGame
from tf_contextual_prediction_with_expert_advice import normalized


class UncertainRewardDiscountedContinuingKofnGameTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_init(self):
        num_worlds = 3
        num_states = 4
        num_actions = 2

        mixture_constraint_weights = normalized(
            tf.random.uniform([num_worlds]))

        self.assertAllClose(
            [0.4556628465652466, 0.30238017439842224, 0.24195699393749237],
            mixture_constraint_weights)

        root_probs = normalized(tf.random.uniform([num_states]))
        transition_model = normalized(
            tf.random.uniform([num_states, num_actions, num_states]), axis=2)
        reward_models = tf.random.normal(
            shape=[num_states, num_actions, num_worlds])
        policy = normalized(tf.random.uniform([num_states, num_actions]))
        discount = 0.99

        patient = UncertainRewardDiscountedContinuingKofnGame(
            mixture_constraint_weights,
            root_probs,
            transition_model,
            reward_models,
            policy,
            gamma=discount)

        self.assertAllClose([0.20772548, 0.35927513, -0.28098956], patient.evs)

        self.assertAllClose(
            [0.30238017439842224, 0.24195699393749237, 0.4556628465652466],
            patient.k_weights)

        self.assertAllClose(0.021704704, patient.root_ev)

        self.assertAllClose(
            [
                [0.20841458439826965, -0.8067079186439514],
                [-0.7348495125770569, 0.4023403525352478],
                [0.6527014970779419, -0.31257206201553345],
                [0.2731754183769226, -0.992716372013092]
            ],
            patient.kofn_utility,
            rtol=1e-5,
            atol=1e-5
        )  # yapf:disable


if __name__ == '__main__':
    tf.test.main()
