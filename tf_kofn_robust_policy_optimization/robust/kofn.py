import tensorflow as tf
from tf_contextual_prediction_with_expert_advice import utility
from tf_kofn_robust_policy_optimization.robust import \
    prob_ith_element_is_sampled


def kofn_ev(evs, weights):
    return tf.tensordot(evs, weights, 1)


def kofn_regret_update(chance_prob_sequence_list, reward_models, weights,
                       learner):
    weighted_rewards = [
        chance_prob_sequence_list[i] * reward_models[i] * weights[i]
        for i in range(len(reward_models))
    ]
    inst_regrets = [learner.instantaneous_regrets(r) for r in weighted_rewards]
    regret_update = learner.updated_regrets(
        sum(inst_regrets[1:], inst_regrets[0]))
    return regret_update


class DeterministicKofnGameTemplate(object):
    def __init__(self, k, n):
        self.n_weights = [0.0] * n
        self.n_weights[n - 1] = 1.0

        self.k = k
        self.k_weights = [0.0] * n
        self.k_weights[k - 1] = 1.0

        self.prob_ith_element_is_sampled = prob_ith_element_is_sampled(
            self.n_weights, self.k_weights)

    def to_yml(self, indent=0):
        ws = ' ' * indent
        return "\n".join([
            '{}{}: {}'.format(ws, key, value)
            for key, value in [('n', self.num_sampled_worlds()), ('k', self.k)]
        ])

    def num_sampled_worlds(self):
        return len(self.n_weights)

    def label(self):
        return '{}-of-{}'.format(self.k, self.num_sampled_worlds())

    def __str__(self):
        return self.label() + ' template'


class UncertainRewardKofnTrainer(object):
    def __init__(self, template, reward_generator, *learners):
        self.template = template
        self._t = tf.train.get_or_create_global_step()
        self._t.assign(0)
        self.reward_generator = reward_generator
        self.learners = learners
        self.prob_ith_element_is_sampled = tf.squeeze(
            self.template.prob_ith_element_is_sampled)

    @property
    def t(self):
        return int(self._t.numpy())

    def game_evs(self, inputs):
        rewards = self.reward_generator(inputs)
        return tf.stack([
            self._eval_game(rewards, learner(inputs)).root_ev
            for learner in self.learners
        ])

    def step(self, inputs):
        rewards = self.reward_generator(inputs)
        losses = []
        evs = []
        for learner in self.learners:
            with tf.GradientTape() as tape:
                policy = learner(inputs)
                action_utilities = self._eval_game(rewards,
                                                   policy).kofn_utility
                loss = learner.loss(
                    tf.stop_gradient(action_utilities),
                    inputs=inputs,
                    policy=policy)
            losses.append(loss)
            evs.append(tf.reduce_mean(utility(policy, action_utilities)))
            learner.apply_gradients(loss, tape)
        self._t.assign_add(1)
        return losses, evs

    def evaluate(self, inputs, test_rewards=None):
        evs = self.game_evs(inputs)
        test_evs = []

        if test_rewards is None:
            return evs
        else:
            test_evs = tf.stack([
                tf.reduce_mean(utility(learner.policy(inputs), test_rewards))
                for learner in self.learners
            ])
            return evs, test_evs

    def _eval_game(self, rewards, policy):
        raise NotImplementedError('Override')
