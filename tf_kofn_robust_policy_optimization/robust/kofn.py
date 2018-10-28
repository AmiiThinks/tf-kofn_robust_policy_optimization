import tensorflow as tf
from tf_contextual_prediction_with_expert_advice import utility
from tf_kofn_robust_policy_optimization import cache
from tf_kofn_robust_policy_optimization.robust import \
    prob_ith_element_is_sampled, \
    rank_to_element_weights


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


def ensure_batch_context_world_shape(tensor):
    tensor = tf.convert_to_tensor(tensor)
    if len(tensor.shape) == 1:
        tensor = tf.reshape(tensor, [1, tensor.shape[0].value, 1])
    elif len(tensor.shape) == 2:
        tensor = tf.expand_dims(tensor, axis=0)
    return tensor


class KofnEvsAndWeights(object):
    def __init__(self, context_values, opponent, context_weights=None):
        '''
        Parameters:
        - context_values: (m X) |S| X n matrix, where m is an optional
            batch number.
        - opponent: n vector of weights representing the opponent's weight
            on each world after they are ranked.
        - context_weights: |S| vector of weights for each context.
        '''
        self.context_values_given_world = ensure_batch_context_world_shape(
            context_values)
        self.opponent = opponent
        self.context_weights = context_weights

        if context_weights is None:
            self.ev_given_world = tf.reduce_mean(
                self.context_values_given_world, axis=1)
        else:
            self.context_weights = ensure_batch_context_world_shape(
                context_weights)
            self.ev_given_world = tf.reduce_sum(
                self.context_values_given_world * self.context_weights, axis=1)
        '''m X n weighting of the unranked worlds.'''
        self.world_weights = rank_to_element_weights(opponent,
                                                     self.ev_given_world)

    @property
    def batch_size(self):
        return self.context_values_given_world.shape[0].value

    @property
    def num_contexts(self):
        return self.context_values_given_world.shape[1].value

    @cache
    def context_values(self):
        return tf.reduce_sum(
            self.context_values_given_world * tf.expand_dims(
                self.world_weights, axis=1),
            axis=-1)

    @cache
    def ev(self):
        return tf.reduce_sum(self.ev_given_world * self.world_weights, axis=-1)


def kofn_action_values(action_values_given_world, world_weights):
    action_values_given_world = tf.convert_to_tensor(action_values_given_world)
    world_weights = tf.convert_to_tensor(world_weights)

    ww_has_batch_dim = len(world_weights.shape) > 1
    ww_batch_size = world_weights.shape[0].value

    avgw_has_batch_dim = len(action_values_given_world.shape) > 3
    avgw_batch_size = action_values_given_world.shape[0].value

    has_batch_dim = ww_has_batch_dim or avgw_has_batch_dim
    if has_batch_dim:
        if not ww_has_batch_dim:
            world_weights = tf.expand_dims(world_weights, 0)
            if avgw_has_batch_dim:
                world_weights = tf.tile(world_weights, [avgw_batch_size, 1])
        if not avgw_has_batch_dim:
            action_values_given_world = tf.expand_dims(
                action_values_given_world, 0)
            if ww_has_batch_dim:
                action_values_given_world = tf.tile(action_values_given_world,
                                                    [ww_batch_size, 1, 1, 1])
        if ww_has_batch_dim and avgw_has_batch_dim:
            world_weights = tf.tile(
                world_weights, [max(1, avgw_batch_size // ww_batch_size), 1])
            action_values_given_world = tf.tile(
                action_values_given_world,
                [max(1, ww_batch_size // avgw_batch_size), 1, 1, 1])

        return tf.einsum('bsaw,bw->bsa', action_values_given_world,
                         world_weights)
    else:
        return tf.tensordot(
            action_values_given_world, world_weights, axes=[-1, -1])


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
