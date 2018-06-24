import tensorflow as tf
from itertools import product
from .utils.tensor import l1_projection_to_simplex
from .utils.sequence import num_pr_sequences, num_ir_sequences


class RegretTable(object):
    '''
    Regret table. Rank-3 Tensor
    (|I| by |A|, where I is the player's information partition).
    '''

    @staticmethod
    def new_ir_table(horizon, num_states, num_actions, **kwargs):
        init = tf.constant(
            0.0,
            # TODO Maybe this should be horizon - 1, not sure.
            shape=(num_ir_sequences(horizon, num_states), num_actions))
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = 'ir_table'
        return tf.get_variable(
            name, initializer=init, trainable=False, **kwargs)

    @staticmethod
    def new_pr_table(horizon, num_states, num_actions, **kwargs):
        init = tf.constant(
            0.0,
            shape=(
                num_pr_sequences(horizon - 1, num_states, num_actions),
                num_actions
            )
        )  # yapf:disable
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = 'ir_table'
        return tf.get_variable(
            name, initializer=init, trainable=False, **kwargs)

    @staticmethod
    def indices_at_timestep_ir(sequences, t, num_states):
        return list(
            product(
                range(t * num_states, (t + 1) * num_states),
                range(sequences.shape[1].value)))

    @staticmethod
    def indices_at_timestep_pr(sequences, t, num_states, num_actions):
        n = num_pr_sequences(t - 1, num_states, num_actions)
        next_n = num_pr_sequences(t, num_states, num_actions)
        return list(product(range(n, next_n), range(sequences.shape[1].value)))

    @staticmethod
    def sequences_at_timestep_ir(sequences, t, num_states):
        return sequences[t * num_states:(t + 1) * num_states, :]

    @staticmethod
    def sequences_at_timestep_pr(sequences, t, num_states, num_actions):
        n = num_pr_sequences(t - 1, num_states, num_actions)
        next_n = num_pr_sequences(t, num_states, num_actions)
        return sequences[n:next_n, :]

    @staticmethod
    def update_regrets_at_timestep_ir(regrets, t, num_states, inst_regrets,
                                      **kwargs):
        indices = RegretTable.indices_at_timestep_ir(regrets, t, num_states)
        flat_updates = tf.reshape(
            (RegretTable.sequences_at_timestep_ir(regrets, t, num_states) +
             inst_regrets),
            shape=[tf.size(inst_regrets)])

        return tf.scatter_nd_update(regrets, indices, flat_updates, **kwargs)

    @staticmethod
    def update_regrets_at_timestep_pr(regrets, t, num_states, num_actions,
                                      inst_regrets, **kwargs):
        indices = RegretTable.indices_at_timestep_pr(regrets, t, num_states,
                                                     num_actions)
        flat_updates = tf.reshape(
            (RegretTable.sequences_at_timestep_pr(regrets, t, num_states,
                                                  num_actions) + inst_regrets),
            shape=[tf.size(inst_regrets)])

        return tf.scatter_nd_update(regrets, indices, flat_updates, **kwargs)

    def __init__(self, horizon, num_states, num_actions, name=None):
        self._horizon = horizon
        self._num_states = num_states
        self.regrets = self.__class__.new_table(
            horizon,
            num_states,
            num_actions,
            name=type(self).__name__ if name is None else name)

    def strat(self):
        return l1_projection_to_simplex(
            self.regrets, axis=tf.rank(self.regrets) - 1)

    def horizon(self):
        return self._horizon

    def num_states(self):
        return self._num_states

    def num_actions(self):
        return self.regrets.shape[-1].value

    def update_regrets(self, inst_regrets, **kwargs):
        '''
        Params:
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - Update node.
        '''
        return self.regrets.assign_add(inst_regrets, **kwargs)


class PrRegretTable(RegretTable):
    @classmethod
    def new_table(cls, horizon, num_states, num_actions, **kwargs):
        return cls.new_pr_table(horizon, num_states, num_actions, **kwargs)

    def num_pr_sequences(self, t):
        return num_pr_sequences(t, self.num_states(), self.num_actions())

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())

    def _strat_at_timestep(self, t):
        return self.__class__.sequences_at_timestep_pr(self.strat(), t,
                                                       self.num_states(),
                                                       self.num_actions())

    def _update_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        return self.__class__.update_regrets_at_timestep_pr(
            self.regrets, t, self.num_states(), self.num_actions(),
            inst_regrets, **kwargs)

    def instantaneous_regrets(self, weighted_sequence_rewards):
        inst_regret_blocks = []
        current_cf_state_values = None
        strat = tf.reshape(
            self.strat(), [
                self.num_pr_prefixes(self.horizon() - 1),
                self.num_states(),
                self.num_actions()
            ]
        )  # yapf:disable
        action_rewards_weighted_by_chance = tf.squeeze(
            tf.reduce_sum(weighted_sequence_rewards, axis=3))

        for t in range(self.horizon() - 1, -1, -1):
            n = self.num_pr_prefixes(t - 1)
            next_n = self.num_pr_prefixes(t)
            current_rewards = action_rewards_weighted_by_chance[n:next_n, :, :]
            if current_cf_state_values is None:
                current_cf_action_values = current_rewards
            else:
                current_cf_action_values = (
                    current_rewards
                    + tf.reshape(
                        tf.reduce_sum(current_cf_state_values, axis=1),
                        current_rewards.shape)
                )  # yapf:disable

            # TODO Should be able to do this with tensordot
            # but it didn't work the first way I tried.
            current_cf_state_values = tf.reduce_sum(
                (strat[n:next_n, :, :] * current_cf_action_values),
                axis=2,
                keepdims=True)

            inst_regret_blocks.append(
                current_cf_action_values - current_cf_state_values)
        inst_regret_blocks.reverse()
        return tf.reshape(
            tf.concat(inst_regret_blocks, axis=0), self.regrets.shape)


class RegretMatchingPlusMixin(object):
    def update_regrets(self, inst_regrets, **kwargs):
        '''
        Params:
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - Update node.
        '''
        return self.regrets.assign(
            tf.maximum(0.0, self.regrets + inst_regrets), **kwargs)

    def _update_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        node = super(
            RegretMatchingPlusMixin,
            self
        ).update_regrets_at_timestep(t, inst_regrets, **kwargs)  # yapf:disable
        return node.assign(tf.maximum(0.0, node), **kwargs)


class PrRegretMatchingPlus(RegretMatchingPlusMixin, PrRegretTable):
    pass
