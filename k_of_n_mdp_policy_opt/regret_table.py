import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from .utils.sequence import num_pr_sequences, num_ir_sequences


class RegretTable(object):
    '''
    Regret table. Rank-3 Tensor
    (|I| by |A|, where I is the player's information partition).
    '''
    @staticmethod
    def new_ir_table(horizon, num_states, num_actions, **kwargs):
        return tf.Variable(
            tf.constant(
                0.0,
                # TODO Maybe this should be horizon - 1, not sure.
                shape=(
                    num_ir_sequences(horizon, num_states),
                    num_actions
                )
            ),
            **kwargs
        )

    @staticmethod
    def new_pr_table(horizon, num_states, num_actions, **kwargs):
        return tf.Variable(
            tf.constant(
                0.0,
                shape=(
                    num_pr_sequences(
                        horizon - 1,
                        num_states,
                        num_actions
                    ),
                    num_actions
                )
            ),
            **kwargs
        )

    @staticmethod
    def sequences_at_timestep_ir(sequences, t, num_states):
        return sequences[t * num_states:(t + 1) * num_states, :]

    @staticmethod
    def sequences_at_timestep_pr(sequences, t, num_states, num_actions):
        n = num_pr_sequences(t - 1, num_states, num_actions)
        next_n = num_pr_sequences(t, num_states, num_actions)
        return sequences[n:next_n, :]

    @staticmethod
    def updated_regrets_at_timestep_ir(
        regrets,
        t,
        num_states,
        inst_regrets,
        **kwargs
    ):
        return tf.assign(
            RegretTable.sequences_at_timestep_ir(
                regrets,
                t,
                num_states
            ),
            RegretTable.sequences_at_timestep_ir(
                regrets,
                t,
                num_states
            ) + inst_regrets,
            **kwargs
        )

    @staticmethod
    def updated_regrets_at_timestep_pr(
        regrets,
        t,
        num_states,
        num_actions,
        inst_regrets,
        **kwargs
    ):
        return tf.assign(
            RegretTable.sequences_at_timestep_pr(
                regrets,
                t,
                num_states,
                num_actions
            ),
            RegretTable.sequences_at_timestep_pr(
                regrets,
                t,
                num_states,
                num_actions
            ) + inst_regrets,
            **kwargs
        )

    def __init__(self, horizon, num_states, num_actions, name=None):
        self._horizon = horizon
        self._num_states = num_states
        self.regrets = self.__class__.new_table(
            horizon,
            num_states,
            num_actions,
            name=type(self).__name__ if name is None else name
        )
        self.strat = tf.transpose(
            l1_projection_to_simplex(tf.transpose(self.regrets))
        )

    def horizon(self): return self._horizon
    def num_states(self): return self._num_states
    def num_actions(self): return self.regrets.shape[-1].value

    def _strat_at_timestep(self, t):
        return tf.transpose(
            l1_projection_to_simplex(
                tf.transpose(self.at_timestep(t))
            )
        )

    def updated_regrets(self, inst_regrets, **kwargs):
        '''
        Params:
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - Update node.
        '''
        return tf.assign_add(self.regrets, inst_regrets, **kwargs)


class PrRegretTable(RegretTable):
    @classmethod
    def new_table(cls, horizon, num_states, num_actions, **kwargs):
        return cls.new_pr_table(
            horizon,
            num_states,
            num_actions,
            **kwargs
        )

    def num_pr_sequences(self, t):
        return num_pr_sequences(
            t,
            self.num_states(),
            self.num_actions()
        )

    def num_pr_prefixes(self, t):
        return int(self.num_pr_sequences(t) / self.num_states())

    def _strat_at_timestep(self, t):
        return self.__class__.sequences_at_timestep_pr(
            self.strat(),
            t,
            self.num_states(),
            self.num_actions()
        )

    def _updated_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        return self.__class__.updated_regrets_at_timestep_pr(
            self.regrets,
            t,
            self.num_states(),
            self.num_actions(),
            inst_regrets,
            **kwargs
        )

    def instantaneous_regrets(self, weighted_sequence_rewards):
        inst_regret_blocks = []
        current_cf_state_values = None
        strat = tf.reshape(
            self.strat,
            [
                self.num_pr_prefixes(self.horizon() - 1),
                self.num_states(),
                self.num_actions()
            ]
        )
        action_rewards_weighted_by_chance = tf.squeeze(
            tf.reduce_sum(
                weighted_sequence_rewards,
                axis=3
            )
        )
        for t in range(self.horizon() - 1, -1, -1):
            n = self.num_pr_prefixes(t - 1)
            next_n = self.num_pr_prefixes(t)
            current_rewards = action_rewards_weighted_by_chance[
                n:next_n,
                :,
                :
            ]
            if current_cf_state_values is None:
                current_cf_action_values = current_rewards
            else:
                current_cf_action_values = (
                    current_rewards +
                    tf.reshape(
                        tf.reduce_sum(
                            current_cf_state_values,
                            axis=1
                        ),
                        current_rewards.shape
                    )
                )

            current_cf_state_values = tf.expand_dims(
                # TODO Should be able to do this with tensordot
                # but it didn't work the first way I tried.
                tf.reduce_sum(
                    (
                        strat[n:next_n, :, :] *
                        current_cf_action_values
                    ),
                    axis=2
                ),
                axis=2
            )
            cf_regrets = (
                current_cf_action_values - current_cf_state_values
            )
            inst_regret_blocks.append(cf_regrets)
        inst_regret_blocks.reverse()
        return tf.reshape(
            tf.concat(inst_regret_blocks, axis=0),
            self.regrets.shape
        )


class RegretMatchingPlusMixin(object):
    def updated_regrets(self, inst_regrets, **kwargs):
        '''
        Params:
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - Update node.
        '''
        return tf.assign_add(
            self.regrets,
            tf.maximum(0.0, inst_regrets),
            **kwargs
        )

    def _updated_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        node = super(
            RegretMatchingPlusMixin,
            self
        ).updated_regrets_at_timestep(
            t,
            inst_regrets,
            **kwargs
        )
        return tf.assign(node, tf.maximum(0.0, node), **kwargs)


class PrRegretMatchingPlus(RegretMatchingPlusMixin, PrRegretTable): pass
