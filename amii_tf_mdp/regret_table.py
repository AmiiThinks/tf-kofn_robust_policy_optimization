import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from .sequence_utils import num_pr_sequences, num_ir_sequences
from .mdp import PrMdpState


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
                shape=(num_ir_sequences(horizon, num_states), num_actions)
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

    def __init__(self, mdp, name=None):
        self.mdp = mdp
        self.regrets = self.__class__.new_table(
            mdp.horizon,
            mdp.num_states(),
            mdp.num_actions(),
            name=type(self).__name__ if name is None else name
        )

    def strat_at_timestep(self, t):
        return tf.transpose(
            l1_projection_to_simplex(tf.transpose(self.at_timestep(t)))
        )

    def strat(self):
        return tf.transpose(
            l1_projection_to_simplex(tf.transpose(self.regrets))
        )


class RegretTableIr(RegretTable):
    @classmethod
    def new_table(cls, horizon, num_states, num_actions, **kwargs):
        return cls.new_ir_table(horizon, num_states, num_actions, **kwargs)

    def strat_at_timestep(self, t):
        '''
        Returns:
        - Rank-2 Tensor (|S| by |A|) that are the regrets at
          timestep t for every state.
        '''
        return self.__class__.sequences_at_timestep_ir(
            self.strat(),
            t,
            self.mdp.num_states()
        )

    def updated_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        '''
        Params:
        - t: Timestep.
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - Update node.
        '''
        return self.__class__.updated_regrets_at_timestep_ir(
            self.regrets,
            t,
            self.mdp.num_states(),
            inst_regrets,
            **kwargs
        )


class RegretTablePr(RegretTable):
    @classmethod
    def new_table(cls, horizon, num_states, num_actions, **kwargs):
        return cls.new_pr_table(horizon, num_states, num_actions, **kwargs)

    def strat_at_timestep(self, t):
        return self.__class__.sequences_at_timestep_pr(
            self.strat(),
            t,
            self.mdp.num_states(),
            self.mdp.num_actions()
        )

    def updated_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        return self.__class__.updated_regrets_at_timestep_pr(
            self.regrets,
            t,
            self.mdp.num_states(),
            self.mdp.num_actions(),
            inst_regrets,
            **kwargs
        )

    def cfr_update(self, initial_state_distribution=None):
        last_regret_update = None
        pr_mdp_state = PrMdpState(
            self.mdp,
            initial_state_distribution,
            name='PrMdpState_cfr_update'
        )
        pr_mdp_state.sequences.initializer.run()
        current_cf_state_values = None
        strat = tf.reshape(
            self.strat(),
            [-1, self.mdp.num_states(), self.mdp.num_actions()]
        )

        with tf.control_dependencies([pr_mdp_state.unroll()]):
            action_rewards_weighted_by_chance = tf.squeeze(
                tf.reduce_sum(
                    pr_mdp_state.sequences * self.mdp.rewards,
                    axis=3
                )
            )
            for t in range(self.mdp.horizon - 1, -1, -1):
                n = self.mdp.num_pr_prefixes(t - 1)
                next_n = self.mdp.num_pr_prefixes(t)
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
                            [
                                -1,
                                self.mdp.num_states(),
                                self.mdp.num_actions()
                            ]
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
                    current_cf_action_values -
                    current_cf_state_values
                )
                d = [cf_regrets]
                if last_regret_update is not None:
                    d.append(last_regret_update)

                with tf.control_dependencies(d):
                    last_regret_update = (
                        self.updated_regrets_at_timestep(
                            t,
                            tf.reshape(
                                cf_regrets,
                                [-1, self.mdp.num_actions()]
                            )
                        )
                    )
        return (
            last_regret_update,
            tf.reduce_sum(current_cf_state_values)
        )


class RegretMatchingPlusMixin(object):
    def updated_regrets_at_timestep(self, t, inst_regrets, **kwargs):
        node = super(
            RegretMatchingPlusMixin,
            self
        ).updated_regrets_at_timestep(
            t,
            inst_regrets,
            **kwargs
        )
        return tf.assign(node, tf.maximum(0.0, node), **kwargs)


class RegretMatchingPlusIr(RegretMatchingPlusMixin, RegretTableIr): pass


class RegretMatchingPlusPr(RegretMatchingPlusMixin, RegretTablePr): pass
