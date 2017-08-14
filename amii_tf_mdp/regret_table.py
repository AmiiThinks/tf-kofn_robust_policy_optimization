import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex


class RegretTable(object):
    '''
    Regret table. Rank-3 Tensor
    (|I| by |A|, where I is the player's information partition).
    '''

    @staticmethod
    def num_sequences(horizon, num_states, num_actions):
        return int(
            num_states * (
                (num_states * num_actions)**(horizon + 1) - 1
            ) / (
                num_states * num_actions - 1
            )
        )

    @staticmethod
    def new_ir_table(horizon, num_states, num_actions):
        return tf.Variable(
            tf.constant(0.0, shape=(horizon * num_states, num_actions))
        )

    @staticmethod
    def new_pr_table(horizon, num_states, num_actions):
        return tf.Variable(
            tf.constant(
                0.0,
                shape=(
                    RegretTable.num_sequences(
                        horizon,
                        num_states,
                        num_actions
                    ),
                    num_actions
                )
            )
        )

    @staticmethod
    def regrets_at_timestep_ir(regrets, t, num_states):
        return regrets[t * num_states:(t + 1) * num_states, :]

    @staticmethod
    def regrets_at_timestep_pr(regrets, t, num_states, num_actions):
        n = RegretTable.num_sequences(t, num_states, num_actions)
        next_n = RegretTable.num_sequences(t + 1, num_states, num_actions)
        return regrets[n:next_n, :]

    @staticmethod
    def updated_regrets_at_timestep_ir(
        regrets,
        t,
        num_states,
        inst_regrets,
        **kwargs
    ):
        return tf.assign(
            RegretTable.regrets_at_timestep_ir(regrets, t, num_states),
            inst_regrets,
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
            RegretTable.regrets_at_timestep_pr(
                regrets,
                t,
                num_states,
                num_actions
            ),
            inst_regrets,
            **kwargs
        )

    def __init__(self, mdp):
        self.mdp = mdp
        self.regrets = self.__class__.new_table(
            mdp.horizon,
            mdp.num_states(),
            mdp.num_actions()
        )

    def strat_at_timestep(self, t):
        return tf.transpose(
            l1_projection_to_simplex(tf.transpose(self.at_timestep(t)))
        )


class RegretTableIr(RegretTable):
    @classmethod
    def new_table(cls, horizon, num_states, num_actions):
        return cls.new_ir_table(horizon, num_states, num_actions)

    def at_timestep(self, t):
        '''
        Returns:
        - Rank-2 Tensor (|S| by |A|) that are the regrets at
          timestep t for every state.
        '''
        return self.__class__.regrets_at_timestep_ir(
            self.regrets,
            t,
            self.mdp.num_states()
        )

    def updated_at_timestep(self, t, inst_regrets, **kwargs):
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
    def new_table(cls, horizon, num_states, num_actions):
        return cls.new_pr_table(horizon, num_states, num_actions)

    def at_timestep(self, t):
        return self.__class__.regrets_at_timestep_pr(
            self.regrets,
            t,
            self.mdp.num_states(),
            self.mdp.num_actions()
        )

    def updated_at_timestep(self, t, inst_regrets, **kwargs):
        return self.__class__.updated_regrets_at_timestep_pr(
            self.regrets,
            t,
            self.mdp.num_states(),
            self.mdp.num_actions(),
            inst_regrets,
            **kwargs
        )


class RegretMatchingPlusMixin(object):
    def updated_at_timestep(self, t, inst_regrets, **kwargs):
        node = super(RegretMatchingPlusMixin, self).updated_at_timestep(
            t,
            inst_regrets,
            **kwargs
        )
        return tf.assign(
            node,
            tf.maximum(0.0, node),
            **kwargs
        )


class RegretMatchingPlusIr(RegretMatchingPlusMixin, RegretTableIr): pass


class RegretMatchingPlusPr(RegretMatchingPlusMixin, RegretTablePr): pass
