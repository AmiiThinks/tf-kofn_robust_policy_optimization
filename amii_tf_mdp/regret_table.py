import tensorflow as tf


class RegretTable(object):
    ''' Regret table. Rank-3 Tensor (T by |S| by |A|).'''

    @staticmethod
    def project_to_positive_orthant(weights):
        return tf.where(
            tf.greater(weights, 0.0),
            weights,
            tf.constant(0.0, shape=weights.shape)
        )

    @staticmethod
    def new_table(horizon, num_states, num_actions):
        return tf.Variable(
            tf.constant(0.0, shape=(horizon, num_states, num_actions))
        )

    @staticmethod
    def regrets_at_timestep(regrets, t):
        return tf.gather_nd(regrets, ((t,),))

    @staticmethod
    def update_regrets_at_timestep(regrets, t, inst_regrets, **kwargs):
        return tf.scatter_nd_add(
            regrets,
            ((t,),),
            tf.reshape(
                inst_regrets,
                shape=(
                    1,
                    inst_regrets.shape[0].value, inst_regrets.shape[1].value
                )
            ),
            **kwargs
        )

    def __init__(self, mdp):
        self.regrets = self.__class__.new_table(
            mdp.horizon, mdp.num_states(), mdp.num_actions()
        )

    def at_timestep(self, t):
        return self.__class__.regrets_at_timestep(self.regrets, t)

    def update_at_timestep(self, t, inst_regrets, **kwargs):
        '''
        Params:
        - t: Timestep.
        - inst_regrets: Rank-2 Tensor (|S| by |A|).

        Returns:
        - self, for chaining.
        '''
        self.__class__.update_regrets_at_timestep(
            self.regrets,
            t,
            inst_regrets,
            **kwargs
        )
        return self
