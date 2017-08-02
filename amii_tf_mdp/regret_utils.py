import tensorflow as tf


def tabular_regrets(horizon, num_states, num_actions):
    return tf.Variable(
        tf.constant(0.0, shape=(horizon, num_states, num_actions))
    )


def regrets_at_timestep(regrets, t):
    return tf.gather_nd(regrets, ((t,),))


def update_regrets_at_timestep(regrets, t, inst_regrets, **kwargs):
    return tf.scatter_nd_add(
        regrets,
        ((t,),),
        tf.reshape(
            inst_regrets,
            shape=(1, inst_regrets.shape[0].value, inst_regrets.shape[1].value)
        ),
        **kwargs
    )


def project_to_positive_orthant(weights):
    return tf.where(
        tf.greater(weights, 0.0),
        weights,
        tf.constant(0.0, shape=weights.shape)
    )
