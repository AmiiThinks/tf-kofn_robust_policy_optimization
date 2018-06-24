import tensorflow as tf
import numpy as np


def weighted_random_shuffle(weights, k=None):
    ''' Kolonko and Wasch’s
        _Sequential Reservoir Sampling with a Nonuniform Distribution_
        reseviour sampling algorithm.
    '''
    if k is None: k = weights.shape[0]
    adjusted_weights = tf.where(
        tf.greater(weights, 0.0),
        tf.log(tf.random_uniform(weights.shape)) / weights,
        tf.constant(float('-inf'), shape=weights.shape))
    values, indices = tf.nn.top_k(adjusted_weights, k=k)
    return indices


def sample_without_replacement(weights, k=1):
    return weighted_random_shuffle(weights, k=k)


def ind_set(i, n, value=1.0):
    return tf.reshape(
        tf.scatter_nd(
            tf.reshape(i, shape=(1, 1)),
            tf.constant(value, shape=(1, 1)),
            shape=(n, 1)),
        shape=(n, ))


def sample(sess, node, num_reps=1):
    return np.array([sess.run(node) for _ in range(num_reps)])
