import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex


def prob_ith_element(n_weights, k_weights):
    '''
    Linear time algorithm to find the probability that the ith element is
    included given weights that chance uses to sample N and k.

    The naive algorithm for computing this is quadratic time. To derive
    the linear time algorithm, look at the probability that k is less than i
    (the completment of the desired probability) and
    split the sum into the part when N = n, for n < i, and the part where
    n >= i.
    '''
    n_prob = l1_projection_to_simplex(n_weights)
    a = n_prob / tf.cumsum(k_weights)
    a = tf.where(tf.is_nan(a), tf.zeros_like(a), a)
    b = tf.cumsum(k_weights, exclusive=True) * tf.cumsum(a, reverse=True)
    return 1.0 - (tf.cumsum(n_prob, exclusive=True) + b)
