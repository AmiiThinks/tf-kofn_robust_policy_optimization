import numpy as np
import tensorflow as tf


def block_ones(num_blocks, block_width):
    '''
    Returns a block_width-wide block-diagonal matrix of ones with dimension
    num_blocks x (num_blocks * block_width)
    '''
    return np.kron(
        np.identity(num_blocks, dtype='float32'),
        np.ones([1, block_width], dtype='float32'))


def matrix_to_block_matrix_op(A):
    '''
    A: n x m
    Returns an n x (n * m) block diagonal matrix that results from offseting
    each row in A by the number of columns of A.
    '''
    n = A.shape[0].value
    tiled_A = tf.tile(A, [1, n])
    mask = tf.constant(block_ones(n, A.shape[1].value))
    return tiled_A * mask


def normalized(v, axis=0):
    v = tf.convert_to_tensor(v)
    n = tf.shape(v)[axis]
    dims_shape = [tf.rank(v)]
    tile_dims = tf.maximum(
        tf.ones(shape=dims_shape, dtype=tf.int32),
        tf.scatter_nd([[axis]], [n], shape=dims_shape))

    z = tf.tile(tf.reduce_sum(v, axis=axis, keepdims=True), tile_dims)
    ur = tf.constant(1.0 / tf.cast(n, tf.float32), shape=v.shape)
    return tf.where(tf.greater(z, 0.0), v / z, ur)


def l1_projection_to_simplex(v, axis=0):
    return normalized(tf.maximum(0.0, v), axis=axis)


def ind_max_op(A, axis=0, tolerance=1e-15):
    almost_max = tf.reduce_max(A, axis=axis, keepdims=True) - tolerance
    return l1_projection_to_simplex(
        tf.where(
            tf.greater_equal(A, almost_max),
            tf.ones_like(A),
            tf.zeros_like(A)
        ),
        axis=axis)  # yapf:disable
