import numpy as np
import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex


def block_ones(num_blocks, block_width):
    '''
    Returns a block_width-wide block-diagonal matrix of ones with dimension
    num_blocks x (num_blocks * block_width)
    '''
    return np.kron(
        np.identity(num_blocks, dtype='float32'),
        np.ones([1, block_width], dtype='float32')
    )


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


def row_normalize_op(P):
    return l1_projection_to_simplex(P, row_normalize=True)


def ind_max_op(A, tolerance=1e-15, axis=1):
    assert(axis < 2)
    return l1_projection_to_simplex(
        tf.where(
            tf.greater_equal(
                A,
                tf.expand_dims(
                    tf.reduce_max(A, axis=axis) - tolerance,
                    axis=axis
                )
            ),
            tf.ones_like(A),
            tf.zeros_like(A)
        ),
        row_normalize=True if axis == 1 else False
    )
