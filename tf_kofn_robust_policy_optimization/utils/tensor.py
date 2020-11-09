import numpy as np
import tensorflow as tf


def standardize_batch_dim(*tensors_with_ndims):
    tensors_with_ndims = [(tf.convert_to_tensor(t), ndims)
                          for t, ndims in tensors_with_ndims]
    tensors = [t for t, _ in tensors_with_ndims]
    has_batch_dim_list = [
        len(t.shape) > ndims for t, ndims in tensors_with_ndims
    ]
    has_batch_dim = any(has_batch_dim_list)

    if has_batch_dim:
        batch_sizes = [
            t.shape[0] if has_batch_dim_list[i] else 1
            for i, (t, _) in enumerate(tensors_with_ndims)
        ]
        max_batch_size = max(batch_sizes)

        for i, (t, ndims) in enumerate(tensors_with_ndims):
            if not has_batch_dim_list[i]:
                t = tf.tile(
                    tf.expand_dims(t, 0), [max_batch_size] + [1] * ndims)
            elif batch_sizes[i] < max_batch_size:
                t = tf.tile(t,
                            [max_batch_size // batch_sizes[i]] + [1] * ndims)
            tensors[i] = t
    return tensors, has_batch_dim


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
    n = A.shape[0]
    tiled_A = tf.tile(A, [1, n])
    mask = tf.constant(block_ones(n, A.shape[1]))
    return tiled_A * mask
