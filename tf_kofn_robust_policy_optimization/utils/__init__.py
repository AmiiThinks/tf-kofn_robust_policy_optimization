import tensorflow as tf
import pickle
from .probability import *
from .sampling import *
from .sequence import *
from .random import *
from .tensor import *


def midpoint_quadrature(y, x_bounds):
    y_mid = (y[:-1] + y[1:]) / 2.0
    return ((x_bounds[1] - x_bounds[0]) / len(y) * y_mid).sum()


def save_pkl(data, path):
    with open('{}.pkl'.format(path), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open('{}.pkl'.format(path), 'rb') as f:
        return pickle.load(f)


def means(*tensors, n=1):
    if len(tensors) < 1:
        return None, None
    elif n > 1:
        tensor_lists = zip(*tensors)
    else:
        tensor_lists = [tensors]

    _mean_tensors = []
    for tensor_list in tensor_lists:
        first_tensor = tensor_list[0]
        tensor_rank = tf.rank(tf.convert_to_tensor(first_tensor))
        tensor_tensor = tf.concat(
            [tf.expand_dims(tensor, axis=tensor_rank) for tensor in tensor_list],
            axis=tensor_rank)
        _mean_tensors.append(tf.reduce_mean(tensor_tensor, axis=tensor_rank))
    return _mean_tensors


def num_actions(transitions):
    return transitions.shape[1].value


def num_states(transitions):
    return transitions.shape[0].value
