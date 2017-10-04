import numpy as np
import tensorflow as tf


def reset_random_state(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
