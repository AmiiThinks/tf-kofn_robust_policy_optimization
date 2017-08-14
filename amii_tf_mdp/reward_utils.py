import tensorflow as tf
from .probability_utils import prob_state_and_action


def reward_distribution(rewards, state, strat=None):
    return tf.tensordot(
        rewards,
        prob_state_and_action(
            state,
            strat=strat,
            num_actions=rewards.shape[1].value
        ),
        axes=((0, 1), (0, 1))
    )
