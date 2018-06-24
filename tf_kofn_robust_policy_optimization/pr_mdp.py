import tensorflow as tf
from .utils import num_states, num_actions
from .utils.sequence import num_pr_prefixes, \
    prob_next_sequence_state_action_and_next_state


def pr_mdp_rollout(horizon, root, transitions):
    sequence_blocks = [
        prob_next_sequence_state_action_and_next_state(
            transitions, tf.reshape(root, (1, 1, root.shape[0].value)))
    ]
    for _ in range(horizon - 1):
        sequence_blocks.append(
            prob_next_sequence_state_action_and_next_state(
                transitions, sequence_blocks[-1]))
    return tf.concat(sequence_blocks, axis=0)


def pr_mdp_expected_value(horizon, num_states, num_actions,
                          pr_mdp_sequence_probs, rewards, strat):
    current_cf_state_values = None
    strat = tf.reshape(strat, [
        num_pr_prefixes(horizon - 1, num_states, num_actions), num_states,
        num_actions
    ])
    action_rewards_weighted_by_chance = tf.squeeze(
        tf.reduce_sum(pr_mdp_sequence_probs * rewards, axis=3))
    for t in range(horizon - 1, -1, -1):
        n = num_pr_prefixes(t - 1, num_states, num_actions)
        next_n = num_pr_prefixes(t, num_states, num_actions)
        current_rewards = action_rewards_weighted_by_chance[n:next_n, :, :]
        current_cf_action_values = (
            current_rewards if current_cf_state_values is None else
            (current_rewards + tf.reshape(
                tf.reduce_sum(current_cf_state_values, axis=1),
                current_rewards.shape)))
        current_cf_state_values = tf.expand_dims(
            # TODO Should be able to do this with tensordot
            # but it didn't work the first way I tried.
            tf.reduce_sum(
                strat[n:next_n, :, :] * current_cf_action_values, axis=2),
            axis=2)
    return tf.reduce_sum(current_cf_state_values)


def pr_mdp_evs(horizon, chance_prob_sequences, reward_models, strat):
    return tf.stack(
        [
            pr_mdp_expected_value(horizon, num_states(reward_models[i]),
                                  num_actions(reward_models[i]),
                                  chance_prob_sequences[i], reward_models[i],
                                  strat) for i in range(len(reward_models))
        ],
        axis=0)


def pr_mdp_optimal_policy_and_value(horizon, num_states, num_actions,
                                    pr_mdp_sequence_probs, rewards):
    strat_pieces = []
    current_cf_state_values = None

    action_rewards_weighted_by_chance = tf.squeeze(
        tf.reduce_sum(pr_mdp_sequence_probs * rewards, axis=3))
    for t in range(horizon - 1, -1, -1):
        n = num_pr_prefixes(t - 1, num_states, num_actions)
        next_n = num_pr_prefixes(t, num_states, num_actions)
        current_rewards = action_rewards_weighted_by_chance[n:next_n, :, :]
        num_sequences = next_n - n
        current_cf_action_values = (
            current_rewards if current_cf_state_values is None else
            (current_rewards + tf.reshape(
                tf.reduce_sum(current_cf_state_values, axis=1),
                current_rewards.shape)))
        br_indices = tf.expand_dims(
            tf.argmax(
                tf.reshape(current_cf_action_values,
                           (num_sequences * num_states,
                            current_cf_action_values.shape[-1].value)),
                axis=1),
            dim=1)
        scatter_indices = tf.concat(
            [
                tf.expand_dims(
                    tf.range(br_indices.shape[0].value, dtype=tf.int64),
                    axis=1), br_indices
            ],
            axis=1)
        strat_pieces.append(
            tf.reshape(
                tf.scatter_nd(
                    scatter_indices,
                    tf.ones(br_indices.shape[:1]),
                    shape=(num_sequences * num_states, num_actions)),
                (num_sequences, num_states, num_actions)))
        # TODO Is this necessary?
        # with tf.control_dependencies([strat_pieces[-1]]):
        current_cf_state_values = tf.expand_dims(
            # TODO Should be able to do this with tensordot
            # but it didn't work the first way I tried.
            tf.reduce_sum(strat_pieces[-1] * current_cf_action_values, axis=2),
            axis=2)
    br_ev = tf.reduce_sum(current_cf_state_values)
    strat_pieces.reverse()
    return (tf.concat(strat_pieces, axis=0), br_ev)
