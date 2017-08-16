def num_pr_sequences(horizon, num_states, num_actions):
    return int(
        num_states * (
            (num_states * num_actions)**(horizon + 1) - 1
        ) / (
            num_states * num_actions - 1
        )
    )


def num_ir_sequences(horizon, num_states): return horizon * num_states


def num_pr_sequences_at_timestep(t, num_states, num_actions):
    return num_states * (num_states * num_actions) ** t
