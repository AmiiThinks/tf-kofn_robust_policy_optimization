import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from ..utils.tf_node import UnboundTfNode


def prob_ith_element(n_weights, k_weights):
    '''
    Linear time algorithm to find the probability that the ith element is
    included given weights that chance uses to sample N and k.

    The naive algorithm for computing this is quadratic time. To derive
    the linear time algorithm, look at the probability that k is less than i
    (the completment of the desired probability) and
    split the sum into the part when N = n for n < i and the part where
    n >= i.
    '''
    n_prob = l1_projection_to_simplex(n_weights)
    w_n_bar = tf.cumsum(k_weights)
    a = n_prob / w_n_bar
    a = tf.where(tf.is_nan(a), tf.zeros_like(a), a)
    w_i_minus_one_bar = tf.cumsum(k_weights, exclusive=True)
    b = w_i_minus_one_bar * tf.cumsum(a, reverse=True)
    return 1.0 - (tf.cumsum(n_prob, exclusive=True) + b)


class KofnGadget(object):
    def __init__(self, n_weights, k_weights, mdp_generator):
        # TODO Getting the probability that chance selects the ith MDP is not
        # as simple as normalizing the prob of the ith element...
        # TODO This only works when only one k_weight is greater than zero.
        self.i_weights = l1_projection_to_simplex(
            prob_ith_element(n_weights, k_weights)
        )
        self.ev_mdps = [mdp_generator() for _ in range(self.max_num_mdps())]
        self.weighted_reward_mdps = self.ev_mdps
        unbound_individual_evs = sum(
            [
                self.ev_mdps[i].unbound_expected_value.composable(
                    'ev{}'.format(i)
                )
                for i in range(1, len(self.ev_mdps))
            ],
            self.ev_mdps[0].unbound_expected_value.composable('ev0')
        )
        unbound_stacked_evs = UnboundTfNode(
            tf.stack(
                [mdp.unbound_expected_value.component for mdp in self.ev_mdps],
                axis=0
            ),
            name='evs'
        )
        unbound_sorted_mdp_indices = UnboundTfNode(
            # Sorted in ascending order
            tf.reverse(
                # Sort in descending order
                tf.nn.top_k(
                    unbound_stacked_evs.component,
                    k=unbound_stacked_evs.component.shape[-1].value,
                    sorted=True
                )[1],
                [0]
            ),
            name='sorted_mdp_indices'
        )
        unbound_mdp_weights = UnboundTfNode(
            tf.scatter_nd(
                tf.expand_dims(unbound_sorted_mdp_indices.component, dim=1),
                self.i_weights,
                [self.max_num_mdps()]
            ),
            name='mdp_weights'
        )
        unbound_expected_value = UnboundTfNode(
            tf.tensordot(
                unbound_stacked_evs.component,
                unbound_mdp_weights.component,
                1
            ),
            name='gadget_ev'
        )
        self.unbound_ev_dependent_nodes = (
            unbound_individual_evs +
            unbound_stacked_evs.composable() +
            unbound_sorted_mdp_indices.composable() +
            unbound_mdp_weights.composable() +
            unbound_expected_value.composable()
        )

        def _weighted_rewards_fdg(mdp):
            def _f(transition_model, rewards, root=None):
                d = {
                    mdp.transition_model: transition_model,
                    mdp.rewards: rewards
                }
                if root is not None:
                    d[mdp.root] = root
                return d
            return _f

        self.unbound_weighted_rewards = [
            UnboundTfNode(
                (
                    self.weighted_reward_mdps[i].unbound_sequences.component *
                    self.weighted_reward_mdps[i].rewards *
                    unbound_mdp_weights.component[i]
                ),
                feed_dict_generator=_weighted_rewards_fdg(
                    self.weighted_reward_mdps[i]
                ),
                name='weighted_rewards{}'.format(i)
            ) for i in range(len(self.weighted_reward_mdps))
        ]
        self.unbound_nodes = (
            self.unbound_ev_dependent_nodes +
            sum(
                [r.composable() for r in self.unbound_weighted_rewards[1:]],
                self.unbound_weighted_rewards[0].composable()
            )
        )

    def bind(self, strat, *transition_reward_root_tuples):
        kwargs = {
            'sorted_mdp_indices': [{}],
            'mdp_weights': [{}],
            'gadget_ev': [{}]
        }
        for i in range(len(transition_reward_root_tuples)):
            transition_model, rewards, root = (
                transition_reward_root_tuples[i]
            )
            kwargs['ev{}'.format(i)] = [
                transition_model,
                rewards,
                {'root': root, 'strat': strat}
            ]
            kwargs['weighted_rewards{}'.format(i)] = [
                transition_model,
                rewards,
                {'root': root}
            ]
        return self.unbound_nodes(**kwargs)

    def max_num_mdps(self): return self.i_weights.shape[0].value

    def create_regret_update_node(self, learner):
        inst_regrets = [
            learner.instantaneous_regrets(r.component)
            for r in self.unbound_weighted_rewards
        ]
        return learner.updated_regrets(
            sum(inst_regrets[1:], inst_regrets[0])
        )
