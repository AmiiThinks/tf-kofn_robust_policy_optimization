#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from amii_tf_mdp.environments.inventory import InventoryMdpGenerator
from amii_tf_mdp.robust.k_of_n import KofnGadget
from amii_tf_mdp.regret_table import PrRegretMatchingPlus
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp
from amii_tf_mdp.utils.tf_node import UnboundTfNode
from amii_tf_mdp.utils.time import TimePrinter
from amii_tf_mdp.utils.random import reset_random_state
from amii_tf_mdp.utils.quadrature import midpoint_quadrature


def new_rm_plus_learner_and_update_node(gadget):
    rm_plus = PrRegretMatchingPlus.from_gadget(gadget)
    rm_plus.regrets.initializer.run()
    return rm_plus, gadget.create_regret_update_node(rm_plus)


def training(
    sess,
    T,
    gadget,
    unbound_mdp_generator_node,
    rm_plus,
    update_regrets
):
    for t in range(T):
        # with TimePrinter('Create MDP Params'):
        mdp_params = unbound_mdp_generator_node().run(sess)

        # with TimePrinter('Bind'):
        bound_node = gadget.bind(sess.run(rm_plus.strat), *mdp_params)
        bound_node.components['update_regrets'] = update_regrets

        # with TimePrinter('CFR+ Update'):
        yield t, bound_node.run(sess)


def train_and_eval(
    sess,
    T,
    gadget,
    unbound_mdp_generator_node,
    num_eval_mdp_reps
):
    rm_plus, update_regrets = new_rm_plus_learner_and_update_node(gadget)

    print(rm_plus.regrets.shape[0].value * rm_plus.regrets.shape[1].value)

    for t, evaluated_nodes in training(
        sess,
        T,
        gadget,
        unbound_mdp_generator_node,
        rm_plus,
        update_regrets
    ):
        if t % 10 == 0:
            print('{}    {}'.format(t, evaluated_nodes['gadget_ev']))

    strat = sess.run(rm_plus.strat)
    evs = []
    with TimePrinter('Eval'):
        for _ in range(num_eval_mdp_reps):
            # with TimePrinter('Create Test MDP Params'):
            mdp_params = unbound_mdp_generator_node().run(sess)

            # with TimePrinter('Bind Test MDP Params'):
            bound_node = gadget.bind(strat, *mdp_params)

            # with TimePrinter('Bind Test MDP Params'):
            d = bound_node.run(sess)
            evs += [
                d['ev{}'.format(i)] for i in range(gadget.max_num_mdps())
            ]
        if len(evs) > 0:
            evs = sorted(evs)
            print('# Bottom: ')
            print(evs[:10])
            print('# Top: ')
            print(evs[-10:])
            print(
                '# Area: {}'.format(
                    midpoint_quadrature(np.array(evs), (0, 100))
                )
            )


random_seed = 10
reset_random_state(random_seed)
T = 100
num_eval_mdp_reps = 100
n_weights = [0.0] * 49 + [1.0]
# max_inventory = 29
max_inventory = 9
num_states = max_inventory + 1
num_actions = max_inventory + 1

g = InventoryMdpGenerator(max_inventory, 0.5, 1, 1.1)
fraction_placeholders = [
    tf.placeholder(tf.float32, shape=()) for _ in range(len(n_weights))
]
root = l1_projection_to_simplex(tf.random_uniform((num_states,)))

with TimePrinter('Create MDP Generator Nodes'):
    mdp_param_nodes = [
        g.transition_and_rewards_tf(
            *g.fraction_of_max_inventory_gaussian_demand_tf(
                fraction_placeholders[i]
            )
        ) + (root,) for i in range(len(n_weights))
    ]
    mdp_generator = UnboundTfNode(
        mdp_param_nodes,
        (
            lambda *fractions: {
                fraction_placeholders[i]: fractions[i]
                for i in range(len(mdp_param_nodes))
            }
        )
    )


def unbound_mdp_generator_node():
    return mdp_generator(
        *np.random.uniform(low=0.3, high=0.7, size=len(n_weights))
    )


horizon = 2

# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        gadget = KofnGadget(
            n_weights,
            [1.0] + [0.0] * (len(n_weights) - 1),
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# k=1')
        reset_random_state(random_seed)
        train_and_eval(
            sess,
            T,
            gadget,
            unbound_mdp_generator_node,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [0.0, 1.0] + [0.0] * (len(n_weights) - 2),
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# k=2')
        reset_random_state(random_seed)
        train_and_eval(
            sess,
            T,
            gadget,
            unbound_mdp_generator_node,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [0.0] * (len(n_weights) - 2) + [1.0, 0.0],
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# k=N-1')
        reset_random_state(random_seed)
        train_and_eval(
            sess,
            T,
            gadget,
            unbound_mdp_generator_node,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [0.0] * (len(n_weights) - 1) + [1.0],
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# k=N')
        reset_random_state(random_seed)
        train_and_eval(
            sess,
            T,
            gadget,
            unbound_mdp_generator_node,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [1.0] * len(n_weights),
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# Totally Mixed')
        reset_random_state(random_seed)
        train_and_eval(
            sess,
            T,
            gadget,
            unbound_mdp_generator_node,
            num_eval_mdp_reps
        )
