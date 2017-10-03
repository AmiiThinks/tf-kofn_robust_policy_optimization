#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from amii_tf_nn.projection import l1_projection_to_simplex
from amii_tf_mdp.environments.inventory import InventoryMdpGenerator
from amii_tf_mdp.robust.k_of_n import KofnGadget
from amii_tf_mdp.regret_table import PrRegretMatchingPlus
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp
from amii_tf_mdp.tf_node import UnboundTfNode
import time


class TimePrinter(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self, *args, **kwargs):
        self.s = time.perf_counter()

    def __exit__(self, *args, **kwargs):
        print(
            '# {} block took {} s'.format(
                self.name,
                time.perf_counter() - self.s
            ),
            flush=True
        )


def midpoint_integration(y):
    y_mid = (y[:-1] + y[1:]) / 2.0
    return sum(100 / len(y) * y_mid)


def train_and_eval(
    sess,
    T,
    gadget,
    unbound_mdp_generator_node,
    num_eval_mdp_reps
):
    rm_plus = PrRegretMatchingPlus(
        gadget.ev_mdps[0].horizon,
        gadget.ev_mdps[0].num_states(),
        gadget.ev_mdps[0].num_actions()
    )
    print(rm_plus.regrets.shape[0].value * rm_plus.regrets.shape[1].value)
    rm_plus.regrets.initializer.run()

    inst_regrets = [
        rm_plus.instantaneous_regrets(r.component)
        for r in gadget.unbound_weighted_rewards
    ]
    update_regrets = rm_plus.updated_regrets(
        sum(inst_regrets[1:], inst_regrets[0])
    )
    for _ in range(T):
        # with TimePrinter('Create MDP Params'):
        mdp_params = unbound_mdp_generator_node(
            *np.random.uniform(
                low=0.3, high=0.7, size=gadget.max_num_mdps()
            )
        ).run(sess)

        # with TimePrinter('Bind'):
        bound_node = gadget.bind(sess.run(rm_plus.strat), *mdp_params)
        bound_node.components['update_regrets'] = update_regrets

        # with TimePrinter('CFR+ Update'):
        bound_node.run(sess)

    strat = sess.run(rm_plus.strat)
    evs = []
    with TimePrinter('Eval'):
        for _ in range(num_eval_mdp_reps):
            # with TimePrinter('Create Test MDP Params'):
            mdp_params = unbound_mdp_generator_node(
                *np.random.uniform(
                    low=0.3,
                    high=0.7,
                    size=gadget.max_num_mdps()
                )
            ).run(sess)

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
                '# Area: {}'.format(midpoint_integration(np.array(evs)))
            )


random_seed = 10


def reset_random_state():
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)


reset_random_state()
T = 100
num_eval_mdp_reps = 1000
n_weights = [0.0] * 9 + [1.0]

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
        reset_random_state()
        train_and_eval(
            sess,
            T,
            gadget,
            mdp_generator,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [0.0] * (len(n_weights) - 1) + [1.0],
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# k=N')
        reset_random_state()
        train_and_eval(
            sess,
            T,
            gadget,
            mdp_generator,
            num_eval_mdp_reps
        )

        gadget = KofnGadget(
            n_weights,
            [1.0] * len(n_weights),
            lambda: PrUncertainMdp(horizon, num_states, num_actions)
        )
        print('# Totally Mixed')
        reset_random_state()
        train_and_eval(
            sess,
            T,
            gadget,
            mdp_generator,
            num_eval_mdp_reps
        )
