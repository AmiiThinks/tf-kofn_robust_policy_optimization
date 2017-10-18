#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import sys
import os
from amii_tf_nn.projection import l1_projection_to_simplex
from amii_tf_mdp.environments.inventory import InventoryMdpGenerator
from amii_tf_mdp.robust.k_of_n import KofnGadget
from amii_tf_mdp.regret_table import PrRegretMatchingPlus
from amii_tf_mdp.pr_uncertain_mdp import PrUncertainMdp
from amii_tf_mdp.utils.tf_node import UnboundTfNode
from amii_tf_mdp.utils.timer import Timer
from amii_tf_mdp.utils.quadrature import midpoint_quadrature
from amii_tf_mdp.utils.experiment import PickleExperiment


def new_rm_plus_learner_and_update_node(gadget):
    rm_plus = PrRegretMatchingPlus.from_gadget(gadget)
    rm_plus.regrets.initializer.run()
    return rm_plus, gadget.create_regret_update_node(rm_plus)


def evaluate_game(
    sess,
    gadget,
    unbound_mdp_generator_node,
    strat,
    update_regrets=None
):
    mdp_param_timer = Timer('Create MDP Params')
    with mdp_param_timer:
        mdp_params = unbound_mdp_generator_node().run(sess)

    bound_node = gadget.bind(strat, *mdp_params)
    if update_regrets is not None:
        bound_node.components['update_regrets'] = update_regrets

    eval_timer = Timer('Compute Game Value')
    with eval_timer:
        evaluated_gadget_values = bound_node.run(sess)
    return (evaluated_gadget_values, mdp_param_timer, eval_timer)


def training(
    sess,
    T,
    gadget,
    unbound_mdp_generator_node,
    rm_plus,
    update_regrets
):
    for t in range(T):
        strat = sess.run(rm_plus.strat)
        evaluated_gadget_values, mdp_param_timer, update_timer = evaluate_game(
            sess,
            gadget,
            unbound_mdp_generator_node,
            strat,
            update_regrets=update_regrets
        )
        yield t, evaluated_gadget_values, mdp_param_timer, update_timer


def train_strat(sess, T, gadget, unbound_mdp_generator_node):
    rm_plus, update_regrets = new_rm_plus_learner_and_update_node(gadget)
    data = []
    for t, evaluated_nodes, mdp_param_timer, update_timer in training(
        sess,
        T,
        gadget,
        unbound_mdp_generator_node,
        rm_plus,
        update_regrets
    ):
        gadget_ev = evaluated_nodes['gadget_ev']
        data.append(
            {
                'gadget_ev': gadget_ev,
                'mdp_param_duration_s': mdp_param_timer.duration_s(),
                'update_duration_s': update_timer.duration_s()
            }
        )
        if t % 10 == 0:
            sys.stdout.write(
                '{}    {}\r'.format(
                    t,
                    gadget_ev
                )
            )
    print('')
    return rm_plus, data


def evaluate_strat(
    sess,
    gadget,
    unbound_mdp_generator_node,
    strat,
    num_eval_mdp_reps
):
    eval_timer = Timer('Eval')
    evs = []
    with eval_timer:
        for _ in range(num_eval_mdp_reps):
            evaluated_gadget_values, _, __ = (
                evaluate_game(sess, gadget, unbound_mdp_generator_node, strat)
            )
            evs += [
                evaluated_gadget_values['ev{}'.format(i)]
                for i in range(gadget.max_num_mdps())
            ]
        evs = sorted(evs)
    return evs, eval_timer


def train_and_eval(
    sess,
    T,
    gadget,
    unbound_mdp_generator_node,
    num_eval_mdp_reps
):
    rm_plus, train_data = train_strat(
        sess,
        T,
        gadget,
        unbound_mdp_generator_node
    )
    strat = sess.run(rm_plus.strat)
    evs, eval_timer = evaluate_strat(
        sess,
        gadget,
        unbound_mdp_generator_node,
        strat,
        num_eval_mdp_reps
    )
    area = midpoint_quadrature(np.array(evs), (0, 100))
    if len(evs) > 0:
        print('# Bottom: ')
        print(evs[:5])
        print('# Top: ')
        print(evs[-5:])
        print(
            '# Area: {}'.format(area)
        )
    eval_timer.log_duration_s()

    return {
        'num_sequences': (
            rm_plus.regrets.shape[0].value * rm_plus.regrets.shape[1].value
        ),
        'training': train_data,
        'evaluation': {
            'evs': evs,
            'duration_s': eval_timer.duration_s(),
            'midpoint_area': area
        }
    }


random_seed = 10

experiment = PickleExperiment(
    'distribution_inventory_k_of_n',
    root=os.path.join(os.getcwd(), 'tmp'),
    seed=random_seed,
    log_level=tf.logging.INFO
)
experiment.ensure_present()

final_data = {
    'num_training_iterations': 100,
    'num_eval_mdp_reps': 100,
    'n_weights': [0.0] * 49 + [1.0],
    'max_inventory': 9,
    'horizon': 2,
    'methods': {}
}
num_sampled_mdps = len(final_data['n_weights'])
num_states = final_data['max_inventory'] + 1
num_actions = final_data['max_inventory'] + 1

g = InventoryMdpGenerator(final_data['max_inventory'], 0.5, 1, 1.1)
fraction_placeholders = [
    tf.placeholder(tf.float32, shape=()) for _ in range(num_sampled_mdps)
]
root = l1_projection_to_simplex(tf.random_uniform((num_states,)))

mdp_generator_timer = Timer('Create MDP Generator Nodes')
with mdp_generator_timer:
    mdp_param_nodes = [
        g.transition_and_rewards_tf(
            *g.fraction_of_max_inventory_gaussian_demand_tf(
                fraction_placeholders[i]
            )
        ) + (root,) for i in range(num_sampled_mdps)
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
final_data['mdp_generation_duration_s'] = mdp_generator_timer.duration_s()
mdp_generator_timer.log_duration_s()


def unbound_mdp_generator_node():
    return mdp_generator(
        *np.random.uniform(low=0.3, high=0.7, size=num_sampled_mdps)
    )


# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        for i in [0] + list(range(4, num_sampled_mdps, 5)):
            name = 'k={}'.format(i + 1)
            final_data['methods'][name] = {'name': name}
            d = final_data['methods'][name]

            d['k_weights'] = [0.0] * num_sampled_mdps
            d['k_weights'][i] = 1.0
            gadget = KofnGadget(
                final_data['n_weights'],
                d['k_weights'],
                lambda: (
                    PrUncertainMdp(
                        final_data['horizon'],
                        num_states,
                        num_actions
                    )
                )
            )
            print('# {}'.format(name))

            experiment.reset_random_state()
            train_and_eval_data = train_and_eval(
                sess,
                final_data['num_training_iterations'],
                gadget,
                unbound_mdp_generator_node,
                final_data['num_eval_mdp_reps']
            )

            d['num_sequences'] = train_and_eval_data['num_sequences']
            d['training'] = train_and_eval_data['training']
            d['evaluation'] = train_and_eval_data['evaluation']
        experiment.save(final_data, 'every_k')
