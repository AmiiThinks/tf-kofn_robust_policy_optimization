#!/usr/bin/env python
import tensorflow as tf
import sys
import os
import numpy as np
from amii_tf_mdp.environments.inventory import InventoryMdpGenerator
from amii_tf_mdp.robust.k_of_n import k_of_n_pr_mdp_evs, k_of_n_ev, \
    k_of_n_mdp_weights, k_of_n_regret_update
from amii_tf_mdp.regret_table import PrRegretMatchingPlus
from amii_tf_mdp.pr_mdp import pr_mdp_rollout
from amii_tf_mdp.utils.timer import Timer
from amii_tf_mdp.utils.quadrature import midpoint_quadrature
from amii_tf_mdp.utils.experiment import PickleExperiment


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
    'n_weights': [0.0] * 29 + [1.0],
    'max_inventory': 29,
    'horizon': 2,
    'methods': {}
}
num_sampled_mdps = len(final_data['n_weights'])
num_states = final_data['max_inventory'] + 1
num_actions = final_data['max_inventory'] + 1

g = InventoryMdpGenerator(final_data['max_inventory'], 0.5, 1, 1.1)

mdp_generator_timer = Timer('Create MDP Generator Nodes')
with mdp_generator_timer:
    roots = [g.root() for _ in range(num_sampled_mdps)]
    transition_models = [
        g.transitions(
            *g.fraction_of_max_inventory_gaussian_demand(
                tf.random_uniform([], minval=0.3, maxval=0.7)
            )
        ) for _ in range(num_sampled_mdps)
    ]
    reward_models = [g.rewards() for _ in range(num_sampled_mdps)]
final_data['mdp_generation_duration_s'] = mdp_generator_timer.duration_s()
mdp_generator_timer.log_duration_s()

# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        for i in [0] + list(range(4, num_sampled_mdps, 5)):
            name = 'k={}'.format(i + 1)
            print('# {}'.format(name))

            final_data['methods'][name] = {'name': name}
            d = final_data['methods'][name]

            d['k_weights'] = [0.0] * num_sampled_mdps
            d['k_weights'][i] = 1.0

            rm_plus = PrRegretMatchingPlus(
                final_data['horizon'],
                num_states,
                num_actions
            )
            rm_plus.regrets.initializer.run()

            chance_prob_sequence_list = [
                pr_mdp_rollout(
                    final_data['horizon'],
                    roots[i],
                    transition_models[i]
                )
                for i in range(len(roots))
            ]
            evs = k_of_n_pr_mdp_evs(
                final_data['horizon'],
                chance_prob_sequence_list,
                reward_models,
                rm_plus.strat
            )
            mdp_weights = k_of_n_mdp_weights(
                final_data['n_weights'],
                d['k_weights'],
                evs
            )
            ev = k_of_n_ev(evs, mdp_weights)
            regret_update = k_of_n_regret_update(
                chance_prob_sequence_list,
                reward_models,
                mdp_weights,
                rm_plus
            )

            experiment.reset_random_state()

            regret_update_timer = Timer('Regret Update')
            training_evs = {0: sess.run(ev)}
            with regret_update_timer:
                for t in range(final_data['num_training_iterations']):
                    gadget_ev, _ = sess.run([ev, regret_update])
                    training_evs[t + 1] = gadget_ev
                    if t % 10 == 0:
                        sys.stdout.write('{}    {}\r'.format(t, gadget_ev))
            print('')
            regret_update_timer.log_duration_s()
            print('')

            eval_timer = Timer('Evaluation')
            evaluation_evs = np.array([])
            with eval_timer:
                for i in range(final_data['num_eval_mdp_reps']):
                    evaluation_evs = np.concatenate(
                        [
                            evaluation_evs,
                            sess.run(evs)
                        ]
                    )
                evaluation_evs.sort()
                area = midpoint_quadrature(evaluation_evs, (0, 100))
            if len(evaluation_evs) > 0:
                print('# Bottom: ')
                print(evaluation_evs[:5])
                print('# Top: ')
                print(evaluation_evs[-5:])
                print(
                    '# Area: {}'.format(area)
                )
            eval_timer.log_duration_s()
            print('')
            print('')

            if 'num_sequences' not in final_data:
                final_data['num_sequences'] = (
                    rm_plus.regrets.shape[0].value *
                    rm_plus.regrets.shape[1].value
                )
            d['training'] = {
                'evs': training_evs,
                'duration_s': regret_update_timer.duration_s()
            }
            d['evaluation'] = {
                'evs': evaluation_evs,
                'duration_s': eval_timer.duration_s(),
                'midpoint_area': area
            }
        experiment.save(final_data, 'every_k')
