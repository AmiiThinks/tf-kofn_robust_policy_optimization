#!/usr/bin/env python
import random
import numpy as np
import tensorflow as tf
from amii_tf_mdp.environments.inventory import InventoryMdpGenerator
from amii_tf_mdp.robust.k_of_n import PrKofnGadget
from amii_tf_mdp.regret_table import RegretMatchingPlusPr
from amii_tf_mdp.mdp import FixedHorizonMdp, PrMdpState


def sample_mdp(max_inventory=29, horizon=2):
    print('# sample_mdp')
    g = InventoryMdpGenerator(max_inventory, 0.5, 1, 1.1)
    mdp = g.mdp(
        *g.fraction_of_max_inventory_gaussian_demand(
            np.random.uniform(low=3.0, high=7.0, size=1)[0]
        )
    )
    return FixedHorizonMdp.upgrade(horizon, mdp)


def train_and_eval(sess, T, n_weights, k_weights, sample_mdp, num_eval_mdps):
    gadget = PrKofnGadget(n_weights, k_weights, sample_mdp)
    rm_plus = RegretMatchingPlusPr(2, 30, 30)
    print(rm_plus.regrets.shape[0].value)
    rm_plus.regrets.initializer.run()

    def update(unrolled_weighted_rewards):
        n = rm_plus.updated_regrets(
            sum(
                [
                    rm_plus.instantaneous_regrets(r)
                    for r in unrolled_weighted_rewards
                ],
                tf.zeros_like(rm_plus.regrets)
            )
        )
        return n

    for _ in range(T):
        n, ev = gadget(rm_plus.strat(), update)
        print(sess.run(ev))
        sess.run(n)
    evs = tf.stack(
        [
            sample_mdp().expected_value(rm_plus.strat())
            for _ in range(num_eval_mdps)
        ],
        axis=0
    )
    evs, _ = tf.nn.top_k(evs, evs.shape[0].value, sorted=True)
    evs = sess.run(tf.reverse(evs, [0]))
    print(evs)


def sample():
    global state
    global sampled_mdps
    random.shuffle(sampled_mdps)
    mdp = sampled_mdps[0]
    if state is None:
        state = PrMdpState(mdp)
        state.sequences.initializer.run()
    else:
        state.mdp = mdp
    return state


T = 100
num_eval_mdps = 100
n_weights = [0.0] * 9 + [1.0]
state = None

# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    with sess.as_default():
        with tf.device('/cpu:0'):
            # num_sampled_mdps = T * len(n_weights) + num_eval_mdps
            num_sampled_mdps = num_eval_mdps
            print('# Sampling {} MDPs'.format(num_sampled_mdps))
            sampled_mdps = [sample_mdp() for _ in range(num_sampled_mdps)]

        train_and_eval(
            sess,
            T,
            n_weights,
            [1.0] + [0.0] * 9,
            sample,
            num_eval_mdps
        )

        train_and_eval(
            sess,
            T,
            n_weights,
            [0.0] * 9 + [1.0],
            sample,
            num_eval_mdps
        )

        train_and_eval(
            sess,
            T,
            n_weights,
            [1.0] * 10,
            sample,
            num_eval_mdps
        )
