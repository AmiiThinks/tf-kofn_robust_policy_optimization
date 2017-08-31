from ..mdp import MdpAnchor
from scipy.stats import norm
import numpy as np
import tensorflow as tf


class InventoryMdpGenerator(object):
    @classmethod
    def gaussian_demand(cls, mean, std):
        def prob_of_demand_when_inventory_clears(d):
            lb = d - 0.5
            return 1.0 - norm.cdf(lb, loc=mean, scale=std)

        def prob_of_demand_when_inventory_remains(d):
            lb = d - 0.5
            ub = d + 0.5
            return (
                norm.cdf(
                    ub,
                    loc=mean,
                    scale=std
                ) -
                norm.cdf(lb, loc=mean, scale=std)
            )
        return (
            prob_of_demand_when_inventory_clears,
            prob_of_demand_when_inventory_remains
        )

    def __init__(self, max_inventory, cost_to_revenue, wholesale_cost, markup):
        self.max_inventory = max_inventory
        self.cost_to_revenue = cost_to_revenue
        self.wholesale_cost = wholesale_cost
        self.markup = markup

    def resale_cost(self):
        return self.wholesale_cost * (1.0 + self.markup)

    def maintenance_cost(self):
        return self.cost_to_revenue * self.resale_cost() - self.wholesale_cost

    def num_states(self): return self.max_inventory + 1

    def num_actions(self): return self.num_states()

    def fraction_of_max_inventory_gaussian_demand(self, fraction):
        mean = fraction * self.max_inventory
        return self.__class__.gaussian_demand(
            mean,
            min(mean, self.max_inventory - mean) / 3.0
        )

    def transition_and_rewards(
        self,
        prob_of_demand_when_inventory_clears,
        prob_of_demand_when_inventory_remains
    ):
        '''
        TODO This function is really slow right now since it uses nested
        loops and isn't in TensorFlow.
        '''
        resale_cost = self.resale_cost()
        maintenance_cost = self.maintenance_cost()
        num_states = self.num_states()
        num_actions = self.num_actions()

        R = np.zeros([num_states, num_actions, num_states])
        T = np.zeros([num_states, num_actions, num_states])
        for s in range(num_states):
            for a in range(num_actions):
                usable_inventory = min(s + a, self.max_inventory)
                restocking_cost = self.wholesale_cost * a

                T[s, a, 0] = prob_of_demand_when_inventory_clears(
                    usable_inventory
                )
                R[s, a, 0] = resale_cost * usable_inventory - restocking_cost
                for s_prime in range(1, usable_inventory + 1):
                    d = usable_inventory - s_prime
                    T[s, a, s_prime] = prob_of_demand_when_inventory_remains(d)
                    R[s, a, s_prime] = (
                        resale_cost * d -
                        restocking_cost -
                        maintenance_cost * s_prime
                    )
        T = T / T.sum(axis=2)
        T[np.isnan(T)] = 0.0
        return (T, R)
