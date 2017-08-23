from ..mdp import Mdp
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

    def mdp(
        self,
        prob_of_demand_when_inventory_clears,
        prob_of_demand_when_inventory_remains
    ):
        R = []
        T = []
        for s in range(self.num_states()):
            R_s = []
            T_s = []
            for a in range(self.num_actions()):
                R_s_a = []
                T_s_a = []
                for s_prime in range(self.num_states()):
                    usable_inventory = min(s + a, self.max_inventory)
                    d = usable_inventory - s_prime
                    if d < 0:
                        T_s_a.append(0.0)
                        R_s_a.append(0.0)
                    else:
                        if s_prime < 1:
                            prob_d = prob_of_demand_when_inventory_clears(d)
                        else:
                            prob_d = prob_of_demand_when_inventory_remains(d)
                        T_s_a.append(prob_d)

                        reward = (
                            self.resale_cost() * d -
                            self.wholesale_cost * a -
                            self.maintenance_cost() * s_prime
                        )
                        R_s_a.append(reward)
                R_s.append(R_s_a)

                T_s_a = np.array(T_s_a)
                T_s_a = T_s_a / T_s_a.sum()
                T_s_a[np.isnan(T_s_a)] = 0.0
                T_s.append(T_s_a)
            R.append(R_s)
            T.append(T_s)
        return Mdp(
            tf.constant(np.array(T), dtype=tf.float32),
            tf.constant(np.array(R), dtype=tf.float32)
        )
