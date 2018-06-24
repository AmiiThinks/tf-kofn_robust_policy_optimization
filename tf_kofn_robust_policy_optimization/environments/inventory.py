import tensorflow as tf
from ..utils.tensor import l1_projection_to_simplex


class InventoryMdpGenerator(object):
    class Cache(object):
        def __init__(
            self,
            resale_cost,
            maintenance_cost,
            num_states,
            num_actions,
            max_inventory,
            wholesale_cost
        ):  # yapf:disable
            self.indices = []
            self.clearing_indices = []
            self.clearing_T_params = []
            self.remaining_indices = []
            self.remaining_T_params = []
            self.R_vals = []

            for s in range(num_states):
                for a in range(num_actions):
                    usable_inventory = min(s + a, max_inventory)
                    restocking_cost = wholesale_cost * a

                    self.indices.append([s, a, 0])
                    self.clearing_indices.append([s, a, 0])
                    self.clearing_T_params.append(usable_inventory)
                    self.R_vals.append(
                        resale_cost * usable_inventory - restocking_cost)

                    for s_prime in range(1, usable_inventory + 1):
                        d = usable_inventory - s_prime

                        self.indices.append([s, a, s_prime])
                        self.remaining_indices.append([s, a, s_prime])
                        self.remaining_T_params.append(d)
                        self.R_vals.append(
                            resale_cost * d -
                            restocking_cost -
                            maintenance_cost * s_prime
                        )  # yapf:disable

    @classmethod
    def gaussian_demand(cls, mean, std):
        def prob_of_demand_when_inventory_clears(d):
            lb = d - 0.5
            return 1.0 - tf.contrib.distributions.Normal(mean, std).cdf(lb)

        def prob_of_demand_when_inventory_remains(d):
            dist = tf.contrib.distributions.Normal(mean, std)
            lb = d - 0.5
            ub = d + 0.5
            return dist.cdf(ub) - dist.cdf(lb)

        return (prob_of_demand_when_inventory_clears,
                prob_of_demand_when_inventory_remains)

    def __init__(self, max_inventory, cost_to_revenue, wholesale_cost, markup):
        self.max_inventory = max_inventory
        self.cost_to_revenue = cost_to_revenue
        self.wholesale_cost = wholesale_cost
        self.markup = markup
        self.cache = None

    def resale_cost(self):
        return self.wholesale_cost * (1.0 + self.markup)

    def maintenance_cost(self):
        return self.cost_to_revenue * self.resale_cost() - self.wholesale_cost

    def num_states(self):
        return self.max_inventory + 1

    def num_actions(self):
        return self.num_states()

    def fraction_of_max_inventory_gaussian_demand(self, fraction):
        mean = fraction * self.max_inventory
        return self.__class__.gaussian_demand(
            mean,
            tf.minimum(mean, self.max_inventory - mean) / 3.0)

    def root(self):
        return l1_projection_to_simplex(
            tf.random_uniform((self.num_states(), )))

    def rewards(self):
        if self.cache is None:
            self.cache = self.__class__.Cache(
                self.resale_cost(),
                self.maintenance_cost(),
                self.num_states(),
                self.num_actions(),
                self.max_inventory,
                self.wholesale_cost
            )  # yapf:disable
        return tf.scatter_nd(
            self.cache.indices,
            self.cache.R_vals,
            shape=(self.num_states(), self.num_actions(), self.num_states()))

    def transitions(self, prob_of_demand_when_inventory_clears,
                    prob_of_demand_when_inventory_remains):
        if self.cache is None:
            self.cache = self.__class__.Cache(
                self.resale_cost(),
                self.maintenance_cost(),
                self.num_states(),
                self.num_actions(),
                self.max_inventory,
                self.wholesale_cost
            )  # yapf:disable
        T = (
            tf.scatter_nd(
                self.cache.clearing_indices,
                prob_of_demand_when_inventory_clears(
                    tf.constant(self.cache.clearing_T_params, dtype=tf.float32)
                ),
                shape=(
                    self.num_states(), self.num_actions(), self.num_states())
            ) + tf.scatter_nd(
                self.cache.remaining_indices,
                prob_of_demand_when_inventory_remains(
                    tf.constant(
                        self.cache.remaining_T_params, dtype=tf.float32)
                ),
                shape=(
                    self.num_states(), self.num_actions(), self.num_states())
            )
        )  # yapf:disable

        z = tf.reduce_sum(T, axis=2)
        z = tf.where(tf.greater(z, tf.zeros_like(z)), z, tf.ones_like(z))
        return T / z
