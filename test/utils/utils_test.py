import tensorflow as tf
import numpy as np
from k_of_n_mdp_policy_opt.utils import means


class UtilsTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.set_random_seed(42)

    def test_mean_models(self):
        a = np.random.normal(size=[3, 2]).astype('float32')
        b = np.random.normal(size=[3, 2]).astype('float32')

        with self.test_session():
            self.assertAllClose((a + b) / 2.0, means(a, b)[0])

            patient = means((a, b), (b, 2 * a), n=2)
            self.assertAllClose((a + b) / 2.0, patient[0])
            self.assertAllClose((2 * a + b) / 2.0, patient[1])


if __name__ == '__main__':
    tf.test.main()
