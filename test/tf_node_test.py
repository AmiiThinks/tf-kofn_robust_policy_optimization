import tensorflow as tf
from amii_tf_mdp.tf_node import UnboundTfNode, BoundTfNode, \
    matching_key_with_different_value


class TfNodeTest(tf.test.TestCase):
    def test_matching_key_with_different_value(self):
        a = {'a': 1, 'b': 2}
        b = {'c': 3}
        self.assertIsNone(matching_key_with_different_value(a, b))

        c = {'a': 9, 'c': 3}
        self.assertEqual(matching_key_with_different_value(a, c), 'a')

    def test_bound_tf_node_without_placeholders(self):
        with self.test_session() as sess:
            x_a = [1.3, 5.3, 8.9]
            x_b = [3.6, 4.3]
            patient = BoundTfNode([tf.constant(x_a), tf.constant(x_b)])
            a, b = patient.run(sess)

            self.assertAllClose(x_a, a)
            self.assertAllClose(x_b, b)

    def test_bound_tf_node_with_placeholders(self):
        with self.test_session() as sess:
            c = tf.placeholder(tf.float32, [3])
            x_b = [3.6, 4.3]

            patient = BoundTfNode(
                [
                    [1.3, 5.3, 8.9] * c,
                    tf.constant(x_b)
                ],
                {c: [1, 2, 3.0]}
            )
            a, b = patient.run(sess)

            self.assertAllClose(
                [1.3, 10.6, 26.7],
                a
            )
            self.assertAllClose(x_b, b)

    def test_bound_tf_node_combine_with_placeholders_incompatible(self):
        with self.test_session() as sess:
            c = tf.placeholder(tf.float32, [3])

            patient1 = BoundTfNode(
                [[1.3, 5.3, 8.9] * c],
                {c: [1, 2, 3.0]}
            )
            patient2 = BoundTfNode(
                [[3.6, 4.3, 6.1] + c],
                {c: [11, 2, 3.0]}
            )
            with self.assertRaises(Exception):
                patient1.combine(patient2)

    def test_bound_tf_node_combine_with_placeholders_compabitable(self):
        with self.test_session() as sess:
            c = tf.placeholder(tf.float32, [3])

            patient1 = BoundTfNode(
                [[1.3, 5.3, 8.9] * c],
                {c: [1, 2, 3.0]}
            )
            patient2 = BoundTfNode(
                [[3.6, 4.3, 6.1] + c],
                {c: [1, 2, 3.0]}
            )
            patient = patient1.combine(patient2)
            a, b = patient.run(sess)
            self.assertAllClose([1.3, 10.6, 26.7], a)
            self.assertAllClose([4.6, 6.3, 9.1], b)

    def test_unbound_tf_node(self):
        with self.test_session() as sess:
            c = tf.placeholder(tf.float32, [3])

            patient1 = UnboundTfNode(
                [1.3, 5.3, 8.9] * c,
                lambda c_val: {c: c_val}
            )
            patient2 = UnboundTfNode(
                [3.6, 4.3, 6.1] + c,
                lambda c_val: {c: c_val}
            )
            patient = patient1([1, 2, 3.0]).combine(patient2([1, 2, 3.0]))
            a, b = patient.run(sess)
            self.assertAllClose([1.3, 10.6, 26.7], a)
            self.assertAllClose([4.6, 6.3, 9.1], b)


if __name__ == '__main__':
    tf.test.main()
