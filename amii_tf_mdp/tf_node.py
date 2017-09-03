def matching_key_with_different_value(dict_a, dict_b):
    outer_dict, inner_dict = dict_b, dict_a
    if len(dict_b) < len(dict_a):
        outer_dict, inner_dict = dict_a, dict_b
    for k in outer_dict.keys():
        if k in inner_dict:
            if outer_dict[k] != inner_dict[k]: return k
    return None


class BoundTfNode(object):
    def __init__(self, components, feed_dict={}):
        self.components = components
        self.feed_dict = feed_dict

    def combine(self, other):
        k = matching_key_with_different_value(
            self.feed_dict,
            other.feed_dict
        )
        if k is not None:
            raise Exception(
                'Incompatible feed dicts in BoundTfNodes at key "{}".'.format(
                    k.name
                )
            )
        new_node = BoundTfNode(
            self.components + other.components,
            self.feed_dict
        )
        new_node.feed_dict.update(other.feed_dict)
        return new_node

    def run(self, sess):
        return sess.run(self.components, feed_dict=self.feed_dict)


class UnboundTfNode(object):
    def __init__(self, component, feed_dict_generator):
        self.component = component
        self._feed_dict_generator = feed_dict_generator

    def __call__(self, *args, **kwargs):
        return BoundTfNode(
            [self.component],
            self._feed_dict_generator(*args, **kwargs)
        )
