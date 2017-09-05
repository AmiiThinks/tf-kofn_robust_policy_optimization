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

    def __add__(self, other): return self.combine(other)

    def is_dict(self): return isinstance(self.components, dict)
    def is_singleton(self): return not self.is_dict()

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

        if (
            (
                self.is_dict() and other.is_singleton()
            ) or (
                other.is_dict() and self.is_singleton()
            )
        ):
            raise Exception('Cannot combine dict and singleton BoundTfNodes.')
        elif self.is_singleton() and other.is_singleton():
            raise Exception('Cannot combine singleton BoundTfNodes.')

        # Both nodes are dicts
        k = matching_key_with_different_value(
            self.components,
            other.components
        )
        if k is not None:
            raise Exception(
                'Components "{}" and "{}" have the same key, "{}".'.format(
                    self.components[k].name,
                    other.components[k].name,
                    k
                )
            )
        new_components = {}
        new_components.update(self.components)
        new_components.update(other.components)

        new_feed_dict = {}
        new_feed_dict.update(self.feed_dict)
        new_feed_dict.update(other.feed_dict)
        return BoundTfNode(new_components, new_feed_dict)

    def __call__(self, sess): return self.run(sess)

    def run(self, sess):
        return sess.run(self.components, feed_dict=self.feed_dict)


class CompositeUnboundTfNode(object):
    def __init__(self, *named_unbound_tf_nodes):
        self.nodes = {}
        for node in named_unbound_tf_nodes:
            if node.name is None:
                raise Exception(
                    'Node must have a name to be part of a CompositeUnboundTfNode.'
                )
            if node.name in self.nodes:
                if self.nodes[node.name] != node:
                    raise Exception(
                        'Nodes being combined in a CompositeUnboundTfNode have the same name, "{}", but different UnboundTfNodes. Component names: "{}" and "{}".'.format(
                            node.name,
                            self.nodes[node.name].component.name,
                            node.component.name
                        )
                    )
            else:
                self.nodes[node.name] = node

    def __add__(self, other): return self.combine(other)
    def combine(self, other):
        return self.__class__(
            *(list(self.nodes.values()) + list(other.nodes.values()))
        )

    def __call__(self, **kwargs):
        k = list(kwargs.keys())
        return sum(
            [
                self.nodes[k[i]](*kwargs[k[i]][0:-1], **kwargs[k[i]][-1])
                for i in range(1, len(k))
            ],
            self.nodes[k[0]](*kwargs[k[0]][0:-1], **kwargs[k[0]][-1])
        )


class UnboundTfNode(object):
    def __init__(self, component, feed_dict_generator=None, name=None):
        self.component = component
        self._feed_dict_generator = (
            (lambda *args, **kwargs: {})
            if feed_dict_generator is None else feed_dict_generator
        )
        self.name = name

    def copy(self):
        return self.__class__(
            self.component,
            feed_dict_generator=self._feed_dict_generator,
            name=self.name
        )

    def __eq__(self, other):
        return (
            self.component == other.component and
            self._feed_dict_generator == other._feed_dict_generator
        )

    def __call__(self, *args, **kwargs):
        return BoundTfNode(
            (
                self.component if self.name is None
                else {self.name: self.component}
            ),
            self._feed_dict_generator(*args, **kwargs)
        )

    def set_name(self, name):
        self.name = name
        return self

    def composable(self, name=None):
        if name is None:
            if self.name is None:
                raise Exception(
                    'UnboundTfNode must have a name to become composable.'
                )
            return CompositeUnboundTfNode(self)
        else:
            return CompositeUnboundTfNode(self.copy().set_name(name))
