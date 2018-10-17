def cache(func, **kwargs):
    attribute = '_{}'.format(func.__name__)

    @property
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator
