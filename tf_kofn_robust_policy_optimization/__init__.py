class InputIterator(object):
    def __init__(self, f, input_generator):
        self._f = f
        self._input_generator = input_generator

    def __iter__(self):
        return self

    def __next__(self):
        return self._f(next(self._input_generator))
