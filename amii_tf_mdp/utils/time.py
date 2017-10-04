import time


class TimePrinter(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self, *args, **kwargs):
        self.s = time.perf_counter()

    def __exit__(self, *args, **kwargs):
        print(
            '# {} block took {} s'.format(
                self.name,
                time.perf_counter() - self.s
            ),
            flush=True
        )
