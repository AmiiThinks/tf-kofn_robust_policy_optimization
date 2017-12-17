import time


class Timer(object):
    def __init__(self, name):
        self.name = name
        self.start_time = 0
        self.end_time = self.start_time

    def __enter__(self, *args, **kwargs):
        self.start_time = time.perf_counter()
        self.end_time = self.start_time

    def __exit__(self, *args, **kwargs):
        self.end_time = time.perf_counter()

    def duration_s(self): return self.end_time - self.start_time

    def log_duration_s(self):
        print(
            '# {} block took {} s'.format(self.name, self.duration_s()),
            flush=True
        )
