from os import path
from amii_tf_nn.experiment import Experiment
from .__init__ import save_pkl, load_pkl


class PickleExperiment(Experiment):
    def save(self, data, name):
        save_pkl(data, path.join(self.path(), name))
        return self

    def load(self, name):
        return load_pkl(path.join(self.path(), name))
