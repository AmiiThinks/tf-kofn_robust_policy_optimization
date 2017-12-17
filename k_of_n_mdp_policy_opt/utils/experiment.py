import pickle
from os import path
from amii_tf_nn.experiment import Experiment


class PickleExperiment(Experiment):
    def save(self, data, name):
        with open(path.join(self.path(), '{}.pkl'.format(name)), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def load(self, name):
        with open(path.join(self.path(), '{}.pkl'.format(name)), 'rb') as f:
            data = pickle.load(f)
        return data
