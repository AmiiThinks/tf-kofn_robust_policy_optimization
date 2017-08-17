import numpy as np


class DecisionProcessTransitions(object):
    def __init__(self, before, after, action=None):
        self.action = action
        self.before = before
        self.after = after

    def __add__(self, other):
        if self.action != other.action:
            raise ValueError(
                "Attempting to combine Transitions after action {} and {}. Actions must match for two Transitions to be compatible.".format(
                    self.action,
                    other.action
                )
            )
        return self.__class__(
            self.before + other.before,
            self.after + other.after,
            self.action
        )


class DecisionProcessTrajectory(object):
    def __init__(self, observations, actions, rewards):
        self.observations = np.array(observations)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards)

    def transitions(self):
        l = {}
        for i in range(len(self.actions)):
            a = self.actions[i]
            if not(a in l): l[a] = {'before': [], 'after': []}
            l[a]['before'].append(self.observations[i])
            l[a]['after'].append(self.observations[i + 1])
        t = []
        for a in sorted(l.keys()):
            l[a]['before'].append(l[a]['after'][-1])
            l[a]['after'].append(l[a]['after'][-1])
            t.append(
                DecisionProcessTransitions(
                    l[a]['before'],
                    l[a]['after'],
                    action=a
                )
            )
        return t
