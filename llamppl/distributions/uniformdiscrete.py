from .distribution import Distribution
import numpy as np

class UniformDiscrete(Distribution):
    def __init__(self, N):
        self.N = N
        self.probs = [1/self.N] * self.N

    def sample(self):
        x = np.random.choice(self.N, p=self.probs)
        return x, self.log_prob(x)

    def argmax(self, idx):
        return np.argsort(self.probs)[-idx]

    def log_prob(self, value):
        return np.log(1/self.N)