import numpy as np


class Linearexp(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps
        self.epsilons = []
    
    def update(self, t):
        if t < self.nsteps:
            self.epsilon = ((self.eps_end - self.eps_begin) / self.nsteps) * t + self.eps_begin
        else:
            self.epsilon = self.eps_end
        self.epsilons.append(self.epsilon)
        return