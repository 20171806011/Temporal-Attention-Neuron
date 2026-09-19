"""LIF Neuron Model
"""
import numpy as np


class LIFNeuron:
    def __init__(self, lambda_leak=0.5, threshold=2.0):
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0

    def forward(self, current_input):
        self.h = self.lambda_leak * self.h + current_input
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


