"""Baseline Models for TAN Comparison
"""
import numpy as np
import torch
import torch.nn as nn

class AdaptiveLIFNeuron:
    """LIF with slow threshold adaptation. Threshold rises after each spike,
    slowly decays back to baseline. This produces habituation.
    """
    def __init__(self, lambda_leak=0.5, threshold0=2.0,
                 alpha=0.3,         # threshold increment per spike
                 rho=0.05):         # threshold recovery rate (per step)
        self.lambda_leak = lambda_leak
        self.threshold0 = threshold0
        self.threshold = threshold0
        self.alpha = alpha
        self.rho = rho
        self.h = 0.0

    def forward(self, x):
        self.h = self.lambda_leak * self.h + x
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
            # Threshold increments after firing
            self.threshold = self.threshold + self.alpha
        else:
            spike = 0
        # Threshold recovery toward baseline
        self.threshold = (1 - self.rho) * self.threshold + self.rho * self.threshold0
        return spike

    def reset(self):
        self.h = 0.0
        self.threshold = self.threshold0






