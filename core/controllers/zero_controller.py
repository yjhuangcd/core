from .controller import Controller
import torch as th
import torch.nn as nn
import math

class ZeroController(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, out_channels):
        """Create a Neural Network controller object.

        Inputs:
        Dynamics, dynamics: Dynamics
        """

        Controller.__init__(self, dynamics)
        self.out_channels = out_channels

    def forward(self, x, t):
        # return zeros for no control comparison
        return th.zeros((x.shape[0], self.out_channels))
