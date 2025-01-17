from .controller import Controller
import torch as th
import torch.nn as nn
import math

class NNController(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, in_channels, out_channels, hidden_channels):
        """Create a Neural Network controller object.

        Inputs:
        Dynamics, dynamics: Dynamics
        """

        Controller.__init__(self, dynamics)
        self.in_channels = in_channels
        self.act = nn.ReLU()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, out_channels)
        )
        # barrier strength
        self.alpha = 1.
        self.highest_co2 = 1.

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(6. / n))
                m.bias.data.zero_()

    def forward(self, x, t):
        # x contains auxiliary states
        x_main = x[:, 0:self.in_channels]
        # d needs to be positive because it's number of people
        d = (x[:, 3:4] + 1) / 2
        # output controls need to be positive and constrained
        u = 150 * th.sigmoid(self.model(x_main))
        # barrier
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        # denominator can't be negative for autolirpa
        thres = - (self.alpha * (self.highest_co2 - x[:, 1:2]) - self.dynamics.g2 * d) / (self.act(self.dynamics.beta_2 * (x[:, 1:2] - self.dynamics.c_in)) + 1e-12)
        u2 = self.act(u2 - thres) + thres
        u = th.concat((u1, u2), dim=1)
        return u
