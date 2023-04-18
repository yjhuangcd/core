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
            # nn.Linear(hidden_channels, hidden_channels),
            # self.act,
            nn.Linear(hidden_channels, out_channels)
        )
        # need to be consistent with building system
        self.beta_1_prime = 1 / 133.
        self.beta_1 = self.beta_1_prime * 10
        self.beta_2 = self.beta_1
        self.c_in = (400. - 600.) / 200.
        self.g1 = 0.0005
        self.g2 = 0.02 * 1000
        # barrier strength
        self.alpha = 1.
        self.highest_co2 = 0.5

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
        thres = (self.alpha * (self.highest_co2 - x[:, 1:2]) - self.g2 * d) / (self.beta_2 * (self.c_in - x[:, 1:2]) + 1e-12)
        u2 = self.act(u2 - thres) + thres
        u = th.concat((u1, u2), dim=1)
        return u
