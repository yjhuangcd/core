from .controller import Controller
from torch import cos, sin
import torch as th
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt


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
        # barrier strength
        self.alpha_1 = 10.
        self.alpha_2 = 10.
        self.max_phi = np.pi/4

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(6. / n))
                m.bias.data.zero_()

    def forward(self, x, t):
        phi = x[:, 0:1]
        eps = x[:, 1:2]
        v = eps + self.dynamics.ref(t)
        phi_dot = x[:, 2:3]
        # x may contain auxiliary states
        x_main = x[:, 0:self.in_channels]
        # output controls need to be positive and constrained
        u = self.model(x_main)
        # # todo: debug
        # u_old = u
        # # barrier
        # coef = (9.3 * cos(phi) + 38.6) / (cos(phi) ** 2 - 24.7)
        # bias = (- 58.8 * v * cos(phi) - 243.5 * v - sin(phi) * (208.3 + phi_dot ** 2 * cos(phi))) / (cos(phi) ** 2 - 24.7)
        # ub = ((self.alpha_1 + self.alpha_2) * phi_dot + self.alpha_1 * self.alpha_2 * (self.max_phi + phi) + bias) / (self.act(-coef) + 1e-12)
        # lb = ((self.alpha_1 + self.alpha_2) * phi_dot - self.alpha_1 * self.alpha_2 * (self.max_phi - phi) + bias) / (self.act(-coef) + 1e-12)
        # u = - self.act(ub - u) + ub
        # u = self.act(u - lb) + lb

        # plt.plot(ub[:20])
        # plt.plot(lb[:20])
        # plt.show()
        #
        # plt.plot(u_old.detach()[:20])
        # plt.plot(u.detach()[:20])
        # plt.show()
        return u
