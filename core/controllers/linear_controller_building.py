
from .controller import Controller
import torch as th
import torch.nn as nn
import math

class LinearController(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, in_channels=2, out_channels=2, K=None):
        """Create a LinearController object.

        Policy is u = -K * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        """

        Controller.__init__(self, dynamics)
        if K is not None:
            self.model = nn.Linear(K.shape[1], K.shape[0], bias=False)
            self.model.weight.data = K
        else:
            self.model = nn.Linear(in_channels, out_channels, bias=False)
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    n = m.in_features
                    m.weight.data.normal_(0, math.sqrt(6. / n))
        self.in_channels = in_channels
        self.act = nn.ReLU()
        # barrier strength
        self.alpha = 1.
        self.highest_co2 = 0.5

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
