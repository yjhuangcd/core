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

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(6. / n))
                m.bias.data.zero_()

    def forward(self, x, t):
        return self.model(x)


class NNController_sim(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, in_channels, out_channels, hidden_channels):
        """Create a Neural Network controller object.

        Inputs:
        Dynamics, dynamics: Dynamics
        """

        Controller.__init__(self, dynamics)
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
        self.in_channels = in_channels

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(6. / n))
                m.bias.data.zero_()

    def forward(self, x, t):
        x = x[:, :self.in_channels]
        return self.model(x)
