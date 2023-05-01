
from .controller import Controller
import torch as th
import torch.nn as nn
import math

class LinearController(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, in_channels=2, out_channels=1, K=None):
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
                    m.bias.data.zero_()

    def forward(self, x, t):
        return - self.model(x)
