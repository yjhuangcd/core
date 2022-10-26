
from .controller import Controller
import torch as th
import torch.nn as nn

class LinearController(Controller):
    """Class for neural network policies."""

    def __init__(self, dynamics, K):
        """Create a LinearController object.

        Policy is u = -K * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        """

        Controller.__init__(self, dynamics)
        self.model = nn.Linear(K.shape[1], K.shape[0], bias=False)
        self.model.weight.data = K

    def forward(self, x, t):
        return - self.model(x)
