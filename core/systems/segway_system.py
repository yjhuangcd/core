from torch import cat, cos, diag, float64, norm, sin, stack, tensor, zeros, set_default_dtype
from torch import atan2
from torch.nn import Module, Parameter
import torch as th

from core.dynamics import SystemDynamics, AffineDynamics
from core.util import  default_fig
from numpy import pi
# set_default_dtype(float64)

# Segway Dynamics from here:
# https://github.com/urosolia/MultiRate/blob/master/python/simulator.py

class Segway(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 3, 1)
        Module.__init__(self)

    def forward(self, x, u=0., t=0.):
        x_dot = x[:, 1:2]
        y_dot = x[:, 2:3]
        y = x[:, 0:1]
        return th.concat([
            y_dot,
            (cos(y) * (-1.8 * u + 11.5 * x_dot + 9.8 * sin(
                y)) - 10.9 * u + 68.4 * x_dot - 1.2 * y_dot * y_dot * sin(y)) / (cos(y) - 24.7),
            ((9.3 * u - 58.8 * x_dot) * cos(y) + 38.6 * u - 234.5 * x_dot - sin(
                y) * (208.3 + y_dot * y_dot * cos(y))) / (cos(y) * cos(y) - 24.7)
        ], dim=1)
