from torch import cos, sin
from torch import atan2
from torch.nn import Module, Parameter
import torch as th
import numpy as np

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
        self.c1 = 1.8
        self.c2 = 11.5
        self.c3 = 10.9
        self.c4 = 68.4
        self.c5 = 1.2
        self.d1 = 9.3
        self.d2 = 58.8
        self.d3 = 38.6
        self.d4 = 234.5  # todo: typo, in the paper is 243.5
        self.d5 = 208.3
        self.b = 24.7

    def forward(self, x, u=0., t=0.):
        x_dot = x[:, 1:2]
        y_dot = x[:, 2:3]
        y = x[:, 0:1]
        return th.concat([
            y_dot,
            (cos(y) * (-self.c1 * u + self.c2 * x_dot + 9.8 * sin(
                y)) - self.c3 * u + self.c4 * x_dot - self.c5 * y_dot * y_dot * sin(y)) / (cos(y) - self.b),
            ((self.d1 * u - self.d2 * x_dot) * cos(y) + self.d3 * u - self.d4 * x_dot - sin(
                y) * (self.d5 + y_dot * y_dot * cos(y))) / (cos(y) * cos(y) - self.b)
        ], dim=1)

    def change_params(self, params):
        self.c1 = params[0]
        self.c2 = params[1]
        self.c3 = params[2]
        self.c4 = params[3]
        self.c5 = params[4]
        self.d1 = params[5]
        self.d2 = params[6]
        self.d3 = params[7]
        self.d4 = params[8]  # todo: typo, in the paper is 243.5
        self.d5 = params[9]
        self.b = params[10]
        return


class Segway_params(SystemDynamics, AffineDynamics, Module):
    # takes in system parameters as inputs
    def __init__(self):
        SystemDynamics.__init__(self, 3, 1)
        Module.__init__(self)

    def forward(self, x, u=0., t=0.):
        x_dot = x[:, 1:2]
        y_dot = x[:, 2:3]
        y = x[:, 0:1]
        # extract system params from x
        c1 = x[:, 3:4]  # 1.8
        c2 = x[:, 4:5]  # 11.5
        c3 = x[:, 5:6]  # 10.9
        c4 = x[:, 6:7]  # 68.4
        c5 = x[:, 7:8]  # 1.2
        d1 = x[:, 8:9]  # 9.3
        d2 = x[:, 9:10]  # 58.8
        d3 = x[:, 10:11]  # 38.6
        d4 = x[:, 11:12]  # 234.5  # todo: typo, in the paper is 243.5
        d5 = x[:, 12:13]  # 208.3
        b = x[:, 13:14]  # 24.7
        return th.concat([
            y_dot,
            (cos(y) * (-c1 * u + c2 * x_dot + 9.8 * sin(
                y)) - c3 * u + c4 * x_dot - c5 * y_dot * y_dot * sin(y)) / (cos(y) - b),
            ((d1 * u - d2 * x_dot) * cos(y) + d3 * u - d4 * x_dot - sin(
                y) * (d5 + y_dot * y_dot * cos(y))) / (cos(y) * cos(y) - b)
        ], dim=1)

class SegwayTrack(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 3, 1)
        Module.__init__(self)
        self.k = 1.
        self.A = 0.1
        self.ref = lambda t: self.A * self.k * cos(self.k * t)
        self.ref_dot = lambda t: - self.A * self.k ** 2 * sin(self.k * t)

    def forward(self, x, u=0., t=0.):
        # states: phi, eps = v - ref, phi_dot
        phi = x[:, 0:1]
        eps = x[:, 1:2]
        v = eps + self.ref(t)
        phi_dot = x[:, 2:3]
        return th.concat([
            phi_dot,
            (cos(phi) * (-1.8 * u + 11.5 * v + 9.8 * sin(
                phi)) - 10.9 * u + 68.4 * v - 1.2 * phi_dot * phi_dot * sin(phi)) / (cos(phi) - 24.7) - self.ref_dot(t),
            ((9.3 * u - 58.8 * v) * cos(phi) + 38.6 * u - 234.5 * v - sin(
                phi) * (208.3 + phi_dot * phi_dot * cos(phi))) / (cos(phi) * cos(phi) - 24.7)
        ], dim=1)
