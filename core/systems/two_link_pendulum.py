from torch import cos, sin
from torch import atan2
from torch.nn import Module, Parameter
import torch as th
import numpy as np

from core.dynamics import SystemDynamics, AffineDynamics
from core.util import  default_fig
from numpy import pi
# set_default_dtype(float64)


class TwoLinkPendulum(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 4, 2)
        Module.__init__(self)
        self.m1 = 0.5
        self.m2 = 0.5
        self.l1 = 0.5
        self.l2 = 0.5
        self._compute_state_space()

    def _compute_state_space(self):
        self.a11 = self.l1 * self.l1 * (self.m1 + self.m2)
        self.a12 = self.l1 * self.l2 * self.m2
        self.a22 = self.l2 * self.l2 * self.m2
        self.b1 = self.l1 * (self.m1 + self.m2) * 9.8
        self.b2 = self.l2 * self.m2 * 9.8

    def forward(self, x, u=0., t=0.):
        theta_1 = x[:, 0:1]
        theta_2 = x[:, 1:2]
        theta_1_dot = x[:, 2:3]
        theta_2_dot = x[:, 3:4]
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        c12 = - self.a12 * theta_2_dot * sin(theta_2 - theta_1)
        c21 = - self.a12 * theta_1_dot * sin(theta_1 - theta_2)

        denominator = self.a11 * self.a22 - (self.a12 * cos(theta_1 - theta_2)) ** 2
        theta_1_double_dot = 1. / denominator * (self.a22 * (u1 + self.b1 * sin(theta_1) - c12 * theta_2_dot) -
                                                 self.a12 * cos(theta_1 - theta_2) * (u2 + self.b2 * sin(theta_2) - c21 * theta_1_dot))
        theta_2_double_dot = 1. / denominator * (- self.a12 * cos(theta_1 - theta_2) * (u1 + self.b1 * sin(theta_1) - c12 * theta_2_dot) +
                                                 self.a11 * (u2 + self.b2 * sin(theta_2) - c21 * theta_1_dot))
        return th.concat([theta_1_dot, theta_2_dot, theta_1_double_dot, theta_2_double_dot], dim=1)

    def change_params(self, params):
        self.m1 = params[0]
        self.m2 = params[1]
        self.l1 = params[2]
        self.l2 = params[3]
        self._compute_state_space()
        return


class TwoLinkPendulum_params(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 4, 2)
        Module.__init__(self)

    def _compute_state_space(self, m1, m2, l1, l2):
        a11 = l1 * l1 * (m1 + m2)
        a12 = l1 * l2 * m2
        a22 = l2 * l2 * m2
        b1 = l1 * (m1 + m2) * 9.8
        b2 = l2 * m2 * 9.8
        return a11, a12, a22, b1, b2

    def forward(self, x, u=0., t=0.):
        theta_1 = x[:, 0:1]
        theta_2 = x[:, 1:2]
        theta_1_dot = x[:, 2:3]
        theta_2_dot = x[:, 3:4]
        # read parameters
        m1 = x[:, 4:5]
        m2 = x[:, 5:6]
        l1 = x[:, 6:7]
        l2 = x[:, 7:8]
        a11, a12, a22, b1, b2 = self._compute_state_space(m1, m2, l1, l2)
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        c12 = - a12 * theta_2_dot * sin(theta_2 - theta_1)
        c21 = - a12 * theta_1_dot * sin(theta_1 - theta_2)

        denominator = a11 * a22 - (a12 * cos(theta_1 - theta_2)) ** 2
        theta_1_double_dot = 1. / denominator * (a22 * (u1 + b1 * sin(theta_1) - c12 * theta_2_dot) -
                                                 a12 * cos(theta_1 - theta_2) * (u2 + b2 * sin(theta_2) - c21 * theta_1_dot))
        theta_2_double_dot = 1. / denominator * (- a12 * cos(theta_1 - theta_2) * (u1 + b1 * sin(theta_1) - c12 * theta_2_dot) +
                                                 a11 * (u2 + b2 * sin(theta_2) - c21 * theta_1_dot))
        return th.concat([theta_1_dot, theta_2_dot, theta_1_double_dot, theta_2_double_dot], dim=1)
