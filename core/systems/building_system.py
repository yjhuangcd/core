from torch import cat, cos, diag, float64, norm, sin, stack, tensor, zeros, set_default_dtype
from torch import atan2
from torch.nn import Module, Parameter
import torch as th

from core.dynamics import SystemDynamics, AffineDynamics
from core.util import  default_fig
from numpy import pi
# set_default_dtype(float64)

class Building(SystemDynamics, AffineDynamics, Module):
    def __init__(self, num_states=4, num_actions=2):
        # for temperature and Co2 control, 4 states and 2 control variables
        SystemDynamics.__init__(self, num_states, num_actions)
        Module.__init__(self)
        # # temp, co2, Tamb, d
        # means = th.tensor([70., 600., 70., 50.]).reshape(1, 1, 4)
        # ranges = th.tensor([10., 200., 10., 50.]).reshape(1, 1, 4)
        self.beta_1_prime = 1 / 133.
        self.beta_1 = self.beta_1_prime * 10
        self.beta_2 = self.beta_1
        self.c_in = (400. - 600.) / 200.
        self.g1 = 0.0005
        self.g2 = 0.02 * 1000

    def forward(self, x, u=0., t=0.):
        # states: temperature, co2, T_amb, d
        temp = x[:, 0:1]
        co2 = x[:, 1:2]
        T_amb = x[:, 2:3]
        # d needs to be positive because it's number of people
        d = (x[:, 3:4] + 1) / 2
        u1 = u[:, 0:1]
        u2 = u[:, 1:2]
        zeros = th.zeros_like(temp)
        temp_dot = - self.beta_1 * temp + self.beta_1_prime * u1 - self.beta_1_prime * temp * u2 + self.beta_1 * T_amb + self.g1 * d
        co2_dot = self.beta_2 * (self.c_in - co2) * u2 + self.g2 * d
        return th.concat([temp_dot, co2_dot, zeros, zeros], dim=1)
