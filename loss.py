import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import torch.nn as nn
import random
import h5py
import os
import torch.utils.data

class Data(nn.Module):
    def __init__(self,f_neur):
        super().__init__()
        self.fun=f_neur

    def forward(self,z,z_true):
        layer = nn.AvgPool2d(32, stride=32)
        z = layer(z)
        output = torch.sum(torch.square(z_true-z))

        return output


class Eq(nn.Module):
    def __init__(self,f_neur,mui):
        super().__init__()
        self.fun = f_neur
        self.mui = mui

    def forward(self,x):
        A=self.fun(x)
        ul = torch.cat((A[:, 0, :, :1], A[:, 0, :, :-1]), 3)
        ur = torch.cat((A[:, 0, :, 1:], A[:, 0, :, -1:]), 3)
        ut = torch.cat((A[:, 0, :, :1], A[:, 0, :-1, :]), 2)
        ub = torch.cat((A[:, 0, 1:, :], A[:, 0, -1:, :]), 2)

        vl = torch.cat((A[:, 1, :, :1], A[:, 1, :, :-1]), 3)
        vr = torch.cat((A[:, 1, :, 1:], A[:, 1, :, -1:]), 3)
        vt = torch.cat((A[:, 1, :, :1], A[:, 1, :-1, :]), 2)
        vb = torch.cat((A[:, 1, 1:, :], A[:, 1, -1:, :]), 2)

        u_x = (ut - ub) * 0.5 / dx
        u_y = (ur - ul) * 0.5 / dy
        u_xx = (ut - ub) / dx ** 2
        u_yy = (ur - ul) / dy ** 2

        v_x = (vt - vb) * 0.5 / dx
        v_y = (vr - vl) * 0.5 / dy
        v_xx = (vt - vb) / dx ** 2
        v_yy = (vr - vl) / dy ** 2



