# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 23:22:40 2023

@author: shiying xiong
"""

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import loss
import gc
import h5py

device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
igst = 10
dt = 0.3




grid_sizex = 512
u_sizex = grid_sizex + 2 * igst
xs = -torch.pi
xe = torch.pi
lx = xe - xs
dx = lx / grid_sizex
exs = xs - igst * dx
exe = xe + igst * dx



grid_sizey = 512
u_sizey = grid_sizey + 2 * igst
ys = -torch.pi
ye = torch.pi
ly = ye - ys
dy = ly / grid_sizey
eys = ys - igst * dy
eye = ye + igst * dy


def cal_boundry(z, igst):
    z = torch.cat((z[:, :, -2*igst:-igst, :], z[:, :, igst: -igst, :], z[:, :, igst:2*igst, :]), dim=2)
    z = torch.cat((z[:, :, :, -2*igst:-igst], z[:, :, :, igst: -igst], z[:, :, :, igst:2*igst]), dim=3)
    return z

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ODEnet(nn.Module):
    def __init__(self, u_sizex, u_sizey, dt, igst):
        super(ODEnet, self).__init__()
        self.igst = igst
        self.u_sizex = u_sizex
        self.u_sizey = u_sizey
        self.dt = dt
        

        self.cal_velocity_from_vorticity = nn.Sequential(ResidualBlock(1, 16),
                                      ResidualBlock(16, 16),
                                      ResidualBlock(16, 32),
                                      ResidualBlock(32, 64),
                                      ResidualBlock(64, 64),
                                      ResidualBlock(64, 64, kernel_size=1),
                                      nn.Conv2d(64, 2, kernel_size=1, padding=0),
                                      )
        self.advection_vorticity = nn.Sequential(ResidualBlock(3, 16),
                                      ResidualBlock(16, 16),
                                      ResidualBlock(16, 32),
                                      ResidualBlock(32, 64),
                                      ResidualBlock(64, 64),
                                      ResidualBlock(64, 64, kernel_size=1),
                                      nn.Conv2d(64, 1, kernel_size=1, padding=0),
                                      )
        self.cal_pressure_from_velocity = nn.Sequential(ResidualBlock(2, 16),
                                   ResidualBlock(16, 16),
                                   ResidualBlock(16, 32),
                                   ResidualBlock(32, 64),
                                   ResidualBlock(64, 64),
                                   ResidualBlock(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 1, kernel_size=1, padding=0),
                                   )        

    def forward(self, n_steps):
        w = nn.Parameter(torch.randn(1, 1, self.u_sizex, self.u_sizey), requires_grad=True).cuda()
        w = cal_boundry(w, self.igst)
        for i_step in range(int(n_steps)):
            u = self.cal_velocity_from_vorticity(w) ###改成数值算法，- naplacian u = curl omega
            u = cal_boundry(u, self.igst)
            fluid = torch.cat((u, w), 1)
            w = self.advection_vorticity(fluid) ###需要用网络学局部运动
            w = cal_boundry(w, self.igst)
        u = self.cal_velocity_from_vorticity(w)
        p = self.cal_pressure_from_velocity(u)  ###改成数值算法
        z = torch.cat((u, p), 1)
        z = cal_boundry(z, self.igst)
        return z

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self,dt):
#         f = h5py.File('data_16.h5',mode="r")
#         self.dt=dt
def dataloder(dt,t):
    f = h5py.File('data_16.h5', mode="r")
    dt = dt
    t = t
    name = '00{}'.format(dt*t)
    data = f[name][:]
    data = data[:,2:].reshape(1,256,3)
    data = torch.as_tensor(data)
    data = data.permute(2,0,1)
    data = data.reshape(3,16,16)
    f.close()
    return data




#生成网络
# f_cal = ODEnet(u_sizex,u_sizey,dt,igst)
# f_cal.to(device)
# #生成loss函数
# # Eqloss=loss.Eq(f_cal)
# Dataloss=loss.Data(f_cal)

def train():
    n_epoch=1000
    n_step = 101
    optimizer = torch.optim.Adam(f_cal.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)
    # train_data_loader = torch.utils.data.Dataloader(data, batch_size=1)

    # train_data_loader.requires_grad = True

    for i in range(n_epoch):
        loss_eq = loss_data = 0.0
        optimizer.zero_grad()
        f_cal.train()
        # for batch_id , data_batch in enumerate(train_data_loader):
        for t in range(n_step):
            z = f_cal(t)
            z_true = dataloder(dt,t).cuda()
            loss_data=Dataloss(z,z_true)



        # losseq = Eqloss(z)
        # loss_eq.backward()
        loss_data.backward()

        optimizer.step()
        optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

        if i % 1 == 0:
            print('epoch:',i,'loss_eq:',losseq.cpu().detach().numpy(),'loss_data:',lossdata.cpu().detach().numpy())

        if i % 200 ==0 :
            torch.save(f_cal.state_dict(), 'E:/save/f_cal/' + 'net_params_' + str(i))
            torch.save(optimizer.state_dict(), 'E:/save/f_cal/' + 'opt_params_' + str(i))

f_cal = ODEnet(u_sizex,u_sizey,dt,igst)
f_cal.to(device)
#生成loss函数
# Eqloss=loss.Eq(f_cal)
Dataloss=loss.Data(f_cal)

train()