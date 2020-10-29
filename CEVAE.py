import os
import math, random
import scipy.stats as stats
import numpy as np
import pandas as pd
import collections
import itertools

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        layers.append(nn.Linear(sizes[0],sizes[1]))
        for in_size, out_size in zip(sizes[1:], sizes[2:]):
            layers.append(nn.ELU())
            layers.append(nn.Linear(in_size, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.length = len(layers)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)
        
    def __len__(self):
        return self.length
        
class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        t_mode,
        y_mode,
        x_mode,
        y_separate_enc
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.t_mode = t_mode
        self.y_mode = y_mode
        self.p_y_zt_nn_layers = p_y_zt_nn_layers
        self.p_y_zt_nn_width = p_y_zt_nn_width
        self.p_t_z_nn_layers = p_t_z_nn_layers
        self.p_t_z_nn_width = p_t_z_nn_width
        self.p_x_z_nn_layers = p_x_z_nn_layers
        self.p_x_z_nn_width = p_x_z_nn_width
        self.y_separate_enc = y_separate_enc
        
        #Can be used as a linear predictor if num_hidden=0
        self.n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])
        #for each x we have the possible std estimator also for simplicity, possibly not used
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [(self.n_x_estimands)*2])
        #These use as many heads as needed
        self.t_heads = 2 if self.t_mode==0 else (1 if self.t_mode==2 else self.t_mode)
        self.t_nn = FullyConnected([z_dim] + p_t_z_nn_layers*[p_t_z_nn_width] + [self.t_heads])
        self.y_heads = 2 if self.y_mode==0 else (1 if self.y_mode==2 else self.y_mode)
        self.y_nn = FullyConnected([z_dim+1] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        self.y0_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        self.y1_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        
        self.to(device)
        
    def forward(self, z, t):
        
        x_res = self.x_nn(z)
        x_pred = x_res[:,:self.n_x_estimands]
        x_std = torch.exp(x_res[:,self.n_x_estimands:])
        
        t_res = self.t_nn(z)
        if self.t_mode == 0:
            t_pred = t_res[:,:1]
            t_std = torch.exp(t_res[:,1:])
        else:
            t_pred = t_res
            t_std = None
        
        if self.y_separate_enc and self.t_mode==2:
            y_res0 = self.y0_nn(z)
            y_res1 = self.y1_nn(z)
            if self.y_mode==0:
                y_pred0 = y_res0[:,:1]
                y_std0 = torch.exp(y_res0[:,1:])
                y_pred1 = y_res1[:,:1]
                y_std1 = torch.exp(y_res1[:,1:])
            else:
                y_pred0 = y_res0
                y_std0 = None
                y_pred1 = y_res1
                y_std1 = None
            y_pred = y_pred1*t + y_pred0*(1-t)
            y_std = y_std1*t + y_std0*(1-t)
        else:
            y_res = self.y_nn(torch.cat([z,t],1))
            if self.y_mode == 0:
                y_pred = y_res[:,:1]
                y_std = torch.exp(y_res[:,1:])
            else:
                y_pred = y_res
                y_std = None
        
        return x_pred,x_std,t_pred,t_std,y_pred,y_std

class Encoder(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        t_mode,
        y_mode,
        q_z_nn_layers,
        q_z_nn_width,
        ty_separate_enc
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.q_z_nn_layers = q_z_nn_layers
        self.q_z_nn_width = q_z_nn_width
        self.t_mode=t_mode
        self.y_mode=y_mode
        self.ty_separate_enc=ty_separate_enc
        
        # q(z|x,t,y)
        self.q_z_nn = FullyConnected([x_dim+2] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        # In case t and y are binary: (tries to fit a gaussian in a simple way, 
        # but a Gaussian posterior is wrong in the first place)
        # Let's try to fit the stds for different inputs also, seems reasonable
        self.q_z_nn_t0y0 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t0y1 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t1y0 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t1y1 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        self.to(device)
        
    def forward(self, x, t, y):
        if self.ty_separate_enc and self.t_mode == 2 and self.y_mode == 2:
            z_res = self.q_z_nn_t0y0(x)*(1-t)*(1-y) + self.q_z_nn_t0y1(x)*(1-t)*(y) + \
                self.q_z_nn_t1y0(x)*(t)*(1-y) + self.q_z_nn_t1y1(x)*(t)*(y)
            z_pred = z_res[:,:self.z_dim]
            z_std = torch.exp(z_res[:,self.z_dim:])
        else:
            z_res = self.q_z_nn(torch.cat([x, t, y], axis=1))
            z_pred = z_res[:,:self.z_dim]
            z_std = torch.exp(z_res[:,self.z_dim:])
        return z_pred, z_std
        
class CEVAE(nn.Module):
    #The CEVAE used for real data
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        q_z_nn_layers,
        q_z_nn_width,
        t_mode,
        y_mode,#0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
        x_mode,#a list, 0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
        ty_separate_enc
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.y_mode = y_mode
        self.t_mode = t_mode
        self.x_mode = x_mode
        
        assert t_mode == 0 or t_mode > 1
        assert y_mode == 0 or y_mode > 1
        assert all([x_m == 0 or x_m > 1 for x_m in x_mode])
        assert len(x_mode) == x_dim
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device,
            t_mode,
            y_mode,
            q_z_nn_layers,
            q_z_nn_width,
            ty_separate_enc
        )
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device,
            p_y_zt_nn_layers,
            p_y_zt_nn_width,
            p_t_z_nn_layers,
            p_t_z_nn_width,
            p_x_z_nn_layers,
            p_x_z_nn_width,
            t_mode,
            y_mode,
            x_mode,
            ty_separate_enc#Toggle on/off with the encoder for now
        )
        self.to(device)
        self.float()

    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, t, y):#Should have t, y
        z_mean, z_std = self.encoder(x, t, y)
        #TODO: works at least for z_dim=1, maybe errors if z_dim>1
        z = self.reparameterize(z_mean, z_std)
        x_pred, x_std, t_pred, t_std, y_pred, y_std = self.decoder(z,t)
        
        return z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std