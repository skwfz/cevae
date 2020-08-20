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
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

class Decoder(nn.Module):
    def __init__(
        self, 
        input_dim, #x dim
        hidden_dim, 
        num_hidden,#hidden layers for the y prediction and x prediction
        activation=nn.ELU(), 
        z_dim=1, 
        device='cpu',
        binary_t_y = False,
        p_y_zt_nn = False,
        p_x_z_nn = False,
        p_t_z_nn = False,
        p_y_zt_std = False,#whether we want to estimate the variance separately for each data unit. Used only if we have full NN
        p_x_z_std = False,
        p_t_z_std = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device
        self.binary_t_y = binary_t_y
        self.p_y_zt_nn = p_y_zt_nn
        self.p_x_z_nn = p_x_z_nn
        self.p_t_z_nn = p_t_z_nn
        self.p_y_zt_std = p_y_zt_std
        self.p_x_z_std = p_x_z_std
        self.p_t_z_std = p_t_z_std
        
        #NOTE: Bias not required for linear models if data is from linear SCM
        #A linear regression for each x. Assumes that z_dim = 1.
        self.x_nns = nn.ModuleList([nn.Linear(z_dim,1, bias=False) for i in range(input_dim)])
        self.x_nns_real = FullyConnected([z_dim] + num_hidden*[hidden_dim] + [input_dim*2 if p_x_z_std else input_dim])

        # p(t|z)
        self.t_nn = nn.Linear(z_dim,1, bias=True if binary_t_y else False)
        self.t_nn_real = FullyConnected([z_dim] + num_hidden*[hidden_dim] + [2 if p_t_z_std else 1])

        # p(y|t,z)
        self.y_nn = nn.Linear(z_dim + 1,1, bias=False)
        self.y0_nn = nn.Linear(z_dim,1, bias=True)#If t and y binary
        self.y1_nn = nn.Linear(z_dim,1, bias=True)
        self.y_nn_real = FullyConnected([z_dim+1] + num_hidden*[hidden_dim] + [2 if p_y_zt_std else 1])#real neural networks
        self.y0_nn_real = FullyConnected([z_dim] + num_hidden*[hidden_dim] + [1])
        self.y1_nn_real = FullyConnected([z_dim] + num_hidden*[hidden_dim] + [1])
        
        #Variance parameters
        self.x_log_std = nn.Parameter(torch.FloatTensor(input_dim*[1.], device=device))
        self.t_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        self.y_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        
        self.to(device)

    def forward(self, z, t):
        bs = z.shape[0]
        
        t_std,y_std=None,None#In case these are not estimated at all
        
        if self.p_x_z_nn:
            if self.p_x_z_std:
                x_res = self.x_nns_real(z)
                x_pred = x_res[:,:self.input_dim].reshape((bs,self.input_dim))
                x_std = torch.exp(x_res[:,self.input_dim:]).reshape((bs,self.input_dim))
            else:
                x_pred = self.x_nns_real(z)
                x_std = torch.exp(self.x_log_std).repeat((bs,1))
        else:
            x_pred = torch.zeros(bs, self.input_dim, device=self.device)
            for i in range(self.input_dim):
                x_pred[:,i] = self.x_nns[i](z)[:,0]
            x_std = torch.exp(self.x_log_std).repeat((bs,1))
        
        #We need to have the observed t here because that ends up in the likelihood function of y through the p(y|z,t)
        #neural network.
        if self.binary_t_y:
            if self.p_t_z_nn:
                t_pred = self.t_nn_real(z)
            else:
                t_pred = self.t_nn(z)
            if self.p_y_zt_nn:
                y_logits0 = self.y0_nn_real(z)
                y_logits1 = self.y1_nn_real(z)
            else:
                y_logits0 = self.y0_nn(z)
                y_logits1 = self.y1_nn(z)
            y_pred = y_logits1*t + y_logits0*(1-t)
        else:
            if self.p_t_z_nn:
                if self.p_t_z_std:
                    t_res = self.t_nn_real(z)
                    t_pred = t_res[:,0].reshape((bs,1))
                    t_std = torch.exp(t_res[:,1]).reshape((bs,1))
                else:
                    t_pred = self.t_nn_real(z)
                    t_std = torch.exp(self.t_log_std).repeat((bs, 1))
            else:
                t_pred = self.t_nn(z)
                t_std = torch.exp(self.t_log_std).repeat((bs, 1))
            if self.p_y_zt_nn:
                if self.p_y_zt_std:
                    y_res = self.y_nn_real(torch.cat([z,t],axis=1))
                    y_pred = y_res[:,0].reshape((bs,1))
                    y_std = torch.exp(y_res[:,1]).reshape((bs,1))
                else:
                    y_pred = self.y_nn_real(torch.cat([z,t],axis=1))
                    y_std = torch.exp(self.y_log_std).repeat((bs,1))
            else:
                y_pred = self.y_nn(torch.cat([z,t],axis=1))
                y_std = torch.exp(self.y_log_std).repeat((bs,1))

        return (#t_pred and y_pred can be interpreted as logits or actual predictions
            x_pred, t_pred, y_pred, x_std, t_std, y_std
        )
    
    def sample(self,size):
        #Create a sample (z,x,t,y) according to the learned joint distribution
        # TODO: outdated
        z_sample = torch.randn((size,self.z_dim))
        print(z_sample.size)
        x_sample = torch.zeros(size, self.input_dim, device=self.device)
        for i in range(self.input_dim):
            x_sample[:,i] = self.x_nns[i](z_sample)[:,0] + \
                torch.randn(size)*torch.exp(self.x_log_std[i])
        
        if self.binary_t_y:
            t_logits = self.treatment_pred(z_sample)
            t_sample = dist.Bernoulli(torch.sigmoid(t_logits)).sample()
            y_logits0 = self.y0_nn(z_sample)
            y_logits1 = self.y1_nn(z_sample)
            y_logits = y_logits1*t_sample + y_logits0*(1-t_sample)
            y_sample = dist.Bernoulli(torch.sigmoid(y_logits)).sample()
        else:
            t_sample = self.t_nn(z_sample) + torch.randn((size,1))*torch.exp(self.t_log_std)
            y_sample = self.y_nn(torch.cat([z_sample,t_sample],axis=1)) + torch.randn((size,1))*torch.exp(self.y_log_std)
        
        return z_sample, x_sample, t_sample, y_sample

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim,
        num_hidden,
        activation=nn.ELU(),
        z_dim=1,
        device='cpu',
        binary_t_y = False,
        q_z_xty_nn = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device
        self.binary_t_y = binary_t_y
        self.q_z_xty_nn = q_z_xty_nn
        
        # q(z|x,t,y)
        # when all variables are gaussian, uses the fact that conditional distributions of gaussians are
        # gaussians with linear dependencies, 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        self.q_z_nn = nn.Linear(input_dim+2,z_dim)
        # This is OK at least if z_dim=1, not sure if full covariance matrix can be estimated like this very well
        # self.z_std = nn.Parameter(torch.ones((z_dim, z_dim), device=device))
        # Makes the assumption that z in z|x,t,y are independent, not necessarily right if *true* z_dim > 1
        self.z_logstd = nn.Parameter(torch.ones(z_dim, device=device))
        
        self.q_z_nn_real = FullyConnected([input_dim+2] + num_hidden*[hidden_dim] + [z_dim*2])#real neural network
        #self.z_logstd_nn = FullyConnected([input_dim+2] + num_hidden*[hidden_dim] + [z_dim])#Should this be shared with mean network?
        
        # In case t and y are binary: (tries to fit a gaussian in a simple way, 
        # but a Gaussian posterior is wrong in the first place)
        # Let's try to fit the stds for different inputs also, seems reasonable
        self.q_z_nn_t0y0 = nn.Linear(input_dim,z_dim)
        self.q_z_nn_t0y1 = nn.Linear(input_dim,z_dim)
        self.q_z_nn_t1y0 = nn.Linear(input_dim,z_dim)
        self.q_z_nn_t1y1 = nn.Linear(input_dim,z_dim)
        self.z_logstd_t0y0 = nn.Parameter(torch.ones(z_dim, device=device))
        self.z_logstd_t0y1 = nn.Parameter(torch.ones(z_dim, device=device))
        self.z_logstd_t1y0 = nn.Parameter(torch.ones(z_dim, device=device))
        self.z_logstd_t1y1 = nn.Parameter(torch.ones(z_dim, device=device))
        
        # This is not entirely similar to the CEVAE article, because we separate by y as well as t, 
        # and also we don't have a shared representation here
        self.q_z_nn_t0y0_real = FullyConnected([input_dim] + num_hidden*[hidden_dim] + [z_dim*2])
        self.q_z_nn_t0y1_real = FullyConnected([input_dim] + num_hidden*[hidden_dim] + [z_dim*2])
        self.q_z_nn_t1y0_real = FullyConnected([input_dim] + num_hidden*[hidden_dim] + [z_dim*2])
        self.q_z_nn_t1y1_real = FullyConnected([input_dim] + num_hidden*[hidden_dim] + [z_dim*2])
        
        self.to(device)

    def forward(self, x, t, y):
        # q(z|x,t,y)
        if self.binary_t_y:
            if self.q_z_xty_nn:
                z_res = self.q_z_nn_t0y0_real(x)*(1-t)*(1-y) + self.q_z_nn_t0y1_real(x)*(1-t)*(y) \
                        + self.q_z_nn_t1y0_real(x)*t*(1-y) + self.q_z_nn_t1y1_real(x)*t*y
                z_pred = z_res[:,:self.z_dim]
                z_std = torch.exp(z_res[:,self.z_dim:])
            else:
                z_pred = self.q_z_nn_t0y0(x)*(1-t)*(1-y) + self.q_z_nn_t0y1(x)*(1-t)*(y) \
                        + self.q_z_nn_t1y0(x)*t*(1-y) + self.q_z_nn_t1y1(x)*t*y
                z_std = torch.exp(self.z_logstd_t0y0*(1-t)*(1-y) + self.z_logstd_t0y1*(1-t)*y \
                        + self.z_logstd_t1y0*t*(1-y) + self.z_logstd_t1y1*t*y)
            return z_pred, z_std
        else:
            if self.q_z_xty_nn:
                z_res = self.q_z_nn_real(torch.cat([x, t, y], axis=1))
                z_pred = z_res[:,:self.z_dim]
                z_std = torch.exp(z_res[:,self.z_dim:])
                return z_pred, z_std
            else:
                z_pred = self.q_z_nn(torch.cat([x, t, y], axis=1))
                z_std = torch.exp(self.z_logstd).repeat((x.shape[0],1))
            return z_pred, z_std
        #return z_pred, torch.mm(self.z_std.T,self.z_std)

        
        
class linearCEVAE(nn.Module):
    def __init__(
        self, 
        input_dim,
        z_dim=1,
        encoder_hidden_dim=4,
        decoder_hidden_dim=3,
        encoder_num_hidden=3,
        decoder_num_hidden=3,
        activation=nn.ELU(),
        device='cpu',
        binary_t_y=False,
        p_y_zt_nn=False,#whether P(y|z,t) is a neural network
        q_z_xty_nn=False,#whether Q(z|x,t,y) is a neural network
        p_x_z_nn = False,#P(x|z) neural network?
        p_t_z_nn = False,#P(t|z) neural network?
        p_y_zt_std = False,#P(y|z,t) stds estimated for each unit separately?
        p_x_z_std = False,#P(x|z) -||-
        p_t_z_std = False#P(t|z) -||-
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device

        self.encoder = Encoder(
            input_dim, 
            encoder_hidden_dim, 
            encoder_num_hidden,
            activation,
            z_dim,
            device=device,
            binary_t_y=binary_t_y,
            q_z_xty_nn=q_z_xty_nn
        )
        self.decoder = Decoder(
            input_dim, 
            decoder_hidden_dim, 
            decoder_num_hidden,
            activation,
            z_dim,
            device=device,
            binary_t_y=binary_t_y,
            p_y_zt_nn=p_y_zt_nn,
            p_x_z_nn = p_x_z_nn,
            p_t_z_nn = p_t_z_nn,
            p_y_zt_std = p_y_zt_std,
            p_x_z_std = p_x_z_std,
            p_t_z_std = p_t_z_std
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
        
        x_pred, t_pred, y_pred, x_std, t_std, y_std = self.decoder(z, t)
        
        return z_mean, z_std, x_pred, t_pred, y_pred, x_std, t_std, y_std