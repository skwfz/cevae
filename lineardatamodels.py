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

class Decoder(nn.Module):
    def __init__(
        self, 
        input_dim, #x dim
        hidden_dim, 
        num_hidden,#hidden layers for the y prediction and x prediction
        activation=nn.ELU(), 
        z_dim=1, 
        device='cpu',
        binary_t_y = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = 1
        self.device = device
        self.binary_t_y = binary_t_y

        #A linear regression for each x. Assumes that z_dim = 1
        self.x_nns = nn.ModuleList([nn.Linear(1,1) for i in range(input_dim)])

        # p(t|z)
        self.treatment_pred = nn.Linear(1,1)

        # p(y|t,z)
        self.y_nn = nn.Linear(2,1)
        self.y0_nn = nn.Linear(1,1)
        self.y1_nn = nn.Linear(1,1)
        
        #Variance parameters
        self.x_log_std = nn.Parameter(torch.FloatTensor(input_dim*[1.], device=device))
        self.t_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        self.y_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        
        self.to(device)

    def forward(self, z, t):
        bs = z.shape[0]
        
        x_pred = torch.zeros(bs, self.input_dim, device=self.device)
        for i in range(self.input_dim):
            x_pred[:,i] = self.x_nns[i](z)[:,0]
        
        #We need to have the observed t here because that ends up in the likelihood function of y through the p(y|x,t)
        #neural network.
        if self.binary_t_y:
            t_pred = self.treatment_pred(z)
            y_logits0 = self.y0_nn(z)
            y_logits1 = self.y1_nn(z)
            y_pred = y_logits1*t + y_logits0*(1-t)
        else:
            t_pred = self.treatment_pred(z)
            y_pred = self.y_nn(torch.cat([z,t],axis=1))

        return (#t_pred and y_pred can be interpreted as logits or actual predictions
            x_pred, t_pred, y_pred
        )
    
    def sample(self,size):
        #Create a sample (z,x,t,y) according to the learned joint distribution
        z_sample = torch.randn((size,self.z_dim))
        x_pred = torch.zeros(size, self.input_dim, device=self.device)
        for i in range(self.input_dim):
            x_pred[:,i] = self.x_nns_binary[i](z_sample)[:,0] + \
                torch.randn((size, self.input_dim))*torch.exp(self.x_log_std[i])
        
        if binary_t_y:
            t_logits = self.treatment_pred(z_sample)
            t_sample = dist.Bernoulli(torch.sigmoid(t_logits)).sample()
            y_logits0 = self.y0_nn(z_sample)
            y_logits1 = self.y1_nn(z_sample)
            y_logits = y_logits1*t_sample + y_logits0*(1-t_sample)
            y_sample = dist.Bernoulli(torch.sigmoid(y_logits)).sample()
        else:
            t_sample = self.treatment_pred(z_sample) + torch.randn((size,1))*torch.exp(self.t_log_std)
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
        device='cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device
        
        # q(z|x,t,y)
        # when all variables are gaussian, uses the fact that conditional distributions of gaussians are
        # gaussians with linear dependencies, 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        self.q_z_nn = nn.Linear(input_dim+2,z_dim)
        # This is OK at least if z_dim=1, not sure if full covariance matrix can be estimated like this very well
        #self.z_std = nn.Parameter(torch.ones((z_dim, z_dim), device=device))
        # Makes the assumption that z in z|x,t,y are independent, not right if z_dim > 1
        self.z_logstd = nn.Parameter(torch.ones(z_dim, device=device))
        
        
        self.to(device)

    def forward(self, x, t, y):
        # q(z|x,t,y)
        z_pred = self.q_z_nn(torch.cat([x, t, y], axis=1))
        return z_pred, torch.exp(self.z_logstd)
        #return z_pred, torch.mm(self.z_std.T,self.z_std)

class linearCEVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        encoder_hidden_dim=4,
        decoder_hidden_dim=3,
        num_hidden=3,
        activation=nn.ELU(),
        device='cpu'
    ):
        super().__init__()
        
        z_dim = 1#1D assumed to make this clear and easy
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.device = device

        self.encoder = Encoder(
            input_dim, 
            encoder_hidden_dim, 
            num_hidden,
            activation,
            z_dim,
            device=device
        )
        self.decoder = Decoder(
            input_dim, 
            decoder_hidden_dim, 
            num_hidden,
            activation,
            z_dim,
            device=device
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
        z_std = z_std.repeat((x.shape[0],1))
        z = self.reparameterize(z_mean, z_std)
        
        x_pred, t_pred, y_pred = self.decoder(z, t)
        
        return z_mean, z_std, x_pred, t_pred, y_pred