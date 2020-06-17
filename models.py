import os
import math, random
import scipy.stats as stats
import numpy as np
import pandas as pd
import collections

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

PLOT_STYLE = 'ggplot'

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


class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """
    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))

class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])
        

    def forward(self, x):#seems reasonable but means that min variance is 1e-3
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(
            loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale

class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)
    
class MultipleBernoulliNet(DistributionNet):
    """Could be used if we want multiple Bernoulli distributions inferred by a neural network"""
    def __init__(self, sizes, finaldim):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [finaldim])
    
    def forward(self,x):
        logits = self.fc(x).clamp(min=-10, max=10)
        return logits
    
    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)

class Decoder(nn.Module):
    def __init__(
        self, 
        input_dim, #x dim
        hidden_dim, 
        num_hidden,#hidden layers for the y prediction and x prediction
        activation=nn.ELU(), 
        z_dim=2, 
        device='cpu',
        t_z_layers=0,
        z_mode='normal',
        x_mode='normal'
        
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.x_mode = x_mode
        self.z_mode = z_mode

        # p(x|z) <- REPLACE with diagnormalnet or bernoullinet or something depending on x
        #Should we have a completely different NN for each X dimension?
        self.hidden_x_nn = DiagNormalNet(
            [z_dim] + num_hidden*[hidden_dim] + [input_dim]
        )
        #One NN to predict all probs. or a different NN for each prob? Also, I guess we could just use t_z_layers here
        #(otherwise num_hidden)
        #If z is binary, what sense does it sense to even have the neural network here?
        self.hidden_x_nns_binary = nn.ModuleList([BernoulliNet([z_dim] + t_z_layers*[hidden_dim]) for i in range(input_dim)])

        # p(t|z) <- #Just 1 layer? How was it in the paper?
        self.treatment_logits_nn = BernoulliNet(
            [z_dim] + t_z_layers * [hidden_dim]
        )

        # p(y|t,z) <- These seem OK
        # TODO doesnt look like this is TARNet <- does it have to be?
        self.y0_nn = BernoulliNet(
            [z_dim] +
            num_hidden * [hidden_dim],
        )
        self.y1_nn = BernoulliNet(
            [z_dim] +
            num_hidden * [hidden_dim]
        )

    def forward(self, z, t):
        xloc = 0
        xscale = 0
        x_logits = 0
        if self.x_mode == 'normal':
            xloc, xscale = self.hidden_x_nn.forward(z) #<*- will be loc, scale
        elif self.x_mode == 'binary':
            x_logits = torch.zeros(z.shape[0], self.input_dim)
            for i in range(self.input_dim):
                x_logits[:,i] = self.hidden_x_nns_binary[i](z)[0][:,0]
        t_logits, = self.treatment_logits_nn(z) # <- OK
        #We need to have the observed t here because that ends up in the likelihood function of y through the p(y|x,t)
        #neural network. Note that the forward function is thus useful mostly in training (we have to choose t)
        y_logits0 = self.y0_nn(z)[0]
        y_logits1 = self.y1_nn(z)[0]
        y_logits = y_logits1*t + y_logits0*(1-t)#torch.where(t==1, y_logits1, y_logits0)

        return (#One should choose between (xloc,xscale) and x_logits appropriately when using
            xloc, xscale, x_logits, t_logits,
            y_logits
        )
    
    def sample(self,size):
        #Create a sample (z,x,t,y) according to the learned joint distribution
        """TODO: Assumes normally distributed z and x for now"""
        if self.z_mode == "normal":
            z_sample = torch.randn((size,self.z_dim))
        elif self.z_mode == "binary":
            z_sample = dist.Bernoulli(torch.Tensor([0.5])).sample((size,))
        if self.x_mode == "normal":
            x_loc, x_var = self.hidden_x_nn(z_sample)
            x_sample = torch.randn((size,self.input_dim))*torch.sqrt(x_var) + x_loc
        elif self.x_mode == "binary":
            x_logits = torch.zeros(size, self.input_dim)
            for i in range(self.input_dim):
                x_logits[:,i] = self.hidden_x_nns_binary[i](z_sample)[0][:,0]
            x_sample = dist.Bernoulli(torch.sigmoid(x_logits)).sample()
        t_logits, = self.treatment_logits_nn(z_sample)
        t_sample = dist.Bernoulli(torch.sigmoid(t_logits)).sample()
        y_logits0 = self.y0_nn(z_sample)[0]
        y_logits1 = self.y1_nn(z_sample)[0]
        y_logits = y_logits1*t_sample + y_logits0*(1-t_sample)
        y_sample = dist.Bernoulli(torch.sigmoid(y_logits)).sample()
        return z_sample, x_sample, t_sample, y_sample

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        num_hidden, 
        activation=nn.ELU(), 
        z_dim=2,
        device='cpu',
        z_mode = 'normal'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.z_mode = z_mode

        # q(z|x,t,y)
        self.q_z_nn = FullyConnected(#correspnds to g1 in the paper
            [input_dim + 1] +#Why + 1? <- because y is also a parameter
            num_hidden * [hidden_dim],#Why num_hidden - 1 again?
            activation
        )
        self.q_z0_nn = DiagNormalNet(#Only 1 layer again, but okay I guess since q_z_nn has many
            [hidden_dim, z_dim]
        )
        self.q_z1_nn = DiagNormalNet(
            [hidden_dim, z_dim]
        )
        self.q_z0_nn_binary = BernoulliNet([hidden_dim])#The situation is a bit funny in the complete binary case, but I guess we can continue using the same architecture
        self.q_z1_nn_binary = BernoulliNet([hidden_dim])

    def forward(self, x, t, y):#Should take as inputs the obs. t and y
        # q(z|x,t,y) <- here we should have obs. t and y, otherwise maybe ok
        hqz = self.q_z_nn.forward(torch.cat((x, y), 1))
        z_mu = 0
        z_var = 0
        z_logits = 0
        if self.z_mode == 'normal':
            loc0, scale0 = self.q_z0_nn(hqz)
            loc1, scale1 = self.q_z1_nn(hqz)
            z_mu = t*loc1 + (1-t)*loc0#torch.where(t == 1, loc1, loc0)
            z_var = t*scale1 + (1-t)*scale0#torch.where(t == 1, scale1, scale0)
        elif self.z_mode == 'binary':
            z_logits0 = self.q_z0_nn_binary(hqz)[0]
            z_logits1 = self.q_z1_nn_binary(hqz)[0]
            z_logits = t*z_logits1 + (1-t)*z_logits0

        return z_mu, z_var, z_logits

class tInferNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        #hidden_dim=2, 
        #num_hidden=2, 
        activation=nn.ELU(),
        device='cpu'
    ):
        super().__init__()
        # q(t|x) <- This has just 1 layer
        self.q_treatment_logits_nn = BernoulliNet(
            [input_dim]
        )
    def forward(self, x):
        return self.q_treatment_logits_nn(x)

class yInferNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        num_hidden, 
        activation=nn.ELU(),
        device='cpu'
    ):
        super().__init__()
        # q(y|x,t)
        # this is for shared head
        self.q_outcome_nn = FullyConnected(
            [input_dim] +
            [hidden_dim] * num_hidden,
            activation
        )
        # these are for different outcomes <- 1 layer again? 
        #Should all networks e.g. have the same amount of layers?
        self.q_outcome_t0_nn = BernoulliNet(
            [hidden_dim]
        )
        self.q_outcome_t1_nn = BernoulliNet(
            [hidden_dim]
        )
    
    def forward(self, x, t):
        # q(y|x,t)
        hqy = self.q_outcome_nn.forward(x)
        params0 = self.q_outcome_t0_nn.forward(hqy)[0]#These are just bs x 1 for bernoulli dist
        params1 = self.q_outcome_t1_nn.forward(hqy)[0]
        y_logits = params1*t + params0*(1-t)#torch.where(t == 1, params1, params0)#t should also be bs x 1?
        return y_logits

class CEVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        encoder_hidden_dim=4, 
        decoder_hidden_dim=4,
        num_hidden=3,
        activation=nn.ELU(), 
        z_dim=2, 
        device='cpu',
        t_z_layers = 0,
        z_mode='normal',
        x_mode='normal'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.z_mode = z_mode
        self.x_mode = x_mode

        self.encoder = Encoder(
            input_dim, 
            encoder_hidden_dim, 
            num_hidden,
            activation,
            z_dim,
            z_mode=z_mode
        )
        self.decoder = Decoder(# <- TODO: need to add the new inputs
            input_dim, 
            decoder_hidden_dim, 
            num_hidden,
            activation,
            z_dim,
            t_z_layers=t_z_layers,
            z_mode=z_mode,
            x_mode=x_mode
        )
        #self.t_infer = tInferNet( <- These should just be trained separately somewhere
        #    input_dim, 
            #hidden_dim, 
            #num_hidden,
        #    activation,
        #    z_dim
        #)
        #self.y_infer = yInferNet(
        #    input_dim, 
        #    hidden_dim, 
        #    num_hidden,
        #    activation,
        #    z_dim
        #)

        self.to(device)
        self.float()

    def reparameterize(self, mean, var):
        # samples from unit norm and does reparam trick
        std = torch.sqrt(var)#This uses the variance with (at least 1e-3) maybe could just be logvar
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def reparameterize_bernoulli(self, logits):
        #Tries to do the Gumbel reparametrization trick
        #u = torch.rand_like(logits)
        #e = -torch.log(-torch.log(u))
        e = torch.rand_like(logits)
        probs = torch.sigmoid(logits)
        z = torch.sigmoid(torch.log(e) - torch.log(1-e) + torch.log(probs) - torch.log(1-probs))
        return z

    def forward(self, x, t, y):#Should have t, y
        z_mean = 0
        z_var = 0
        z_logits = 0
        
        z_mean, z_var, z_logits = self.encoder.forward(x, t, y)
        if self.z_mode == 'normal':
            z = self.reparameterize(z_mean, z_var)#Note just 1 sample
        elif self.z_mode == 'binary':
            z = self.reparameterize_bernoulli(z_logits)
        (x_mean, x_var, x_logits, t_logits,
            y_logits) = self.decoder(z,t)

        #Auxiliary inference distributions used for out of sample prediction
        #tinfer_logit = self.t_infer(x)[0]
        #yinfer_logit = self.y_infer(x,t)

        return (
            z_mean, z_var, z_logits, x_mean, x_var, x_logits, t_logits, y_logits
        )
    
    def evaluate_batch(self, x):
        """Returns an estimate of E[y|x,do(t=1)] and E[y|x,do(t=0)]"""
        #This is assuming binary t and y for now
        #Should be used inside with torch.no_grad()
        bs = x.shape[0]
        tinfer_prob = F.sigmoid(self.t_infer(x)[0])#bs x 1
        yinfer_prob_t0 = F.sigmoid(self.y_infer(x,torch.zeros((x.shape[0],1))))
        yinfer_prob_t1 = F.sigmoid(self.y_infer(x,torch.ones((x.shape[0],1))))
        z_tyx_dist = [[None,None],[None,None]]
        for y in range(2):
            for t in range(2):
                mean, var = self.encoder(x,t*torch.ones(bs, 1),y*torch.ones(bs, 1))#bs x z_dim shape
                z_tyx_dist[y][t] = (mean, var)
        #Samples for the integration
        z_sample = torch.randn(100,x.shape[0],self.z_dim)#components of z are N(0,1), shape 100xbsxzdim
        #Probabilities of q(z|x) (= sum_t sum_y [ q(z|t,y,x)q(y|t,x)q(t|x) ])
        q_z_tyx = [[0,0],[0,0]]
        for t in range(2):
            for y in range(2):
                mean, std = z_tyx_dist[y][t]
                q_z_tyx[y][t] = torch.FloatTensor(stats.norm.pdf(z_sample, mean, std).prod(axis=2)).T#shape bsx100
        q_z_x = q_z_tyx[0][0]*(1-tinfer_prob)*(1-yinfer_prob_t0) + \
            q_z_tyx[0][1]*(tinfer_prob)*(1-yinfer_prob_t1) + \
            q_z_tyx[1][0]*(1-tinfer_prob)*(yinfer_prob_t0) + \
            q_z_tyx[1][1]*(tinfer_prob)*(yinfer_prob_t1)#shape bsx100
        #The probability of y=1 given x and do(t=1), also the expectation for bernoulli y
        #Can these get >1 because of sampling from a continuous distribution?
        q_y_zt1 = F.sigmoid(self.decoder.y1_nn(z_sample)[0]).squeeze().T#shape 100xbsx1 -> bsx100
        q_y_zt0 = F.sigmoid(self.decoder.y0_nn(z_sample)[0]).squeeze().T
        E_y_x_do1 = (q_y_zt1 * q_z_x).mean(axis=1)
        E_y_x_do0 = (q_y_zt0 * q_z_x).mean(axis=1)#Should be shape (bs,)
        return E_y_x_do1, E_y_x_do0
        
    
    def evaluate(self, x):
        """Returns an estimate of E[y|x,do(t=1)] and E[y|x,do(t=0)] for x = number"""
        #This is assuming binary t and y for now
        #Should be used inside with torch.no_grad()
        bs = x.shape[0]
        tinfer_prob = F.sigmoid(self.t_infer(x))#Just a number
        yinfer_prob_t0 = F.sigmoid(self.y_infer(x,0))
        yinfer_prob_t1 = F.sigmoid(self.y_infer(x,1))
        z_tyx_dist = [[None,None],[None,None]]
        for y in range(2):
            for t in range(2):
                mean, var = self.encoder(x,t*torch.ones(bs, 1),y*torch.ones(bs, 1))#z_dim shape?
                z_tyx_dist[y][t] = (mean, var)
        #Samples for the integration
        z_sample = torch.randn(100,self.z_dim)#components of z are N(0,1)
        #Probabilities of q(z|x) (= sum_t sum_y [ q(z|t,y,x)q(y|t,x)q(t|x) ])
        q_z_x = torch.zeros((100,x.shape[0]))
        q_z_tyx = [[0,0],[0,0]]
        for t in range(2):
            for y in range(2):
                mean, std = z_tyx_dist[y][t]
                q_z_tyx[y][t] = torch.FloatTensor(stats.norm.pdf(z_sample, mean, std).prod(axis=1))#shape 100
        q_z_x = q_z_tyx[0][0]*(1-tinfer_prob)*(1-yinfer_prob_t0) + \
            q_z_tyx[0][1]*(tinfer_prob)*(1-yinfer_prob_t1) + \
            q_z_tyx[1][0]*(1-tinfer_prob)*(yinfer_prob_t0) + \
            q_z_tyx[1][1]*(tinfer_prob)*(yinfer_prob_t1)
        E_y_x_do1 = (F.sigmoid(self.decoder.y1_nn(z_sample)[0]) * q_z_x).mean()
        E_y_x_do0 = (F.sigmoid(self.decoder.y0_nn(z_sample)[0]) * q_z_x).mean()
        return E_y_x_do1, E_y_x_do0