import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

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
        x_dim,
        z_dim,
        device,
        ngf=64,
        nc=1,#Amount of channels, always one here
        p_y_zt_nn=False,
        p_y_zt_nn_layers=3,
        p_y_zt_nn_width=10,
        p_t_z_nn=False,
        p_t_z_nn_layers=3,
        p_t_z_nn_width=10,
        p_x_z_nn=False,
        p_x_z_nn_layers=3,
        p_x_z_nn_width=10
    ):
        super().__init__()
        self.x_dim = x_dim
        self.device = device
        
        #Image generating networks
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(z_dim,4*ngf,kernel_size=4,stride=2,bias=False),nn.ReLU())
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(4*ngf,2*ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ReLU())
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(2*ngf,ngf,kernel_size=4,stride=2,padding=2,bias=False), nn.ReLU())
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1,bias=False), nn.Tanh())
        self.logstd = nn.Parameter(torch.Tensor([1]))
        
        #Proxy networks
        self.x_nn = FullyConnected([p_x_z_nn_width]*p_x_z_nn_layers + [x_dim]) if p_x_z_nn else nn.ModuleList([nn.Linear(z_dim,1, bias=False) for i in range(x_dim)])
        self.x_log_std = nn.Parameter(torch.FloatTensor(x_dim*[1.]).to(device))
        
        #Treatment network
        self.t_nn = FullyConnected([p_t_z_nn_width]*p_t_z_nn_layers + [1]) if p_t_z_nn else nn.Linear(z_dim,1, bias=True)
        
        #y network
        self.y0_nn = FullyConnected([p_y_zt_nn_width]*p_y_zt_nn_layers + [1]) if p_y_zt_nn else nn.Linear(z_dim,1, bias=True)#If t and y binary
        self.y1_nn = FullyConnected([p_y_zt_nn_width]*p_y_zt_nn_layers + [1]) if p_y_zt_nn else nn.Linear(z_dim,1, bias=True)

    def forward(self,z,t):
        #z is dim (batch_size,z_dim)
        image = self.ct1(z[:,:,None,None])
        image = self.ct2(image)
        image = self.ct3(image)
        image = self.ct4(image)
        x_pred = torch.zeros(z.shape[0], self.x_dim, device=self.device)
        for i in range(self.x_dim):
            x_pred[:,i] = self.x_nn[i](z)[:,0]
        t_pred = self.t_nn(z)
        y_logits0 = self.y0_nn(z)
        y_logits1 = self.y1_nn(z)
        y_pred = y_logits1*t + y_logits0*(1-t)
        return image, torch.exp(self.logstd), x_pred, torch.exp(self.x_log_std), t_pred, y_pred
    
class Encoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        ngf=64,
        nc=1
    ):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(ngf,2*ngf,kernel_size=4,stride=2,padding=2,bias=False), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(2*ngf,4*ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ReLU())
        self.c4 = nn.Sequential(nn.Conv2d(4*ngf,100,kernel_size=4,stride=2,bias=False))
        self.fc = nn.Sequential(nn.Linear(100+x_dim+2,100),nn.ReLU())#I guess that this could be optimized somehow
        self.mean = nn.Linear(100,z_dim)
        self.logstd = nn.Linear(100,z_dim)
        
    
    def forward(self,image,x,t,y):
        temp = self.c1(image)
        temp = self.c2(temp)
        temp = self.c3(temp)
        temp = self.c4(temp)
        temp = self.fc(torch.cat([temp[:,:,0,0],x,t,y],1))
        z_mean = self.mean(temp)
        z_std = torch.exp(self.logstd(temp))
        return z_mean,z_std#dim (batch_size,z_dim)

class ImageCEVAE(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim=1,
        device='cpu',
        p_y_zt_nn=False,
        p_y_zt_nn_layers=3,
        p_y_zt_nn_width=10,
        p_t_z_nn=False,
        p_t_z_nn_layers=3,
        p_t_z_nn_width=10,
        p_x_z_nn=False,
        p_x_z_nn_layers=3,
        p_x_z_nn_width=10
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device=device
        )
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device=device
        )
        self.to(device)
        self.float()

    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, image, x, t, y):#Should have t, y
        z_mean, z_std = self.encoder(image, x, t, y)
        #TODO: works at least for z_dim=1, maybe errors if z_dim>1
        z = self.reparameterize(z_mean, z_std)
        
        image, image_std, x_pred, x_std, t_pred, y_pred = self.decoder(z,t)
        
        return image, image_std, z_mean, z_std, x_pred, x_std, t_pred, y_pred