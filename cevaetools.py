import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from CEVAE import *
import scipy
import pandas as pd
import pickle
import os
import glob
import re
import functools

class CEVAEDataset(Dataset):
    def __init__(self, df):
        x_cols = [c for c in df.columns if c.startswith("x")]
        self.X = torch.Tensor(df[x_cols].to_numpy())
        self.t = torch.Tensor(df["t"].to_numpy()[:,None])
        self.y = torch.Tensor(df["y"].to_numpy()[:,None])
        self.length = len(df)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            't': self.t[idx],
            'y': self.y[idx]
        }
    def __len__(self):
        return self.length
    
from scipy.stats import multinomial
def categorical_data_df(num_samples, z_probs, x_probs, t_probs, y_probs):
    """z_probs of shape (z_categories,), x_probs of shape (z_categories, x_dim, x_categories), 
    t_probs of shape (z_categories, t_categories), y_probs of shape (z_categories, t_categories, y_categories)"""
    z_c = z_probs.shape[0]
    _, x_dim, x_c = x_probs.shape
    _, t_c, y_c = y_probs.shape
    z = np.random.choice(z_c,num_samples,p=z_probs)
    x = np.zeros((num_samples,x_dim))
    t = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for z_cat in range(z_c):
        num_z_cat = sum(z==z_cat)
        for x_d in range(x_dim):
            x[z==z_cat,x_d] = np.random.choice(x_c,num_z_cat,p=x_probs[z_cat,x_d,:])
        t[z==z_cat] = np.random.choice(t_c,num_z_cat,p=t_probs[z_cat,:])
    for z_cat in range(z_c):
        for t_cat in range(t_c):
            indices = (z==z_cat) & (t==t_cat)
            y[indices] = np.random.choice(y_c,sum(indices),p=y_probs[z_cat,t_cat,:])
    z = z[:,None]
    t = t[:,None]
    y = y[:,None]
    df = pd.DataFrame(np.concatenate([z,x,t,y],1),columns=['z'] + ['x{}'.format(i) for i in range(x_dim)] + 
                      ['t','y'])
    return df

from scipy.stats import dirichlet
def generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha):
    """z_alpha of shape (z_categories,), x_alpha of shape (z_categories, x_dim, x_categories),
    t_alpha of shape (z_categories, t_categories), y_alpha of shape (z_categories, t,categories, y_categories)"""
    z_c, x_dim, x_c = x_alpha.shape
    _, t_c, y_c = y_alpha.shape
    z_probs = dirichlet.rvs(z_alpha, size=1)[0]
    x_probs = np.zeros(x_alpha.shape)
    t_probs = np.zeros(t_alpha.shape)
    y_probs = np.zeros(y_alpha.shape)
    for z_cat in range(z_c):
        for x_d in range(x_dim):
            x_probs[z_cat,x_d,:] = dirichlet.rvs(x_alpha[z_cat,x_d,:],size=1)[0]
        t_probs[z_cat,:] = dirichlet.rvs(t_alpha[z_cat,:],size=1)[0]
        for t_cat in range(t_c):
            y_probs[z_cat,t_cat,:] = dirichlet.rvs(y_alpha[z_cat,t_cat,:],size=1)[0]
    return z_probs,x_probs,t_probs,y_probs

def generate_dist_and_data(num_samples,z_alpha,x_alpha,t_alpha,y_alpha):
    #Wrapper function e.g. for function run_model_for_data_sets
    z_probs,x_probs,t_probs,y_probs = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
    df = categorical_data_df(num_samples, z_probs, x_probs, t_probs, y_probs)
    return (df,(z_probs,x_probs,t_probs,y_probs))

import torch.nn.functional as F
def estimate_model_py_dot(model,n=10000):
    """Assumes that y,t are categorical"""
    if model.y_mode != 0:
        py_dot = np.zeros((model.t_mode, model.y_mode))
    else:
        py_dot = np.zeros(model.t_mode)
    for t_idx in range(model.t_mode):
        z = torch.randn(n,model.z_dim)
        t = torch.ones(n,1)*t_idx
        _,_,_,_,y_pred,y_std = model.decoder(z,t)
        if model.y_mode == 2:#y_pred shape (n,1)
            py_dot[t_idx,:] = np.array([1-torch.sigmoid(y_pred).mean().item(),torch.sigmoid(y_pred).mean().item()])
        elif model.y_mode > 2:#y_pred shape (n,y_mode)
            py_dot[t_idx,:] = F.softmax(y_pred,dim=1).mean(0).detach().numpy()
        elif model.y_mode == 0:
            py_dot[t_idx] = y_pred.mean().item()#Really, E[y|do(t)]
    return py_dot

def estimate_true_py_dot(z_probs,y_probs):
    return (y_probs*z_probs[:,None,None]).sum(0)

def estimate_AID(model,z_probs,t_probs,y_probs,n=10000):
    t_marginal_probs = (z_probs[:,None]*t_probs).sum(0)
    model_py_dot = estimate_model_py_dot(model,n)
    true_py_dot = estimate_true_py_dot(z_probs,y_probs)
    AID = (np.abs(model_py_dot - true_py_dot).sum(1)*t_marginal_probs).sum()
    return AID

def train_model(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,
              p_y_zt_nn_layers=3, p_y_zt_nn_width=10, 
              p_t_z_nn_layers=3, p_t_z_nn_width=10,
              p_x_z_nn_layers=3, p_x_z_nn_width=10,
              q_z_nn_layers=3, q_z_nn_width=10,
              t_mode=2, y_mode=2, x_mode=[0], ty_separate_enc=False):
    
    model = CEVAE(x_dim, z_dim, device, p_y_zt_nn_layers,
        p_y_zt_nn_width, p_t_z_nn_layers, p_t_z_nn_width,
        p_x_z_nn_layers, p_x_z_nn_width, 
        q_z_nn_layers, q_z_nn_width,
        t_mode,y_mode,x_mode,ty_separate_enc)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    losses = {"total": [], "kld": [], "x": [], "t": [], "y": []}
    def kld_loss(mu, std):
        #Note that the sum is over the dimensions of z as well as over the units in the batch here
        var = std.pow(2)
        kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
        return kld
    
    different_modes = list(set(x_mode))
    x_same_mode_indices = dict()
    for mode in different_modes:
        x_same_mode_indices[mode] = [i for i,m in enumerate(x_mode) if m==mode]
    def get_losses(z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std,
                  x, t, y):
        kld = kld_loss(z_mean,z_std)
        x_loss = 0
        t_loss = 0
        y_loss = 0
        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(x_mode):
            if mode==0:
                x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            elif mode==2:
                x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
                pred_i += 1
            else:
                x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
                pred_i += mode
        if t_mode==2:
            t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
        elif t_mode==0:
            t_loss = -dist.Normal(loc=t_pred,scale=t_std).log_prob(t).sum()
        else:
            t_loss = -dist.Categorical(logits=t_pred).log_prob(t[:,0]).sum()
        if y_mode ==2:
            y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
        elif y_mode == 0:
            y_loss = -dist.Normal(loc=y_pred,scale=y_std).log_prob(y).sum()
        else:
            y_loss = -dist.Categorical(logits=y_pred).log_prob(y[:,0]).sum()
        return kld, x_loss, t_loss, y_loss
    
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = []
        epoch_kld_loss = []
        epoch_x_loss = []
        epoch_t_loss = []
        epoch_y_loss = []
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            x = data['X'].to(device)
            t = data['t'].to(device)
            y = data['y'].to(device)
            z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = model(x,t,y)
            kld, x_loss, t_loss, y_loss = get_losses(z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y)
            loss = kld + x_loss + t_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss.append(loss.item())
            epoch_kld_loss.append(kld.item())
            epoch_x_loss.append(x_loss.item())
            epoch_t_loss.append(t_loss.item())
            epoch_y_loss.append(y_loss.item())
        
        losses['total'].append(sum(epoch_loss))
        losses['kld'].append(sum(epoch_kld_loss))
        losses['x'].append(sum(epoch_x_loss))
        losses['t'].append(sum(epoch_t_loss))
        losses['y'].append(sum(epoch_y_loss))
        
        scheduler.step()
  
        if print_logs:
            #print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(sum(epoch_loss)))
            print("x: {}, t: {}, y: {}, kld: {}".format(sum(epoch_x_loss), sum(epoch_t_loss),
                                                        sum(epoch_y_loss), sum(epoch_kld_loss)))
            print
    
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['t'])
    ax[1,0].plot(losses['y'])
    ax[1,1].plot(losses['kld'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("t loss")
    ax[1,0].set_title("y loss")
    ax[1,1].set_title("kld loss")
    plt.show()
    
    return model, losses


def expand_parameters(params, iterated):
    """Helper function to get the elements in params to be lists of len(iterated)"""
    new_params = len(params)*[None]
    for i in range(len(params)):
        if not isinstance(params[i], list):
            new_params[i] = len(iterated)*[params[i]]#dim (len(train_arguments), len(iterated))
        else:
            assert len(params[i]) == len(iterated)
            new_params[i] = params[i].copy()
    return new_params

def run_model_for_data_sets(datasize, param_times,
                            folder, name, 
                            BATCH_SIZE, generate_data,
                            device, train_arguments, labels, z_alpha, x_alpha, t_alpha, y_alpha):
    """train_arguments is a list with the following:
    num_epochs, lr_start, lr_end, x_dim, z_dim,
      p_y_zt_nn_layers, p_y_zt_nn_width, 
      p_t_z_nn_layers, p_t_z_nn_width,
      p_x_z_nn_layers, p_x_z_nn_width,
      q_z_nn_layers, q_z_nn_width,
      t_binary, y_binary, x_mode, ty_separate_enc"""
    """Runs the model for a parameter sweep. Saves the results in data/{folder}.
    Currently just empties everything in the folder before starting on new stuff.
    Idea: Some of the arguments in train_arguments are datasize is lists, and 
    we iterate through those and save the results. 'iterated' is the list object which names 
    the results.
    NEW: generates the data generating distribution according to z_alpha, x_alpha, t_alpha and y_alpha"""
    
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
    
    datasize = expand_parameters([datasize], labels)[0]
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments))) #dim (len(iterated, len(train_arguments))
    
    datas = {label: {} for label in labels}
    models = {label: {} for label in labels}
    losses = {label: {} for label in labels}
    aux_datas = [0]*param_times
    for j in range(param_times):
        aux_data = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
        aux_datas[j] = aux_data
    for i in range(len(labels)):
        print("Param value {}".format(labels[i]))
        for j in range(param_times):
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            df = generate_data(num_samples, *aux_datas[j])
            dataset = CEVAEDataset(df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            model, loss = train_model(device, False, False, dataloader, *train_arguments[i])
            
            #data = (z, images, x, t, y, dataset)
            datas[labels[i]][j] = df
            models[labels[i]][j] = model
            losses[labels[i]][j] = loss

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,labels[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(df, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(loss, file)
            print("Estimated causal effect: {} true value: {}".format(estimate_AID(model,aux_datas[j][0],aux_datas[j][2],aux_datas[j][3]), 0))
    
    with open("./data/{}/aux_datas_{}".format(folder,name), "wb") as file:
        pickle.dump(aux_datas, file)
    
    return datas, models, losses, aux_datas


def run_model_for_x_dims(datasize, param_times,
                            folder, name, 
                            BATCH_SIZE,
                            device, train_arguments, alpha, x_dim, n_cat):
    """This assumes that x_dim is a list which we want to iterate over.
    More specific because x_dim is more difficult to mess with using the run_model_for_data_sets function."""
    """TODO: Think about the software engineering side of all this. What would be a better, simpler way to run
    all these experiments? Can I figure out a really generic function?"""
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
            
    datasize = expand_parameters([datasize], x_dim)[0]
    train_arguments = expand_parameters(train_arguments, x_dim)
    train_arguments = list(map(list, zip(*train_arguments))) #dim (len(iterated), len(train_arguments))
    
    datas = {label: {} for label in x_dim}
    models = {label: {} for label in x_dim}
    losses = {label: {} for label in x_dim}
    aux_datas = {label: {} for label in x_dim}
    
    for i in range(len(x_dim)):
        print("Param value {}".format(x_dim[i]))
        for j in range(param_times):
            z_alpha = np.array([2]*n_cat)
            x_alpha = np.array([[[2]*n_cat]*x_dim[i]]*n_cat)#x_dim 10
            t_alpha = np.array([[2]*n_cat]*n_cat)#t_cat 2
            y_alpha = np.array([[[2]*n_cat]*n_cat]*n_cat)#y_cat 2
            z_probs,x_probs,t_probs,y_probs = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
            
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            df = categorical_data_df(num_samples,z_probs,x_probs,t_probs,y_probs)
            dataset = CEVAEDataset(df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            print("Train arguments: ", *train_arguments[i])
            model, loss = train_model(device, False, False, dataloader, *train_arguments[i])
            
            aux_data = (z_probs,x_probs,t_probs,y_probs)
            datas[x_dim[i]][j] = df
            models[x_dim[i]][j] = model
            losses[x_dim[i]][j] = loss
            aux_datas[x_dim[i]][j] = aux_data

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,x_dim[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(df, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(loss, file)
            with open("./data/{}/aux_data_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(aux_data, file)
    
    return datas, models, losses, aux_datas