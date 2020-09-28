import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from imagedata import *
from imageCEVAE import *
import scipy
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
import glob
import re

def estimate_imageCEVAE_ATE(model):
    """Uses Monte Carlo Integration"""
    z_dim = model.z_dim
    device = model.device
    n = 100000
    z = torch.randn(n,z_dim).to(device)
    py_dot1 = torch.sigmoid(model.decoder.y1_nn(z)).mean()
    py_dot0 = torch.sigmoid(model.decoder.y0_nn(z)).mean()
    ATE = py_dot1 - py_dot0
    return ATE, py_dot1, py_dot0

def best_estimate_ate(z,t,y):
    """Returns the best ATE and p(y|do(t)) that one could estimate if one new the true z and the generating process"""
    df = pd.DataFrame(torch.cat([z,t,y],1).detach().numpy(), columns=['z{}'.format(i) for i in range(3)] + ['t', 'y'])
    logreg_t1 = LogisticRegression(penalty='none')
    logreg_t0 = LogisticRegression(penalty='none')
    logreg_t1.fit(X=df[df.t==1].iloc[:,:1],y=df[df.t==1]['y'])
    logreg_t0.fit(X=df[df.t==0].iloc[:,:1],y=df[df.t==0]['y'])
    z_sample = np.random.randn(1000000,1)
    p_y_dot1_best = logreg_t1.predict_proba(z_sample)[:,1].mean()
    p_y_dot0_best = logreg_t0.predict_proba(z_sample)[:,1].mean()
    return p_y_dot1_best-p_y_dot0_best, p_y_dot1_best, p_y_dot0_best

def viz_image_space(generator, dim1=0, dim2=1, gendim=3):
    """"""
    with torch.no_grad():
        n = 10
        z = gendim*[None]
        z[dim1] = torch.linspace(-2.5,2.5,n)[:,None].repeat(n,1)
        z[dim2] = torch.linspace(2.5,-2.5,n)[:,None].repeat_interleave(n,0)
        for i in range(gendim):
            if i != dim1 and i != dim2:
                z[i] = torch.zeros(n**2,1)
        z = torch.cat(z,1)[:,:,None,None]
        out = generator(z).squeeze().cpu().detach().numpy()
    out_grid = np.zeros((28*n,28*n))
    grid_vals = np.linspace(-1,1,n)
    for i in range(n**2):
        x = (i % n)
        y = (i // n)
        out_grid[y*28:(y+1)*28,x*28:(x+1)*28] = out[i]
    plt.figure()
    plt.xlabel("Dim {}".format(dim1))
    plt.ylabel("Dim {}".format(dim2))
    plt.imshow(out_grid, extent=(-2.5, 2.5, -2.5, 2.5))
    plt.show()
    
def viz_other_space(generator, dim1=0, dim2=1, gendim=3):
    n = 100
    z_range = np.linspace(-2.5,2.5,n)
    z0,z1 = np.meshgrid(z_range,np.array(list(reversed(z_range))))
    z2 = np.zeros((n,n,1))
    z = gendim*[None]
    z[dim1] = z0[:,:,None]
    z[dim2] = z1[:,:,None]
    for i in range(gendim):
        if i != dim1 and i != dim2:
            z[i] = torch.zeros(n,n,1)
    z = np.concatenate(z,2).reshape(-1,gendim)
    z = torch.Tensor(z)
    y_pred = generator(z)
    plt.figure()
    plt.imshow(y_pred.reshape(100,100).cpu().detach(), extent = (z_range[0],z_range[-1],z_range[0],z_range[-1]))
    plt.xlabel("Dim {}".format(dim1))
    plt.ylabel("Dim {}".format(dim2))
    plt.show()

def kld_loss(mu, std):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    var = std.pow(2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return kld

def train_model(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,
              p_y_zt_nn=False, p_y_zt_nn_layers=3, p_y_zt_nn_width=10, 
              p_t_z_nn=False, p_t_z_nn_layers=3, p_t_z_nn_width=10,
              p_x_z_nn=False, p_x_z_nn_layers=3, p_x_z_nn_width=10, loss_scaling=1):
    
    model = ImageCEVAE(x_dim, z_dim, device=device, p_y_zt_nn=p_y_zt_nn, p_y_zt_nn_layers=p_y_zt_nn_layers,
        p_y_zt_nn_width=p_y_zt_nn_width, p_t_z_nn=p_t_z_nn, p_t_z_nn_layers=p_t_z_nn_layers, p_t_z_nn_width=p_t_z_nn_width,
        p_x_z_nn=p_x_z_nn, p_x_z_nn_layers=p_x_z_nn_layers, p_x_z_nn_width=p_x_z_nn_width)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    losses = {"total": [], "kld": [], "image": [], "x": [], "t": [], "y": []}
    
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_image_loss = 0
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            image = data['image'].to(device)
            x = data['X'].to(device)
            t = data['t'].to(device)
            y = data['y'].to(device)
            image_mean, image_std, z_mean, z_std, x_pred, x_std, t_pred, y_pred = model(image,x,t,y)
            kld = kld_loss(z_mean, z_std)
            #image_loss = -dist.Normal(loc=image_mean, scale = image_std).log_prob(image).sum()
            image_loss = -dist.Bernoulli(logits=image_mean).log_prob(image).sum()*loss_scaling
            x_loss = -dist.Normal(loc=x_pred, scale = x_std).log_prob(x).sum()
            t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
            y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
            loss = kld + image_loss + x_loss + t_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss += loss
            epoch_kld_loss += kld
            epoch_image_loss += image_loss
            epoch_x_loss += x_loss
            epoch_t_loss += t_loss
            epoch_y_loss += y_loss
        
        losses['total'].append(epoch_loss.item())
        losses['kld'].append(epoch_kld_loss.item())
        losses['image'].append(epoch_image_loss.item())
        losses['x'].append(epoch_x_loss.item())
        losses['t'].append(epoch_t_loss.item())
        losses['y'].append(epoch_y_loss.item())
        
        scheduler.step()
        
        if print_logs:
            print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(epoch_loss))
            print("Image: {}, x: {}, t: {}, y: {}".format(epoch_image_loss, epoch_x_loss,epoch_t_loss,epoch_y_loss))
            print()
            if epoch_x_loss > 1e8:
                generator = lambda z: model.decoder.x_nn(z.to(device))
                viz_other_space(generator, dim1=0, dim2=0, gendim=1)
                print("x_std: ", x_std)

        if plot_curves and z_dim > 1:
            fig,ax = plt.subplots(2,5, figsize=(15,5))
            with torch.no_grad():
                z = torch.randn(10,z_dim).to(device)
                t = torch.zeros(z.shape[0],1).to(device)
                out, _, _, _, _, _ = model.decoder(z, t)
                out = torch.sigmoid(out).squeeze().cpu().detach().numpy()
                for i in range(10):
                    x = i % 5
                    y = i // 5
                    ax[y][x].imshow(out[i])
                    ax[y][x].set_title("{:.2f},{:.2f}".format(z[i,0],z[i,1]))
            plt.show()

    plt.figure()
    plt.plot([loss for loss in losses['total'] if loss < 1e8])
    plt.title("Loss at end of each epoch")
    plt.show()
    
    return model, losses

def train_decoder(device, model, print_logs, train_loader, num_epochs, lr_start, lr_end):
    """Continues the training of the model, but freezes the encoder and only trains the p(y|z,t), p(t|z) and p(x|z)
    networks"""
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    for epoch in range(num_epochs):
        if print_logs:
            print("Epoch {}:".format(epoch))
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        for data in train_loader:
            image = data['image'].to(device)
            x = data['X'].to(device)
            t = data['t'].to(device)
            y = data['y'].to(device)
            with torch.no_grad():
                z_mean,z_std = model.encoder(image,x,t,y)
                z = model.reparameterize(z_mean, z_std)
            image_mean, image_std, x_pred, x_std, t_pred, y_pred = model.decoder(z,t)
            #image_loss = -dist.Normal(loc=image_mean, scale = image_std).log_prob(image).sum()
            #image_loss = -dist.Bernoulli(logits=image_mean).log_prob(image).sum()
            x_loss = -dist.Normal(loc=x_pred, scale = x_std).log_prob(x).sum()
            t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
            y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
            loss = x_loss + t_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_x_loss += x_loss.item()
            epoch_t_loss += t_loss.item()
            epoch_y_loss += y_loss.item()
        scheduler.step()
        if print_logs:
            print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("x: {}, t: {}, y: {}".format(epoch_x_loss,epoch_t_loss,epoch_y_loss))
            print()

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
                            BATCH_SIZE, generate_data, dataparameters, track_function, true_value,
                            device, train_arguments, labels, 
                            post_decoder_training=False, post_decoder_arguments=[], loss_scaling=1):
    """train_arguments is a list with the following:
    num_epochs, lr_start, lr_end, x_dim, z_dim,
    p_y_zt_nn, p_y_zt_nn_layers, p_y_zt_nn_width, 
    p_t_z_nn, p_t_z_nn_layers, p_t_z_nn_width,
    p_x_z_nn, p_x_z_nn_layers, p_x_z_nn_width"""
    """Runs the model for a parameter sweep. Saves the results in data/{folder}.
    Currently just empties everything in the folder before starting on new stuff.
    Idea: Some of the arguments in train_arguments are datasize is lists, and 
    we iterate through those and save the results. 'iterated' is the list object which names 
    the results
    if post_decoder_training=True then does extra training for the decoder only. post_decoder_arguments consists of:
    num_epochs, lr_start, lr_end"""
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
    for i in range(len(labels)):
        for j in range(param_times):
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            z, images, x, t, y, dataset = generate_data(num_samples, *dataparameters)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            model, loss = train_model('cuda', False, False, dataloader, *train_arguments[i], loss_scaling)
            if post_decoder_training:
                train_decoder(device, model, print_logs, train_loader, *post_decoder_arguments)
            
            data = (z, images, x, t, y, dataset)
            #datas[labels[i]][j] = data
            #models[labels[i]][j] = model
            #losses[labels[i]][j] = loss

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,labels[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(data, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(loss, file)
            print("Estimated causal effect: {} true value: {}".format(track_function(model), true_value))
            
    return datas, models, losses


def load_dfs_models(folder, name, train_arguments, datasize, labels, device):
    """Loads dataframes and trained models from data/{folder}/ that match the experiment name"""
    datasize = expand_parameters([datasize], labels)
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments)))
    #We see only the labels in the folder, but we want the indices for accessing other arguments (train_arguments)
    labels_to_index = dict(zip(labels, range(len(labels))))
    
    datas = {}
    models = {}
    losses = {}
    for file in os.listdir("data/{}".format(folder)):
        match = re.search(r"([^_]*)_(.*)_(\d*)_(\d*)", file)
        if match.group(2) == name:
            if match.group(1) == "data":
                if not int(match.group(3)) in datas:
                    with open("data/{}/{}".format(folder,file), "rb") as file:
                        datas[int(match.group(3))] = {int(match.group(4)): pickle.load(file)}
                else:
                    with open("data/{}/{}".format(folder,file), "rb") as file:
                        datas[int(match.group(3))][int(match.group(4))] = pickle.load(file)
            elif match.group(1) == "loss":
                if not int(match.group(3)) in losses:
                    with open("data/{}/{}".format(folder,file), "rb") as file:
                        losses[int(match.group(3))] = {int(match.group(4)): pickle.load(file)}
                else:
                    with open("data/{}/{}".format(folder,file), "rb") as file:
                        losses[int(match.group(3))][int(match.group(4))] = pickle.load(file)
            elif match.group(1) == "model":
                index = labels_to_index[int(match.group(3))]
                num_epochs, lr_start, lr_end, x_dim, z_dim, p_y_zt_nn, p_y_zt_nn_layers, p_y_zt_nn_width, p_t_z_nn, p_t_z_nn_layers, p_t_z_nn_width, p_x_z_nn, p_x_z_nn_layers, p_x_z_nn_width = train_arguments[index]
                model = ImageCEVAE(x_dim, z_dim, device=device, p_y_zt_nn=p_y_zt_nn, p_y_zt_nn_layers=p_y_zt_nn_layers,
                    p_y_zt_nn_width=p_y_zt_nn_width, p_t_z_nn=p_t_z_nn, p_t_z_nn_layers=p_t_z_nn_layers, 
                    p_t_z_nn_width=p_t_z_nn_width, p_x_z_nn=p_x_z_nn, 
                    p_x_z_nn_layers=p_x_z_nn_layers, p_x_z_nn_width=p_x_z_nn_width)
                model.load_state_dict(torch.load("data/{}/{}".format(folder,file)))
                model.eval()
                if not int(match.group(3)) in models:
                    models[int(match.group(3))] = {int(match.group(4)): model}
                else:
                    models[int(match.group(3))][int(match.group(4))] = model
    return datas, models, losses