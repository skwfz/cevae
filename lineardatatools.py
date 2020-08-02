from lineardatamodels import *
from lineartoydata import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import itertools

sns.set_style('whitegrid')
PLOT_STYLE='ggplot'

import os
import re
import glob

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import ignite.metrics as ignite_metrics
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

def safe_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-20,20)))

def p_y_zt_from_model(model, p_y_zt_nn):
    """Wrapper function that returns convenient p(y|z,t) estimates from the model with binary t,y. Assumes z_dim=1"""
    def p_y_zt1_linear(z):
        return safe_sigmoid(model.decoder.y1_nn.weight.detach().numpy()*z + model.decoder.y1_nn.bias.detach().numpy())
    def p_y_zt0_linear(z):
        return safe_sigmoid(model.decoder.y0_nn.weight.detach().numpy()*z + model.decoder.y0_nn.bias.detach().numpy())
    def p_y_zt1_nn(z):
        if type(z) == type(0.1):
            z = np.array([z])
        return safe_sigmoid(model.decoder.y1_nn_real(torch.Tensor(z[:,None])).squeeze().detach().numpy())
    def p_y_zt0_nn(z):
        if type(z) == type(0.1):
            z = np.array([z])
        return safe_sigmoid(model.decoder.y0_nn_real(torch.Tensor(z[:,None])).squeeze().detach().numpy())
    if p_y_zt_nn:
        return p_y_zt1_nn, p_y_zt0_nn
    else:
        return p_y_zt1_linear, p_y_zt0_linear

def p_y_zt_from_true_dist(y_a0, y_b0, y_a1, y_b1):
    def p_y_zt1(z):
        return safe_sigmoid(y_a1*z + y_b1)
    def p_y_zt0(z):
        return safe_sigmoid(y_a0*z + y_b0)
    return p_y_zt1, p_y_zt0

def linear_binary_ty_ate(p_y_zt1_func, p_y_zt0_func):
    """Calculates the ATE assuming the standard normal p(z) distribution and given P(y|z,t).
    Assumes z_dim==1 for the model."""
    p_y_dot1,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt1_func(z), -np.inf, np.inf)
    p_y_dot0,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt0_func(z), -np.inf, np.inf)
    return p_y_dot1 - p_y_dot0

def linear_binary_ty_ate_2(p_y_zt1_func, p_y_zt0_func):
    """Same as linear_binary_ty_ate but uses the trapezoidal rule directly. Probably less issues than with scipy."""
    n = 10000
    box_width = 20/(n-1)
    full_range = np.linspace(-10,10,n)
    p_y_dot1 = ((p_y_zt1_func(full_range[:-1])*scipy.stats.norm.pdf(full_range[:-1]) \
                 + p_y_zt1_func(full_range[1:])*scipy.stats.norm.pdf(full_range[1:]))*box_width/2).sum()
    p_y_dot0 = ((p_y_zt0_func(full_range[:-1])*scipy.stats.norm.pdf(full_range[:-1]) \
                 + p_y_zt0_func(full_range[1:])*scipy.stats.norm.pdf(full_range[1:]))*box_width/2).sum()
    return p_y_dot1 - p_y_dot0

def linear_binary_ty_ate_3(model, z_dim):
    """Handles also higher z dimensions. Uses the basic integration rule"""
    n = 1000
    box_width = 12/n
    z_range = np.linspace(-6,6,n)
    z = torch.Tensor(list(itertools.product(z_range, repeat=z_dim)))
    t1 = torch.ones(n**z_dim,1)
    t0 = torch.zeros(n**z_dim,1)
    ypred1 = torch.sigmoid(model.decoder(z,t1)[2]).detach().numpy().squeeze()
    ypred0 = torch.sigmoid(model.decoder(z,t0)[2]).detach().numpy().squeeze()
    p_y_dot1 = np.sum(ypred1 * np.product(scipy.stats.norm.pdf(z),1))*box_width**z_dim
    p_y_dot0 = np.sum(ypred0 * np.product(scipy.stats.norm.pdf(z),1))*box_width**z_dim
    return p_y_dot1 - p_y_dot0
        
    

def avg_causal_L1_dist(model, c_yt, c_yz, s_y, c_t, s_t, c_x, p_y_zt_nn, n=100, lim=6):
    """Calculates
    ∫|𝑃(𝑦|𝑑𝑜(𝑡),𝑃𝑡𝑟𝑢𝑒(𝑦|𝑑𝑜(𝑡))|𝐿1𝑃(𝑡)𝑑𝑡
    for the model with continuous t and y. Assumes z_dim=1̈́
    TODO: Maybe this should have some kind error analysis"""
    #First calculate the P(t) function
    t_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/(n-1)
    y_len = 2*lim/(n-1)
    t_len = 2*lim/(n-1)
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t

    #P(t) (integration by the simplest possible rule here, could make better with e.g. trapezoidal)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len#shape (z_range, t_range)

    #Find out whether we should flip z in case the model learned it the wrong way around
    #NOTE: Probably not needed for P(y|do(t)) since integrated out, but otherwise could matter
    z_range_model = z_range if np.sign(model.decoder.x_nns[0].weight.item()) == np.sign(c_x[0]) else np.flip(z_range)

    #P(y|z,t)
    zt_range = np.concatenate([np.repeat(z_range_model[:,None],n,axis=0),np.tile(t_range[:,None],(n,1))],axis=1)#shape (1000*1000, 2)
    if p_y_zt_nn:
        py_zt_mean_model = torch.reshape(model.decoder.y_nn_real(torch.Tensor(zt_range)),(n,n)).detach().numpy()#shape (z_range, t_range)
    else:
        py_zt_mean_model = torch.reshape(model.decoder.y_nn(torch.Tensor(zt_range)),(n,n)).detach().numpy()#shape (z_range, t_range)
    py_zt_std_model = torch.exp(model.decoder.y_log_std).detach().numpy()
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y

    #P(y|do(t))
    py_dot_model = np.zeros((n,n))#shape (t_range, y_range)
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_model = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_model, py_zt_std_model)#shape (z_range, t_range)
        py_dot_model[:,y_index] = (py_zt_model * scipy.stats.norm.pdf(z_range_model[:,None])).sum(axis=0)*z_len
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len

    #The average distances between P_model(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_model - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    avg_causal_dist = (causal_dist * pt_true).sum()*t_len
    return avg_causal_dist, py_dot_model, py_dot_true, y_range, t_range, pt_true

def avg_causal_L1_dist_general(model, c_yt, c_yz, s_y, c_t, s_t, c_x, n=100, lim=6):
    """Calculates
    ∫|𝑃(𝑦|𝑑𝑜(𝑡),𝑃𝑡𝑟𝑢𝑒(𝑦|𝑑𝑜(𝑡))|𝐿1𝑃(𝑡)𝑑𝑡
    for the model with continuous t and y. Doesn't assume z_dim=1.
    TODO: Maybe this should have some kind error analysis"""
    #First calculate the P(t) function
    t_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/n
    y_len = 2*lim/n
    t_len = 2*lim/n
    z_dim = model.z_dim
    
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t
    
    #P(t)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len#shape (t_range,)
    
    #P(y|do(t)) for the model
    py_zt_std_model = torch.exp(model.decoder.y_log_std).detach().numpy()
    py_zt_std_true = s_y
    py_dot_model = np.zeros((n,n))#shape (t_range, y_range)
    for z in itertools.product(z_range, repeat=z_dim):
        zt_range = np.concatenate([np.tile(np.array(z),(n,1)),t_range[:,None]], axis=1)
        if model.decoder.p_y_zt_nn:
            if model.decoder.p_y_zt_std:
                py_zt_res_model = model.decoder.y_nn_real(torch.Tensor(zt_range)).detach().numpy()
                py_zt_mean_model = py_zt_res_model[:,0]
                py_zt_std_model = np.exp(py_zt_res_model[:,1][:,None])#Overwrites the constant std assumed above
            else:
                py_zt_mean_model = torch.reshape(model.decoder.y_nn_real(torch.Tensor(zt_range)),(n,)).detach().numpy()#shape (t_range,)
        else:
            py_zt_mean_model = torch.reshape(model.decoder.y_nn(torch.Tensor(zt_range)),(n,)).detach().numpy()#shape (z_range, 
        py_zt_model = scipy.stats.norm.pdf(y_range[None,:], py_zt_mean_model[:,None], py_zt_std_model)#shape (t_range, y_range)
        py_dot_model += py_zt_model * scipy.stats.norm.pdf(z).prod() * z_len**z_dim
    
    #P(y|do(t)) for the true distribution
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len
        
    #The average distances between P_model(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_model - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    avg_causal_dist = (causal_dist * pt_true).sum()*t_len
    return avg_causal_dist, py_dot_model, py_dot_true, y_range, t_range, pt_true

def run_model_for_data_sets(datasizes, datasize_times, num_epochs,
                            lr_start, lr_end, input_dim, z_dim, folder,name, BATCH_SIZE,
                            binary_t_y, p_y_zt_nn, q_z_xty_nn,
                            generate_df, dataparameters, track_function, true_value,
                            device='cpu', p_x_z_nn=False, p_t_z_nn=False, p_y_zt_std=False, p_x_z_std=False,
                            p_t_z_std=False, decoder_hidden_dim=3, decoder_num_hidden=3,
                            encoder_hidden_dim=4, encoder_num_hidden=3,):
    """Runs the model for different sample sizes multiple times for each size. Saves the results in data/{folder}.
    Currently just empties everything in the folder, but we also have the parameter 'name' to save the results of 
    different experiments in the same folder, if we happen to need that. """
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
    
    if not isinstance(num_epochs, list):
        num_epochs = datasize_times*[num_epochs]
    if not isinstance(lr_start, list):
        lr_start = datasize_times*[lr_start]
    if not isinstance(lr_end, list):
        lr_end = datasize_times*[lr_end]
    
    i,j = 0,0
    dfs = {datasize: {} for datasize in datasizes}
    models = {datasize: {} for datasize in datasizes}
    while i < len(datasizes):
        while j < datasize_times:
            num_samples = datasizes[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            df = generate_df(num_samples, *dataparameters)
            dataset = LinearDataset(df)
            dataloader = LinearDataLoader(dataset, validation_split=0.0)
            train_loader, test_loader = dataloader.get_loaders(batch_size=BATCH_SIZE)
            #dummy test loader
            test_loader, _ = LinearDataLoader(LinearDataset(df[:1]), validation_split=0.0).get_loaders(batch_size=1)
            #Running the model
            model = run_cevae(num_epochs=num_epochs[i], lr_start=lr_start[i], lr_end=lr_end[i],
                train_loader=train_loader, test_loader=test_loader, input_dim=input_dim, z_dim=z_dim,
                plot_curves=False, print_logs=False, device=device,
                binary_t_y=binary_t_y, p_y_zt_nn=p_y_zt_nn, q_z_xty_nn=q_z_xty_nn,
                p_x_z_nn = p_x_z_nn, p_t_z_nn = p_t_z_nn, p_y_zt_std = p_y_zt_std,
                p_x_z_std = p_x_z_std, p_t_z_std = p_t_z_std, decoder_hidden_dim=decoder_hidden_dim, 
                decoder_num_hidden=decoder_num_hidden, encoder_hidden_dim=encoder_hidden_dim, 
                encoder_num_hidden=encoder_num_hidden)

            dfs[num_samples][j] = df
            models[num_samples][j] = model

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,num_samples,j))
            df.to_pickle("./data/{}/data_{}_{}_{}".format(folder,name,num_samples,j))
            print("Estimated causal effect: {} true value: {}".format(track_function(model), true_value))
            j += 1
        j = 0
        i += 1
    return dfs, models

def load_dfs_models(folder, name,
                   input_dim, z_dim, device, binary_t_y,
                  p_y_zt_nn, q_z_xty_nn, 
                  decoder_hidden_dim=3, decoder_num_hidden=3,
                  encoder_hidden_dim=4, encoder_num_hidden=3,
                  p_x_z_nn=False, p_t_z_nn=False, p_y_zt_std=False, p_x_z_std=False,
                  p_t_z_std=False
                   ):
    """Loads dataframes and trained models from data/{folder}/ that match the experiment name"""
    dfs = {}
    models = {}
    for file in os.listdir("data/{}".format(folder)):
        match = re.search(r"([^_]*)_(.*)_(\d*)_(\d*)", file)
        if match.group(2) == name:
            if match.group(1) == "data":
                if not int(match.group(3)) in dfs:
                    dfs[int(match.group(3))] = {int(match.group(4)): pd.read_pickle("data/{}/{}".format(folder,file))}
                else:
                    dfs[int(match.group(3))][int(match.group(4))] = pd.read_pickle("data/{}/{}".format(folder,file))
            elif match.group(1) == "model":
                model = linearCEVAE(input_dim, z_dim, device=device, binary_t_y=binary_t_y, 
                  p_y_zt_nn=p_y_zt_nn, decoder_hidden_dim=decoder_hidden_dim, decoder_num_hidden=decoder_num_hidden, 
                  q_z_xty_nn=q_z_xty_nn, encoder_hidden_dim=encoder_hidden_dim, encoder_num_hidden=encoder_num_hidden,
                  p_x_z_nn=p_x_z_nn, p_t_z_nn=p_t_z_nn, p_y_zt_std=p_y_zt_std, p_x_z_std=p_x_z_std,p_t_z_std=p_t_z_std)
                model.load_state_dict(torch.load("data/{}/{}".format(folder,file)))
                model.eval()
                if not int(match.group(3)) in models:
                    models[int(match.group(3))] = {int(match.group(4)): model}
                else:
                    models[int(match.group(3))][int(match.group(4))] = model
    return dfs, models

def kld_loss(mu, std):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    var = std.pow(2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return kld

def run_cevae(num_epochs, lr_start, lr_end, train_loader, test_loader, input_dim, z_dim=1,
              device='cpu', plot_curves=True, print_logs=True, binary_t_y=False,
              p_y_zt_nn=False, decoder_hidden_dim=3, decoder_num_hidden=3,
              q_z_xty_nn=False, encoder_hidden_dim=4, encoder_num_hidden=3,
              p_x_z_nn = False, p_t_z_nn = False, p_y_zt_std = False,
              p_x_z_std = False, p_t_z_std = False
             ):
    model = linearCEVAE(input_dim, z_dim=z_dim, device=device, binary_t_y=binary_t_y, 
              p_y_zt_nn=p_y_zt_nn, decoder_hidden_dim=decoder_hidden_dim, decoder_num_hidden=decoder_num_hidden, 
              q_z_xty_nn=q_z_xty_nn, encoder_hidden_dim=encoder_hidden_dim, encoder_num_hidden=encoder_num_hidden,
              p_x_z_nn = p_x_z_nn, p_t_z_nn = p_t_z_nn, p_y_zt_std = p_y_zt_std, p_x_z_std = p_x_z_std,
              p_t_z_std = p_t_z_std)
    optimizer = Adam(model.parameters(), lr=lr_start)

    def _prepare_batch(batch):
        x = batch['X'].to(device)
        t = batch['t'].to(device)
        y = batch['y'].to(device)
        if len(x.shape) == 1:#Temporary, making sure that shapes are bs x dim
            x = x[:,None]
            y = y[:,None]
            t = t[:,None]
        return x, t, y
    
    def bernoulli_E_loss(z_probs, logits0, logits1, vals):
        return -(z_probs*dist.Bernoulli(logits=logits1).log_prob(vals) + 
                    (1-z_probs)*dist.Bernoulli(logits=logits0).log_prob(vals)).sum()

    def process_function(engine, batch):
        optimizer.zero_grad()
        model.train()
        x, t, y = _prepare_batch(batch)
        bs = x.shape[0]
        
        if len(x.shape) == 1:
            print("Shapes in process_function: {} {} {}".format(x.shape, t.shape, y.shape))

        z_mean, z_std, x_pred, t_pred, y_pred, x_std, t_std, y_std = model(x,t,y)
        
        
        kld = kld_loss(z_mean, z_std)
        if binary_t_y:
            t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
            y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
        else:
            t_loss = -dist.Normal(loc=t_pred, scale = t_std).log_prob(t).sum()
            y_loss = -dist.Normal(loc=y_pred, scale = y_std).log_prob(y).sum()
        x_loss = -dist.Normal(loc=x_pred, scale = x_std).log_prob(x).sum()

        tr_loss = x_loss + t_loss + y_loss + kld
        tr_loss.backward()
        optimizer.step()

        return (
            y_loss.item(),
            kld.item(),
            x_loss.item(),
            t_loss.item()
        )

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, t, y = _prepare_batch(batch)
            bs = x.shape[0]
            
            z_mean, z_std, x_pred, t_pred, y_pred, x_std, t_std, y_std = model(x,t,y)
            
            kld = kld_loss(z_mean, z_std)
            if binary_t_y:
                t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
                y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
            else:
                t_loss = -dist.Normal(loc=t_pred, scale = t_std).log_prob(t).sum()
                y_loss = -dist.Normal(loc=y_pred, scale = y_std).log_prob(y).sum()
            x_loss = -dist.Normal(loc=x_pred, scale = x_std).log_prob(x).sum()
            #Is the x loss OK if x has 2 dimensions?
            tr_loss = x_loss + t_loss + y_loss + kld
            
            #ITE inference
            E_y_x_do1, E_y_x_do0 = 0,0#model.evaluate_batch(x)
            ITE_x = E_y_x_do1 - E_y_x_do0
            
            return {
                "E_y_x_do1":E_y_x_do1,"E_y_x_do0":E_y_x_do0,"ITE_x":ITE_x,
                "y":y, "y_pred":y_pred,
                "x":x, "x_pred":x_pred,
                "t":t, "t_pred":t_pred,
                "z_mean":z_mean, "z_std":z_std,
                "total_loss":tr_loss,
                "t_loss":t_loss, "y_loss":y_loss, "x_loss":x_loss, "kld":kld
            }

    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)
    train_evaluator = Engine(evaluate_function)
    pbar = ProgressBar(persist=False)
    
    #Set up learning rate scheduling
    exp_scheduler = ExponentialLR(optimizer, gamma=(lr_end/lr_start)**(1/num_epochs))
    scheduler = LRScheduler(exp_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    eval_metrics = {"total_loss": ignite_metrics.Average(
            output_transform=lambda x: x['total_loss']),
                   "y_reconc_loss": ignite_metrics.Average(
            output_transform=lambda x: x['y_loss']),
                   "x_reconc_loss": ignite_metrics.Average(
            output_transform=lambda x: x['x_loss']),
                   "t_reconc_loss": ignite_metrics.Average(
            output_transform=lambda x: x['t_loss']),
                   "kld_loss": ignite_metrics.Average(
            output_transform=lambda x: x['kld'])}

    for eval_engine in [evaluator, train_evaluator]:
        for name, metric in eval_metrics.items():
            metric.attach(eval_engine, name)

    tr_metrics_history = {k: [] for k in eval_metrics.keys()}
    val_metrics_history = {k: [] for k in eval_metrics.keys()}

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        if device == "cuda":
            torch.cuda.synchronize()
        train_evaluator.run(train_loader)
        evaluator.run(test_loader)

    def handle_logs(evaluator, trainer, mode, metrics_history):
        metrics = evaluator.state.metrics
        for key, value in evaluator.state.metrics.items():
            metrics_history[key].append(value)
            
        lr = 0
        for param_group in optimizer.param_groups:
                lr = param_group['lr']

        print_str = f"{mode} Results - Epoch {trainer.state.epoch} - "+\
                    f"y_reconc_loss: {metrics['y_reconc_loss']:.4f} " +\
                    f"x_reconc_loss: {metrics['x_reconc_loss']:.4f} " +\
                    f"t_reconc_loss: {metrics['t_reconc_loss']:.4f} " +\
                    f"kld_loss: {metrics['kld_loss']:.4f} " +\
                    f"total_loss: {metrics['total_loss']:.4f} " +\
                    f"learning rate: {lr:.4f}"
        if print_logs:
            print(print_str)

    train_evaluator.add_event_handler(
        Events.COMPLETED,
        handle_logs,
        trainer,
        'Training',
        tr_metrics_history
    )

    trainer.run(train_loader, max_epochs=num_epochs)

    # Plot training curve
    if plot_curves:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(5, 2, figsize=(12, 12))
            sns.lineplot(x=range(num_epochs), y=tr_metrics_history['total_loss'], ax=ax[0, 0], err_style=None, label='training')
            ax[0, 0].set_xlabel('Epochs')
            ax[0, 0].set_ylabel('Total Loss')
            ax[0, 0].set_title('Total Loss')

            sns.lineplot(x=range(num_epochs), y=tr_metrics_history['y_reconc_loss'], ax=ax[0, 1], err_style=None, label='training')
            ax[0, 1].set_xlabel('Epochs')
            ax[0, 1].set_ylabel('y_reconc_loss')
            ax[0, 1].set_title('y_reconc_loss')

            sns.lineplot(x=range(num_epochs), y=tr_metrics_history['x_reconc_loss'], ax=ax[1, 0], err_style=None, label='training')
            ax[1, 0].set_xlabel('Epochs')
            ax[1, 0].set_ylabel('x_reconc_loss')
            ax[1, 0].set_title('X Reconstruction Loss')

            sns.lineplot(x=range(num_epochs), y=tr_metrics_history['t_reconc_loss'], ax=ax[1, 1], err_style=None, label='training')
            ax[1, 1].set_xlabel('Epochs')
            ax[1, 1].set_ylabel('t_reconc_loss')
            ax[1, 1].set_title('t_reconc_loss')

            sns.lineplot(x=range(num_epochs), y=tr_metrics_history['kld_loss'], ax=ax[2, 0], err_style=None, label='training')
            ax[2, 0].set_xlabel('Epochs')
            ax[2, 0].set_ylabel('kld_loss')
            ax[2, 0].set_title('kld_loss')
    return model