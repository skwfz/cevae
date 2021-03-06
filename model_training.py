import os
import math, random
import scipy.stats as stats
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

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

from models import *
from binarytoydata import *

def kld_loss(mu, var, kld_coef=1.):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)#Need to check that this is right, especially with logvar being a var
    return kld_coef * kld

def kld_loss_bernoulli(logits):
    # Assumes that the prior p(z=1) = 0.5
    probs = torch.sigmoid(logits)
    #kld = -(torch.log(probs) + torch.log(1-probs) - 2*np.log(0.5)).sum()
    kld = (probs*torch.log(2*probs) + (1-probs)*torch.log(2*(1-probs))).sum()
    return kld

def x_reconc_loss(mean, var, x):
    return -dist.normal.Normal(mean,torch.sqrt(var)).log_prob(x).sum()

def run_cevae(num_epochs, lr_start, lr_end, train_loader, test_loader, input_dim, z_dim, z_mode='normal',
              x_mode='normal', device='cpu',encoder_hidden_dim=4, decoder_hidden_dim=4, num_hidden=3):
    model = CEVAE(input_dim,encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim,
                  num_hidden=num_hidden, activation=nn.ELU(), z_dim=z_dim, device=device, z_mode=z_mode, x_mode=x_mode)
    optimizer = Adam(model.parameters(), lr=lr_start)

    #Loss functions needed: x,t and y reconstruction losses, kld loss and t & y infer losses

    #TODO: Check that these are all ok, should maybe just create the dists and get the log_prob from them
    yf_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')#t and y should in principle form one classification problem
    t_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')#Sum because kld_loss assumes sum
    #t_aux_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')#This could be mean since it doesn't couple with the VAE
    #y_aux_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    x_binary_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def _prepare_batch(batch):
        x = batch['X'].to(device)
        t = batch['t'].to(device)
        yf = batch['yf'].to(device)
        if len(x.shape) == 1:#Temporary, making sure that shapes are bs x dim
            x = x[:,None]
            yf = yf[:,None]
            t = t[:,None]
        return x, t, yf

    def process_function(engine, batch):
        optimizer.zero_grad()
        model.train()
        x, t, yf = _prepare_batch(batch)
        
        if len(x.shape) == 1:
            print("Shapes in process_function: {} {} {}".format(x.shape, t.shape, y.shape))
        (z_mean, z_var, z_logits, x_mean, x_var, x_logits, t_logits, 
            y_logits) = model(x,t,yf)
        
        #mse_loss = reconc_loss_fn(x_reconc, x) #These will match with the new losses
        #Alternative: log(sqrt(2 pi x_var)) + (x-x_mean)/(2*x_var)
        if z_mode == 'normal':
            kld = kld_loss(z_mean, z_var)
        elif z_mode == 'binary':
            kld = kld_loss_bernoulli(z_logits)
        if x_mode == 'normal':#Does this work for input_dim > 1?
            x_loss = -dist.normal.Normal(x_mean,torch.sqrt(x_var)).log_prob(x).sum()
        elif x_mode == 'binary':
            x_loss = x_binary_loss_fn(x_logits, x)
        t_loss = t_loss_fn(t_logits, t)#Let's hope that this is correct (first logit, then actual t)
        yf_loss = yf_loss_fn(y_logits, yf)

        # aux losses
        #t_aux_loss = t_aux_loss_fn(tinfer_logit, t)
        #y_aux_loss = y_aux_loss_fn(yinfer_logit, yf)

        tr_loss = x_loss + t_loss + yf_loss + kld# + t_aux_loss + y_aux_loss
        tr_loss.backward()
        optimizer.step()

        return (
            yf_loss.item(),
            kld.item(),
            x_loss.item(),
            t_loss.item()
        )

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, t, yf = _prepare_batch(batch)
            #How should the model be operated during test time? Technically we're not allowed to use
            #t or yf. 
            (z_mean, z_var, z_logits, x_mean, x_var, x_logits, t_logits, 
                y_logits) = model(x,t,yf)#Do we need this for anything?
            
            #ITE inference
            E_y_x_do1, E_y_x_do0 = 0,0#model.evaluate_batch(x)
            ITE_x = E_y_x_do1 - E_y_x_do0
            
            return {
                "E_y_x_do1":E_y_x_do1,"E_y_x_do0":E_y_x_do0,"ITE_x":ITE_x,
                "y1":batch['y1'],"y0":batch['y0'],
                "y_logits":y_logits, "yf":yf,
                "x_mean":x_mean, "x_var":x_var, "x_logits":x_logits,"x":x,
                "t_logits":t_logits, "t":t,
                "z_mean":z_mean, "z_var":z_var, "z_logits":z_logits
            }

    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)
    train_evaluator = Engine(evaluate_function)
    pbar = ProgressBar(persist=False)
    
    #Set up learning rate scheduling
    exp_scheduler = ExponentialLR(optimizer, gamma=(lr_end/lr_start)**(1/num_epochs))
    scheduler = LRScheduler(exp_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # eval_metrics
    eval_metrics = {
        # PEHE is defined as ((y1 - y0)_true - (y1 - y0)_pred)**2 .mean() for each patient
        "pehe": ignite_metrics.Average(
           output_transform=lambda x: (x["ITE_x"] - (x["y1"] - x["y0"])).pow(2).mean()),
        "mae_ate": ignite_metrics.Average(
           output_transform=lambda x: ((x["ITE_x"] - (x["y1"] - x["y0"])).mean())),
        "y_reconc_loss": ignite_metrics.Loss(
            yf_loss_fn, output_transform=lambda x: [x["y_logits"], x["yf"]]),
        "t_reconc_loss": ignite_metrics.Loss(
            t_loss_fn, output_transform=lambda x: [x["t_logits"], x["t"]])
    }
    if x_mode == 'normal':
        eval_metrics['x_reconc_loss'] = ignite_metrics.Average(
            output_transform=lambda x: x_reconc_loss(x["x_mean"], x["x_var"], x["x"]))
    elif x_mode == 'binary':
        eval_metrics['x_reconc_loss'] = ignite_metrics.Average(
            output_transform=lambda x: x_binary_loss_fn(x["x_logits"], x["x"]))
    if z_mode == 'normal':
        eval_metrics['kld_loss'] = ignite_metrics.Loss(
            kld_loss, output_transform=lambda x: [x["z_mean"], x["z_var"]])
    elif z_mode == 'binary':
        eval_metrics['kld_loss'] = ignite_metrics.Average(
            output_transform=lambda x: kld_loss_bernoulli(x["z_logits"]))
    
    eval_metrics['total_loss'] = (eval_metrics["y_reconc_loss"] + 
                                  eval_metrics["x_reconc_loss"] + 
                                  eval_metrics["t_reconc_loss"] + 
                                  eval_metrics["kld_loss"])

    for eval_engine in [evaluator, train_evaluator]:#What is this about? Something to do with the continuous tracking of losses?
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
        # import pdb; pdb.set_trace()
        for key, value in evaluator.state.metrics.items():
            metrics_history[key].append(value)
            
        lr = 0
        for param_group in optimizer.param_groups:
                lr = param_group['lr']

        print_str = f"{mode} Results - Epoch {trainer.state.epoch} - "+\
                    f"PEHE: {metrics['pehe']:.4f} " +\
                    f"MAE ATE: {metrics['mae_ate']:.4f} " +\
                    f"y_reconc_loss: {metrics['y_reconc_loss']:.4f} " +\
                    f"x_reconc_loss: {metrics['x_reconc_loss']:.4f} " +\
                    f"t_reconc_loss: {metrics['t_reconc_loss']:.4f} " +\
                    f"kld_loss: {metrics['kld_loss']:.4f} " +\
                    f"total_loss: {metrics['total_loss']:.4f} " +\
                    f"learning rate: {lr:.4f}"
        print(print_str)

    train_evaluator.add_event_handler(
        Events.COMPLETED,
        handle_logs,
        trainer,
        'Training',
        tr_metrics_history
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        handle_logs,
        trainer,
        'Validate',
        val_metrics_history
    )

    trainer.run(train_loader, max_epochs=num_epochs)

    # Plot training curve
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(5, 2, figsize=(12, 12))
        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['total_loss'], ax=ax[0, 0], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['total_loss'], ax=ax[0, 0], err_style=None, label='validation')
        ax[0, 0].set_xlabel('Epochs')
        ax[0, 0].set_ylabel('Total Loss')
        ax[0, 0].set_title('Total Loss')

        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['y_reconc_loss'], ax=ax[0, 1], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['y_reconc_loss'], ax=ax[0, 1], err_style=None, label='validation')
        ax[0, 1].set_xlabel('Epochs')
        ax[0, 1].set_ylabel('y_reconc_loss')
        ax[0, 1].set_title('y_reconc_loss')

        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['x_reconc_loss'], ax=ax[1, 0], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['x_reconc_loss'], ax=ax[1, 0], err_style=None, label='validation')
        ax[1, 0].set_xlabel('Epochs')
        ax[1, 0].set_ylabel('x_reconc_loss')
        ax[1, 0].set_title('X Reconstruction Loss')

        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['t_reconc_loss'], ax=ax[1, 1], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['t_reconc_loss'], ax=ax[1, 1], err_style=None, label='validation')
        ax[1, 1].set_xlabel('Epochs')
        ax[1, 1].set_ylabel('t_reconc_loss')
        ax[1, 1].set_title('t_reconc_loss')

        #fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['pehe'], ax=ax[2, 0], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['pehe'], ax=ax[2, 0], err_style=None, label='validation')
        ax[2, 0].set_xlabel('Epochs')
        ax[2, 0].set_ylabel('PEHE')
        ax[2, 0].set_title('PEHE')

        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['mae_ate'], ax=ax[2, 1], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['mae_ate'], ax=ax[2, 1], err_style=None, label='validation')
        ax[2, 1].set_xlabel('Epochs')
        ax[2, 1].set_ylabel('MAE ATE')
        ax[2, 1].set_title('ATE')
        
        sns.lineplot(x=range(num_epochs), y=tr_metrics_history['kld_loss'], ax=ax[4, 0], err_style=None, label='training')
        sns.lineplot(x=range(num_epochs), y=val_metrics_history['kld_loss'], ax=ax[4, 0], err_style=None, label='validation')
        ax[4, 0].set_xlabel('Epochs')
        ax[4, 0].set_ylabel('kld_loss')
        ax[4, 0].set_title('kld_loss')
    return model