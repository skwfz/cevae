from lineardatamodels import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
PLOT_STYLE='ggplot'

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
              device='cpu', plot_curves=True, print_logs=True):
    model = linearCEVAE(input_dim, device=device)
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

        z_mean, z_std, x_pred, t_pred, y_pred = model(x,t,y)
            
        kld = kld_loss(z_mean, z_std)
        t_loss = -dist.Normal(loc=t_pred, scale = torch.exp(model.decoder.t_log_std).repeat((bs, 1))).log_prob(t).sum()
        y_loss = -dist.Normal(loc=y_pred, scale = torch.exp(model.decoder.y_log_std).repeat((bs, 1))).log_prob(y).sum()
        x_loss = -dist.Normal(loc=x_pred, scale = torch.exp(model.decoder.x_log_std).repeat((bs, 1))).log_prob(x).sum()
        #Is the x loss OK if x has 2 dimensions?

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
            
            z_mean, z_std, x_pred, t_pred, y_pred = model(x,t,y)
            
            kld = kld_loss(z_mean, z_std)
            t_loss = -dist.Normal(loc=t_pred, scale = torch.exp(model.decoder.t_log_std).repeat((bs, 1))).log_prob(t).sum()
            y_loss = -dist.Normal(loc=y_pred, scale = torch.exp(model.decoder.y_log_std).repeat((bs, 1))).log_prob(y).sum()
            x_loss = -dist.Normal(loc=x_pred, scale = torch.exp(model.decoder.x_log_std).repeat((bs, 1))).log_prob(x).sum()
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