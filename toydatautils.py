import pandas as pd
import numpy as np
import math, random
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import stats
import seaborn as sns

PLOT_STYLE = 'ggplot'

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def rounded_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.sigmoid(logits))

def true_z_given_ty(a,b):
    TAU = np.zeros((2, 2))
    for z in range(2):
        for t in range(2):
            TAU[z, t] = sigmoid(a * (z + b * (2 * t - 1)))
    return TAU

def prepare_data_df(NUM_SAMPLES, sig_z0, sig_z1, P0=0.5, T_TAU_Z0=0.75, T_TAU_Z1=0.25, a=3, b=2) -> pd.DataFrame:
    """Prepared data sets using different model parameters. a and b correspond to the parameters 
    of the y|z,t distribution Bern(Sigmoid(a(z + b(2t - 1)))))"""
    # First sample from Bernoulli for z_i
    z_dist = dist.Bernoulli(probs=P0)
    zs = z_dist.sample((NUM_SAMPLES, ))
    print(f"Sanity checking z_dist:\n{zs.mean():.3f}")
    df = pd.DataFrame(zs.numpy(), columns=['z'])
    
    TAU = true_z_given_ty(a,b)

    # Next sampling t_i and x_i from z_i
    x_s, t_s = [], []
    for idx in df.index:
        z_i = df.loc[idx, 'z']
        x_var = (np.square(sig_z1) * z_i + np.square(sig_z0) * (1 - z_i))
        x_sample = dist.Normal(
            loc=z_i, 
            scale=np.sqrt(x_var)
        ).sample()
        x_s.append(x_sample.item())
        
        t_sample = dist.Bernoulli(T_TAU_Z0 * z_i + T_TAU_Z1 * (1 - z_i)).sample()
        t_s.append(t_sample.item())

    df['X'] = x_s
    df['t'] = t_s

    # Lastly sample y_i conditioned on z_i and t_i
    y0s, y1s = [], []
    for idx in df.index:
        z_i = df.loc[idx, 'z']
        t_i = df.loc[idx, 't']

        # outcome when t = 0
        y0_sample = dist.Bernoulli(TAU[int(z_i), 0]).sample()
        y0s.append(y0_sample.item())

        # outcome when t = 1
        y1_sample = dist.Bernoulli(TAU[int(z_i), 1]).sample()
        y1s.append(y1_sample.item())
        
    df['y0'] = y0s # untreated outcome for given z
    df['y1'] = y1s # treated outcome for given z
    df['yf'] =  np.where(df['t'] == 0, df['y0'], df['y1'])

    print(f"t value  counts:\n{df['t'].value_counts()}")
    print(f"y0 value counts:\n{df['y0'].value_counts()}")
    print(f"y1 value  counts:\n{df['y1'].value_counts()}")

    return df

def visualize_data(df: pd.DataFrame, a, b) -> None:
    # Visualize the generated data
    
    TAU = true_z_given_ty(a,b)
    
    with plt.style.context(PLOT_STYLE):
        kwargs = dict(bins=50, )
        fig, ax = plt.subplots(3, 2, figsize=(10, 15))
        sns.distplot(df.query("z == 0")['X'], ax=ax[0, 0], label='z = 0', **kwargs)
        sns.distplot(df.query("z == 1")['X'], ax=ax[0, 0], label='z = 1', **kwargs)
        ax[0, 0].axvline(df.query("z == 0")['X'].mean(), color="gray", linestyle="dotted")
        ax[0, 0].axvline(df.query("z == 1")['X'].mean(), color="gray", linestyle="dotted")
        ax[0, 0].set_xlabel('X')
        ax[0, 0].set_title('p(X|z)')
        ax[0, 0].legend()


        sns.scatterplot(x=df.query("z == 0 & t == 0")['X'], y=TAU[0, 0], ax=ax[0, 1], label='z = 0, t = 0')
        sns.scatterplot(x=df.query("z == 1 & t == 0")['X'], y=TAU[1, 0], ax=ax[0, 1], label='z = 1, t = 0')
        sns.scatterplot(x=df.query("z == 0 & t == 1")['X'], y=TAU[0, 1], ax=ax[0, 1], label='z = 0, t = 1')
        sns.scatterplot(x=df.query("z == 1 & t == 1")['X'], y=TAU[1, 1], ax=ax[0, 1], label='z = 1, t = 1')
        ax[0, 1].set_xlabel('X')
        ax[0, 1].set_ylabel('prob')
        ax[0, 1].set_title('p(y=1|z, t)')
        ax[0, 1].legend()

        sns.distplot(df.query("z == 0 & t == 0")['X'], ax=ax[1, 0], label='z = 0', **kwargs)
        sns.distplot(df.query("z == 1 & t == 0")['X'], ax=ax[1, 0], label='z = 1', **kwargs)
        ax[1, 0].axvline(df.query("z == 0 & t == 0")['X'].mean(), color="gray", linestyle="dotted")
        ax[1, 0].axvline(df.query("z == 1 & t == 0")['X'].mean(), color="gray", linestyle="dotted")
        ax[1, 0].set_xlabel('X')
        ax[1, 0].set_title('Untreated p(X|z)')
        ax[1, 0].legend()

        sns.distplot(df.query("z == 0 & t == 1")['X'], ax=ax[1, 1], label='z = 0', **kwargs)
        sns.distplot(df.query("z == 1 & t == 1")['X'], ax=ax[1, 1], label='z = 1', **kwargs)
        ax[1, 1].axvline(df.query("z == 0 & t == 1")['X'].mean(), color="gray", linestyle="dotted")
        ax[1, 1].axvline(df.query("z == 1 & t == 1")['X'].mean(), color="gray", linestyle="dotted")
        ax[1, 1].set_xlabel('X')
        ax[1, 1].set_title('Treated p(X|z)')
        ax[1, 1].legend()

        sns.scatterplot(x=df.query("z == 0 & t == 0")['X'], y=0.25, ax=ax[2, 0], label='z = 0, t = 0', alpha=0.7)
        sns.scatterplot(x=df.query("z == 0 & t == 1")['X'], y=0.25, ax=ax[2, 0], label='z = 0, t = 1', alpha=0.7)
        sns.scatterplot(x=df.query("z == 1 & t == 0")['X'], y=0.75, ax=ax[2, 0], label='z = 1, t = 0', alpha=0.7)
        sns.scatterplot(x=df.query("z == 1 & t == 1")['X'], y=0.75, ax=ax[2, 0], label='z = 1, t = 1', alpha=0.7)
        ax[2, 0].set_xlabel('X')
        ax[2, 0].set_title('Propensity p(t|z)')
        ax[2, 0].legend();


        sns.distplot(df.query("t == 0")['X'], ax=ax[2, 1], label='t = 0')
        sns.distplot(df.query("t == 1")['X'], ax=ax[2, 1], label='t = 1')
        ax[2, 1].set_xlabel('X')
        ax[2, 1].set_title('Propensity p(t|X)')
        ax[2, 1].legend();        

        return

def visualize_pdfs(P0, T_TAU_Z0, T_TAU_Z1, SIG_Z0, SIG_Z1, a, b):
    x_grid = np.linspace(-15, 15, 100)
    norm_z0 = stats.norm(0, SIG_Z0) # cluster 1
    norm_z1 = stats.norm(1, SIG_Z1) # cluster 2
    
    TAU = true_z_given_ty(a,b)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # p(z=1|X) ~ p(x|z=1) p(z=1)
        
        aux_z0 = P0 * norm_z1.pdf(x_grid)
        aux_z1 = (1 - P0) * norm_z0.pdf(x_grid)
        curve = aux_z0 / (aux_z0 + aux_z1)
        sns.lineplot(x_grid, curve, ax=ax[0, 0])
        ax[0, 0].set_xlabel('X')
        ax[0, 0].set_ylabel('prob')
        ax[0, 0].set_title('p(z=1|X)')

        # p(z=1|X, t) ~ p(X|z=1) p(t|z=1) p(z=1)
        aux_z0 = P0 * T_TAU_Z0 * norm_z1.pdf(x_grid)
        aux_z1 = (1 - P0) * T_TAU_Z1 * norm_z0.pdf(x_grid)
        curve = aux_z0 / (aux_z0 + aux_z1)
        sns.lineplot(x_grid, curve, ax=ax[0, 1], label='t = 1')

        aux_z0 = P0 * (1 - T_TAU_Z0) * norm_z1.pdf(x_grid)
        aux_z1 = (1 - P0) * (1 - T_TAU_Z1) * norm_z0.pdf(x_grid)
        curve = aux_z0 / (aux_z0 + aux_z1)
        sns.lineplot(x_grid, curve, ax=ax[0, 1], label='t = 0')

        ax[0, 1].set_xlabel('X')
        ax[0, 1].set_ylabel('prob')
        ax[0, 1].set_title('p(z=1|X, t)')

        # p(y=1|X, t) = 
        #   p(y=1|t, z=1) p(z=1|X, t) +
        #   p(y=1|t, z=0) p(z=0|X, t) 
        # = 
        #   p(y=1|t, z=1) p(z=1|X, t) + 
        #   p(y=1|t, z=0) (1 - p(z=1|X, t))

        aux_z0 = P0 * T_TAU_Z0 * norm_z1.pdf(x_grid)
        aux_z1 = (1 - P0) * T_TAU_Z1 * norm_z0.pdf(x_grid)
        pos_z = aux_z0 / (aux_z0 + aux_z1) # p(z=1|x, t=1)
        # TAU coords = 
        # z0_t0  z0_t1
        # z1_t0  z1_t1
        curve = TAU[1, 1] * pos_z + TAU[0, 1] * (1 - pos_z)
        sns.lineplot(x_grid, curve, ax=ax[1, 0], label='t = 1')

        aux_z0 = P0 * (1 - T_TAU_Z0) * norm_z1.pdf(x_grid)
        aux_z1 = (1 - P0) * (1 - T_TAU_Z1) * norm_z0.pdf(x_grid)
        pos_z = aux_z0 / (aux_z0 + aux_z1) # p(z=1|x, t=0)
        # TAU coords = 
        # z0_t0  z0_t1
        # z1_t0  z1_t1
        curve = TAU[1, 0] * pos_z + TAU[0, 0] * (1 - pos_z)
        sns.lineplot(x_grid, curve, ax=ax[1, 0], label='t = 0')
        ax[1, 0].set_xlabel('X')
        ax[1, 0].set_ylabel('prob')
        ax[1, 0].set_title('p(y=1|X, t)')

        # p(y|X, do(t)) = 
        #   p(y|X, do(t), z=0) p(z=0|X, do(t)) +
        #   p(y|X, do(t), z=1) p(z=1|X, do(t))
        # =
        #   p(y|X, t, z=0) p(z=0|X, t) +
        #   p(y|X, t, z=1) p(z=1|X, t)
        aux_z0 = P0 * norm_z0.pdf(x_grid)
        aux_z1 = (1 - P0) * norm_z1.pdf(x_grid)
        pos_z0 = aux_z0 / (aux_z0 + aux_z1) # p(z=0|X)
        # TAU coords = 
        # z0_t0  z0_t1
        # z1_t0  z1_t1
        curve = TAU[0, 0] * pos_z0 + TAU[1, 0] * (1 - pos_z0)
        sns.lineplot(x_grid, curve, ax=ax[1, 1], label='do(t = 0)')

        curve = TAU[0, 1] * pos_z0 + TAU[1, 1] * (1 - pos_z0)
        sns.lineplot(x_grid, curve, ax=ax[1, 1], label='do(t = 1)')
        ax[1, 1].set_xlabel('X')
        ax[1, 1].set_ylabel('prob')
        ax[1, 1].set_title('p(y=1|X, do(t))')

# Define pytorch datasets and loaders
class ToyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.length = data.shape[0]
        self.t = data.loc[:, 't'].values
        self.X = data.loc[:, 'X'].values
        self.y0 = data.loc[:, 'y0'].values
        self.y1 = data.loc[:, 'y1'].values
        self.yf = data.loc[:, 'yf'].values

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            't': self.t[idx],
            'y0': self.y0[idx],
            'y1': self.y1[idx],
            'yf': self.yf[idx]
        }

    def __len__(self):
        return self.length

class ToyDataLoader(DataLoader):
    def __init__(self, dataset, validation_split=0.2, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, valid_indices = indices[split:], indices[: split]

        self.dataset = dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(valid_indices)

    def collate_fn(self, batch):
        keys = list(batch[0].keys())
        processed_batch = {k: [] for k in keys}
        for _, sample in enumerate(batch):
            for key, value in sample.items():
                processed_batch[key].append(value)
        
        processed_batch['t'] = torch.FloatTensor(processed_batch['t'])
        processed_batch['X'] = torch.FloatTensor(processed_batch['X'])
        processed_batch['y0'] = torch.FloatTensor(processed_batch['y0'])
        processed_batch['y1'] = torch.FloatTensor(processed_batch['y1'])
        processed_batch['yf'] = torch.FloatTensor(processed_batch['yf'])
        return processed_batch

    def train_loader(self, batch_size, num_workers=0):
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader

    def test_loader(self, batch_size, num_workers=0):
        test_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            sampler=self.valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True
        )

        return test_loader

    def get_loaders(self, batch_size):
        train_loader = self.train_loader(batch_size)
        test_loader = self.test_loader(batch_size)

        return train_loader, test_loader

