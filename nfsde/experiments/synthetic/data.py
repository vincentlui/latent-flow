from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchcde

from nfsde.experiments.synthetic.generate import generate, DATA_DIR


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.npz'
    file_list = [x.stem for x in DATA_DIR.iterdir() if check(x)]
    file_list += [x.stem for x in (DATA_DIR).iterdir() if check(x)]
    return sorted(file_list)

def load_dataset(name, dir=None, add_noise=False):
    if dir is None:
        dir = DATA_DIR
    else:
        dir = Path(dir)
    generate(dir)
    if not name.endswith('.npz'):
        name += '.npz'
    loader = dict(np.load(dir / name, allow_pickle=True))
    return TimeSeriesDataset(loader['init'][:,None], loader['time'][...,None], loader['seq'], add_noise=add_noise)

def get_data_loaders(name, batch_size, dir=None, add_noise=False, train_size=0.6, val_size=0.2):
    # Returns input_dim, n_classes=None, 3*torch.utils.data.DataLoader
    trainset, valset, testset = load_dataset(name, dir, add_noise).split_train_val_test(train_size, val_size)
    dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(valset, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainset.dim, 0, dl_train, dl_val, dl_test, trainset.mean, trainset.std

def get_single_loader(name, batch_size):
    dataset = load_dataset(name)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dl

class TimeSeriesDataset(Dataset):
    def __init__(self, initial, times, values, normalize=False, add_noise=False, noise_std=0.01, interpolate=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if isinstance(initial, torch.Tensor):
            self.initial = initial
            self.times = times
            self.values = values
        else:
            self.initial = torch.Tensor(initial)#.to(device)
            self.times = torch.Tensor(times)#.to(device)
            self.values = torch.Tensor(values)#.to(device)
        if interpolate:
            self.values = torchcde.linear_interpolation_coeffs(self.values)
            self.initial = self.values[:, None, 0]
        self.mean, self.std = None, None
        if normalize:
            self.mean, self.std = calc_mean_and_std(values)
        self.add_noise = add_noise
        self.noise_std = noise_std


    def split_train_val_test(self, train_size=0.6, val_size=0.2):
        ind1 = int(len(self.initial) * train_size)
        ind2 = ind1 + int(len(self.initial) * val_size)

        trainset = TimeSeriesDataset(self.initial[:ind1], self.times[:ind1], self.values[:ind1], normalize=True,
                                     add_noise=self.add_noise, noise_std=self.noise_std)
        valset = TimeSeriesDataset(self.initial[ind1:ind2], self.times[ind1:ind2], self.values[ind1:ind2])
        valset.set_mean_and_std(trainset.mean, trainset.std)
        testset = TimeSeriesDataset(self.initial[ind2:], self.times[ind2:], self.values[ind2:])
        testset.set_mean_and_std(trainset.mean, trainset.std)

        return trainset, valset, testset

    def set_mean_and_std(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def dim(self):
        return self.values[0].shape[-1]

    def __getitem__(self, key):
        if self.mean is not None and self.std is not None:
            initial = (self.initial[key] - self.mean) / self.std
            values = (self.values[key] - self.mean) / self.std

        if self.add_noise:
            noise = torch.randn_like(values) * self.noise_std
            values += noise
        return initial, self.times[key], values

    def __len__(self):
        return len(self.initial)

    def __repr__(self):
        return f'TimeSeriesDataset({self.__len__()})'


def calc_mean_and_std(x):
    mean = torch.mean(x[~torch.isnan(x)])
    std = torch.std(x[~torch.isnan(x)])

    return mean, std