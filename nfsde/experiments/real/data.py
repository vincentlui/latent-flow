from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from nfsde.experiments.real.mujoco_physics import HopperPhysics
from nfsde.experiments.real.person_activity import PersonActivity, variable_time_collate_fn_activity
from nfsde.experiments.real.lib.parse_datasets import parse_datasets
import pandas as pd
from sklearn import model_selection
import torchcde

DATA_DIR = Path(__file__).parents[1] / 'data'

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
    loader = dict(np.load(dir / f"{name}/training.npz", allow_pickle=True))
    init = loader['init']
    time = loader['time']
    seq = loader['seq']
    if name in ['hopper2', 'energy', 'baqd2']:
        return TimeSeriesDataset(init, time, seq, normalize=True)
    return TimeSeriesDataset(init, time, seq)


def get_real_data_loaders(name, batch_size, dir=None, add_noise=False, train_size=0.6, val_size=0.2):
    if name == 'activity':
        n = 100
        n_samples = min(10000, n)
        dataset_obj = PersonActivity(DATA_DIR / 'activity', download=True, n_samples=n_samples,)

        train_data, test_data = model_selection.train_test_split(dataset_obj, train_size=0.8, shuffle=False)
        train_data, val_data = model_selection.train_test_split(dataset_obj, train_size=0.75, shuffle=False)

        input_dim = train_data[0][2].shape[-1]
        output_dim = train_data[0][-1].shape[-1]

        batch_size = min(min(len(dataset_obj), batch_size), n)
        dltrain = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: variable_time_collate_fn_activity(batch, None, None,
                                                                                        data_type='train'))
        dlval = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda batch: variable_time_collate_fn_activity(batch, None, None,
                                                                                      data_type='test'))
        dltest = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda batch: variable_time_collate_fn_activity(batch, None, None,
                                                                                       data_type='test'))
        return input_dim, 0, dltrain, dlval, dltest, None, None, #True
    else:
    # Returns input_dim, n_classes=None, 3*torch.utils.data.DataLoader
        trainset, valset, testset = load_dataset(name, dir, add_noise).split_train_val_test(train_size, val_size)
        dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(valset, batch_size=batch_size, shuffle=False)
        dl_test = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainset.dim, 0, dl_train, dl_val, dl_test, trainset.mean, trainset.std, #False

def get_single_loader(name, batch_size):
    dataset = load_dataset(name)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dl

class TimeSeriesDataset(Dataset):
    def __init__(self, initial, times, values, labels=None, normalize=False, add_noise=False, noise_std=0.01, interpolate=True):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.labels = None
        if isinstance(initial, torch.Tensor):
            self.initial = initial
            self.times = times
            self.values = values
            if labels is not None:
                self.labels = labels
        else:
            self.initial = torch.Tensor(initial)
            self.times = torch.Tensor(times)
            self.values = torch.Tensor(values)
            if labels is not None:
                self.labels = torch.Tensor(labels)
        if interpolate:
            self.values = torchcde.linear_interpolation_coeffs(self.values)
            self.initial = self.values[:, None, 0]
        self.mean, self.std = None, None
        self.normalize = normalize
        if normalize:
            self.mean, self.std = calc_mean_and_std(self.values)
        self.add_noise = add_noise
        self.noise_std = noise_std

    def split_train_val_test(self, train_size=0.6, val_size=0.2):
        ind1 = int(len(self.initial) * train_size)
        ind2 = ind1 + int(len(self.initial) * val_size)

        if self.labels is not None:
            trainset = TimeSeriesDataset(self.initial[:ind1], self.times[:ind1], self.values[:ind1], self.labels[:ind1], normalize=True)
            valset = TimeSeriesDataset(self.initial[ind1:ind2], self.times[ind1:ind2], self.values[ind1:ind2], self.labels[ind1:ind2])
            testset = TimeSeriesDataset(self.initial[ind2:], self.times[ind2:], self.values[ind2:], self.labels[ind2:])
        else:
            trainset = TimeSeriesDataset(self.initial[:ind1], self.times[:ind1], self.values[:ind1],
                                         normalize=self.normalize)
            valset = TimeSeriesDataset(self.initial[ind1:ind2], self.times[ind1:ind2],
                                       self.values[ind1:ind2])
            testset = TimeSeriesDataset(self.initial[ind2:], self.times[ind2:], self.values[ind2:])
        if self.normalize:
            valset.set_mean_and_std(trainset.mean, trainset.std)
            testset.set_mean_and_std(trainset.mean, trainset.std)

        return trainset, valset, testset

    def set_mean_and_std(self, mean, std):
        self.mean = mean
        self.std = std

    @property
    def dim(self):
        return self.values[0].shape[-1]

    @property
    def num_class(self):
        if self.labels is not None:
            return self.labels[0].shape[-1]
        return 0

    def __getitem__(self, key):
        initial = self.initial[key]
        values = self.values[key]
        if self.mean is not None and self.std is not None:
            initial = (self.initial[key] - self.mean) / self.std
            values = (self.values[key] - self.mean) / self.std
        if self.labels is not None:
            initial = torch.cat([initial, self.labels[key]], dim=-1)
        if self.add_noise:
            noise = torch.randn_like(values) * self.noise_std
            values += noise
        return initial, self.times[key], values

    def __len__(self):
        return len(self.initial)

    def __repr__(self):
        return f'TimeSeriesDataset({self.__len__()})'


def calc_mean_and_std(x):
    mean = torch.mean(x, dim=(0,1))
    std = torch.std(x,dim=(0,1))

    return mean, std

def get_max(x):
    max = torch.max(x, dim=(0,1))

    return max

# def parse(name, dir):
#     if name == 'baqd':
#         files = []
#         for file in csv_files:
#             csv_dir = dir / file
#             files.append(pd.read_csv(csv_dir))
#         df = pd.concat(files, ignore_index=True)
#         i = 0
#         for s in stations:
#             df.loc[df['station'] == s, 'station'] = i
#             i += 1
#         data = df[target_columns].to_numpy()
#         one_hot_label = np.zeros((len(data), data[:, 0].max() + 1))
#         one_hot_label[np.arange(len(data)), data[:, 0].astype(int)] = 1
#         data = torchcde.linear_interpolation_coeffs(torch.Tensor(data[:, 1:].astype(float))).numpy()
#
#         one_hot_label = one_hot_label.reshape(-1, 24, len(stations))[:, :1].astype(float)
#         time = data[:, 0].reshape(-1, 24, 1)
#         seq = data[:, 1:].reshape(-1, 24, 1)
#         init = seq[:, :1]
#
#     elif name == 'hopper':
#         h = HopperPhysics(root='nfsde/experiments/data', download=False)
#         dataset = h.get_dataset()
#         n, num_obs, obs_dim = dataset.shape
#         time = torch.linspace(0, 1, num_obs).unsqueeze(0).unsqueeze(-1).repeat_interleave(n, dim=0)
#         init = dataset[:, :1]
#         seq = dataset
#
#     return init, time, seq #, one_hot_label
