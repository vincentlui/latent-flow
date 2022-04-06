import os
from pathlib import Path

import numpy as np
import torch
import torchcde
import pandas as pd
from torchvision.datasets.utils import download_url
from lib.utils import normalize_data


DATA_DIR = Path('/opt/ml/input/data/training')
if not DATA_DIR.exists():
	DATA_DIR = Path(__file__).parents[1] / 'data/energy'

csv_file = 'energydata_complete.csv'

target_columns = ['Appliances', 'lights', 'T1']

class Energy(object):
    T = 6*6
    D = len(target_columns)

    training_file = 'training.npz'

    def __init__(self, root):
        self.root = root

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
        self._parse(self.data_folder)

        data_file = os.path.join(self.data_folder, self.training_file)

        # self.data = torch.Tensor(torch.load(data_file))
        # self.data, self.data_min, self.data_max = utils.normalize_data(self.data)

    # def _download(self):
    #     if self._check_exists():
    #         return
    #
    #     print('Downloading the dataset [325MB] ...')
    #     os.makedirs(self.data_folder, exist_ok=True)
    #     url = 'http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt'
    #     download_url(url, self.data_folder, 'training.pt', None)


    def _check_exists(self):
        return os.path.exists(os.path.join(self.data_folder, self.training_file))

    @property
    def data_folder(self):
        return DATA_DIR

    def get_dataset(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def size(self, ind=None):
        if ind is not None:
            return self.data.shape[ind]
        return self.data.shape

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def _parse(self, dir):
        df = pd.read_csv(dir / csv_file)
        data = df[target_columns].to_numpy()

        # We want that data start at 00:00:00
        # so we discard first few and last few samples
        # reshape to batch, timestep, dim
        data = data[42:19626].reshape(-1, self.T, self.D)
        sequences = data
        # sequences = normalize_data(torch.Tensor(data[..., 1:]))[0]

        # shuffle the data for training
        sequences = sequences[torch.randperm(data.shape[0])]

        times = np.arange(0, self.T)[None].repeat(sequences.shape[0], axis=0)[..., None] / self.T
        initial_values = sequences[:, :1]

        np.savez(dir / self.training_file, init=initial_values, seq=sequences, time=times)


if __name__ == '__main__':
    e = Energy(root=DATA_DIR)
