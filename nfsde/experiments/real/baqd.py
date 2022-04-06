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
	DATA_DIR = Path(__file__).parents[1] / 'data/baqd'

csv_files = [
    'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    'PRSA_Data_Changping_20130301-20170228.csv',
    'PRSA_Data_Dingling_20130301-20170228.csv',
    'PRSA_Data_Dongsi_20130301-20170228.csv',
    'PRSA_Data_Guanyuan_20130301-20170228.csv',
    'PRSA_Data_Gucheng_20130301-20170228.csv',
    'PRSA_Data_Huairou_20130301-20170228.csv',
    'PRSA_Data_Nongzhanguan_20130301-20170228.csv',
    'PRSA_Data_Shunyi_20130301-20170228.csv',
    'PRSA_Data_Tiantan_20130301-20170228.csv',
    'PRSA_Data_Wanliu_20130301-20170228.csv',
    'PRSA_Data_Wanshouxigong_20130301-20170228.csv'
]
stations = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan',
       'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan',
       'Wanliu', 'Wanshouxigong']
target_columns = ['station', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

class BAQD(object):
    T = 24
    D = len(target_columns) - 2

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
        files = []
        for file in csv_files:
            csv_dir = dir / file
            files.append(pd.read_csv(csv_dir))
        df = pd.concat(files, ignore_index=True)
        i = 0
        for s in stations:
            df.loc[df['station'] == s, 'station'] = i
            i += 1
        data = df[target_columns].to_numpy()
        one_hot_label = np.zeros((len(data), data[:, 0].max() + 1))
        one_hot_label[np.arange(len(data)), data[:, 0].astype(int)] = 1
        data = torchcde.linear_interpolation_coeffs(torch.Tensor(data[:, 1:].astype(float)))

        # reshape to batch, hour, dim
        data = data.reshape(-1, self.T, data.size(-1))

        # shuffle the data for training
        data = data[torch.randperm(data.size(0))]

        times = data[..., :1] / 23.
        sequences = normalize_data(data[..., 1:])[0]
        # sequences = data[..., 1:]
        initial_values = sequences[:, :1]
        # labels = one_hot_label.reshape(-1, 24, len(stations))[:, :1].astype(float)

        times, sequences, initial_values = times.numpy(), sequences.numpy(), initial_values.numpy()
        np.savez(dir / self.training_file, init=initial_values, seq=sequences, time=times)#, label=labels)


if __name__ == '__main__':
    baqd = BAQD(root=DATA_DIR)
