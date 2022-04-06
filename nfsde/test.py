import torch
import numpy as np
import matplotlib.pyplot as plt
from nfsde.models.CTFP import CTFP
from nfsde.base_sde import Brownian, OU
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.modules.sde_gan import SDEGAN
from nfsde.modules.flow_gan import FlowGAN
from nfsde.util import dotdict
from nfsde.experiments.real.lib.parse_datasets import parse_datasets
from nfsde.experiments.real.data import get_real_data_loaders
import os
# import pytorch_lightning as pl
import torchcde
import torchsde

from torch.distributions import Normal


args = dotdict()
args.data_dir= 'experiments/data/synth'
args['initial_noise_size'] = 5
args['noise_size'] = 3
args['hidden_state_dim'] = 5
args['hidden_layers'] = 2
args['hidden_dim'] = 64
args['experiment'] = 'synthetic'
args['data'] = 'ou'
args['batch_size'] = 10
args['w_dim'] = 3
args.z_dim = 3
args.model = 'flow'
args.flow_layers = 1
args.time_net = 'TimeFourier'
args.time_hidden_dim = 8
args.activation = 'ReLU'
args.d_hidden_layers = 1
args.d_hidden_dim = 16
args.init_mult1 = 0
args.init_mult2 = 0

args.extrap = 0
args.data = 'physionet'#'activity'
args.timepoints = 20
args.max_t = 10
args.n = 1000
args.quantization = 0.1
args.batch_size = 10
args.classify = False
x = get_real_data_loaders('activity', 10)
print(x)