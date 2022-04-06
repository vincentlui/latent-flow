import argparse
import logging

import numpy as np
import torch

from nfsde.experiments.synthetic.experiment import Synthetic
from nfsde.experiments.synthetic.experiment_ctfp import Synthetic_CTFP
from nfsde.experiments.synthetic.experiment_qv import Synthetic_qv
from nfsde.experiments.synthetic.experiment_mc import Synthetic_mc
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser('Neural flows')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--experiment', type=str, help='Which experiment to run',
                    choices=['synthetic', 'synthetic2','synthetic3', 'synthetic4'])
parser.add_argument('--model', type=str, help='Whether to use ODE or flow based model or RNN',
                    choices=['ode', 'flow', 'rnn'])
parser.add_argument('--data',  type=str, help='Dataset name',
                    choices=['hopper', 'physionet', 'activity', # latent ode
                             'sine', 'square', 'triangle', 'sawtooth', 'sink', 'ellipse', 'brownian', 'ou', 'gbm', 'linear', 'ou2', 'lorenz',# synthetic
                             'mimic3', 'mimic4', '2dou', #gru-ode-bayes
                             'hawkes1', 'hawkes2', 'poisson', 'renewal', 'reddit', 'mooc', 'lastfm', 'wiki', # tpp
                             'pinwheel', 'earthquake', 'covid', 'bike', # stpp
                    ])

# Training loop args
parser.add_argument('--epochs', type=int, default=1000, help='Max training epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay (regularization)')
parser.add_argument('--lr-scheduler-step', type=int, default=-1, help='Every how many steps to perform lr decay')
parser.add_argument('--lr-decay', type=float, default=0.9, help='Multiplicative lr decay factor')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--clip', type=float, default=1, help='Gradient clipping')

# NN args
parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=1, help='Size of hidden layer')
parser.add_argument('--activation', type=str, default='Tanh', help='Hidden layer activation')
parser.add_argument('--final-activation', type=str, default='Identity', help='Last layer activation')

# ODE args
parser.add_argument('--odenet', type=str, default='concat', help='Type of ODE network', choices=['concat', 'gru']) # gru only in GOB
parser.add_argument('--solver', type=str, default='dopri5', help='ODE solver', choices=['dopri5', 'rk4', 'euler'])
parser.add_argument('--solver_step', type=float, default=0.05, help='Fixed solver step')
parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance')
parser.add_argument('--rtol', type=float, default=1e-3, help='Relative tolerance')

# Flow model args
parser.add_argument('--flow-model', type=str, default='coupling', help='Model name', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--flow-layers', type=int, default=1, help='Number of flow layers')
parser.add_argument('--time-net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
parser.add_argument('--time-hidden-dim', type=int, default=1, help='Number of time features (only for Fourier)')

#Posterior args
parser.add_argument('--is-latent', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use latent state?')
parser.add_argument('--flow-dim', type=int, default=4, help='Size of hidden sde')
parser.add_argument('--w-dim', type=int, default=8, help='Size of base SDE')
parser.add_argument('--posterior-hidden-dim', type=int, default=8, help='Size of LSTM flow hidden state')
parser.add_argument('--z-dim', type=int, default=8, help='Size of Z')
parser.add_argument('--iwae-train', type=int, default=3, help='Number of samples for iwae train')
parser.add_argument('--iwae-test', type=int, default=10, help='Number of samples for iwae test')

# Logs
parser.add_argument('--log_dir', type=str, help='log path', default=os.getcwd() + '/result')

args = parser.parse_args()

def get_experiment(args, logger):
    if args.experiment == 'synthetic':
        return Synthetic(args, logger)
    elif args.experiment == 'synthetic2':
        return Synthetic_CTFP(args, logger)
    elif args.experiment == 'synthetic3':
        return Synthetic_qv(args, logger)
    else:
        raise ValueError(f'Need to specify experiment')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experiment = get_experiment(args, logger)

    experiment.train()
    experiment.finish()
