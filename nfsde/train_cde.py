import argparse
import logging

import numpy as np
import torch

from nfsde.experiments.synthetic.experiment import Synthetic
from nfsde.experiments.synthetic.experiment_ctfp import Synthetic_CTFP
from nfsde.experiments.synthetic.experiment_qv import Synthetic_qv
from nfsde.experiments.synthetic.experiment_mc import Synthetic_mc
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.experiments.real.data import get_data_loaders as get_read_data_loader
from nfsde.modules.classification import Classification
from nfsde.modules.sde_gan import SDEGAN
from nfsde.modules.flow_gan import FlowGAN
from nfsde.util import MetricsCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser('Neural flows')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--experiment', type=str, help='Which experiment to run',
                    choices=['synthetic'])
parser.add_argument('--model', type=str, help='Whether to use ODE or flow based model or RNN',)
                    #choices=['ode', 'flow', 'sde-gan'])
parser.add_argument('--data',  type=str, help='Dataset name',
                    choices=['brownian', 'ou', 'gbm', 'linear', 'ou2', 'lorenz',# synthetic
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
parser.add_argument('--early-stop', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Early stopping')
parser.add_argument('--optim',  type=str, default='Adam', help='optimizer', choices=['Adam', 'Adadelta', 'SGD'])

# Validation args
parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
parser.add_argument('--train-size', type=float, default=0.6)
parser.add_argument('--val-size', type=float, default=0.2)

# NN args
parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=1, help='Size of hidden layer')
parser.add_argument('--activation', type=str, default='Tanh', help='Hidden layer activation')
parser.add_argument('--final-activation', type=str, default='Identity', help='Last layer activation')

# CDE args
parser.add_argument('--hidden-state-dim', type=int, default=5,  help='dim of cde state')

# Logs
parser.add_argument('--log_dir', type=str, help='log path', default=os.getcwd() + '/result')
parser.add_argument('--data_dir', type=str, help='data path', default=None)
parser.add_argument('--generator-path', type=str, help='generator path', default=None)

args = parser.parse_args()

def get_module(args, data, generator):
    return Classification(args, data, generator)

def get_experiment(args):
    if args.model == 'sde-gan':
        return SDEGAN
    else:
        return FlowGAN


if __name__ == '__main__':
    logging.info('Received the following configuration:')
    logging.info(f'Args: {args}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = get_data_loaders(args.data, args.batch_size, args.data_dir, train_size=args.train_size, val_size=args.val_size)
    model_class = get_experiment(args)
    generator = model_class.load_from_checkpoint(args.generator_path, data=data)
    module = get_module(args, data, generator)

    callbacks = []
    callbacks.append(MetricsCallback())
    modelcpt = ModelCheckpoint(every_n_epochs=1, filename="checkpoint-{epoch:04d}", save_top_k=-1, save_last=True)
    callbacks.append(modelcpt)
    if args.early_stop:
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="auto",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.clip,
        callbacks=callbacks,
    )
    trainer.fit(model=module)

    trainer.test()
    logging.info(f'test loss: {callbacks[0].get_test_loss()}')