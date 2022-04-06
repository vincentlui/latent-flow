import argparse
import logging

import numpy as np
import torch

from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.experiments.real.data import get_real_data_loaders
from nfsde.modules.sde_gan import SDEGAN
from nfsde.modules.flow_gan import FlowGAN
from nfsde.modules.flow_mc import FlowMC
from nfsde.modules.flow_mc_z import FlowMCZ
from nfsde.modules.ctfp import CTFPModule
from nfsde.modules.flow_vae import FlowVAE
from nfsde.modules.lsde import LSDE
from nfsde.modules.clpf import CLPFModule
from nfsde.modules.synth_predict_real import SynthPredictReal
from nfsde.util import MetricsCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser('Neural flows')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--experiment', type=str, help='Which experiment to run',
                    choices=['synthetic', 'synthetic2','synthetic3', 'synthetic4'])
parser.add_argument('--model', type=str, help='Whether to use ODE or flow based model or RNN',)
                    #choices=['ode', 'flow', 'sde-gan'])
parser.add_argument('--data',  type=str, help='Dataset name',)
                    #choices=['brownian', 'ou', 'gbm', 'linear', 'ou2', 'lorenz',# synthetic
                    #])

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
parser.add_argument('--swa', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use stochastic weight averaging')

# Validation args
parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
parser.add_argument('--train-size', type=float, default=0.6)
parser.add_argument('--val-size', type=float, default=0.2)

# NN args
parser.add_argument('--hidden-layers', type=int, default=1, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=1, help='Size of hidden layer')
parser.add_argument('--activation', type=str, default='Tanh', help='Hidden layer activation')
parser.add_argument('--final-activation', type=str, default='Identity', help='Last layer activation')


# Flow model args
parser.add_argument('--flow-model', type=str, default='coupling', help='Model name', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--flow-layers', type=int, default=1, help='Number of flow layers')
parser.add_argument('--time-net', type=str, default='TimeLinear', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
parser.add_argument('--time-hidden-dim', type=int, default=1, help='Number of time features (only for Fourier)')
parser.add_argument('--learn-std', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Train STD')
parser.add_argument('--std-ema-factor', type=float, default=0., help='STD EMA factor')
parser.add_argument('--std-likelihood', type=float, default=1., help='STD Likelihood')

#Posterior args
parser.add_argument('--posterior-model', type=str, default='ode-rnn', help='posterior encoder model')
parser.add_argument('--flow-dim', type=int, default=4, help='Size of hidden sde')
parser.add_argument('--w-dim', type=int, default=3, help='Size of base SDE')
parser.add_argument('--posterior-hidden-dim', type=int, default=8, help='Size of LSTM flow hidden state')
parser.add_argument('--encoder-hidden-state-dim', type=int, default=8, help='Hidden state dim of encoder')
parser.add_argument('--decoder-hidden-state-dim', type=int, default=8, help='Hidden state dim of decoder')
parser.add_argument('--z-dim', type=int, default=3, help='Size of Z')
parser.add_argument('--encoder-hidden-layers', type=int, default=1, help='Number of hidden layers of encoder')
parser.add_argument('--encoder-hidden-dim', type=int, default=16, help='Size of hidden layer of encoder')
parser.add_argument('--decoder-hidden-layers', type=int, default=1, help='Number of hidden layers of decoder')
parser.add_argument('--decoder-hidden-dim', type=int, default=16, help='Size of hidden layer of decoder')
parser.add_argument('--add-noise', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Add noise to data')
parser.add_argument('--kl-loss', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use kl loss')
parser.add_argument('--iwae-train', type=int, default=3, help='Number of samples for iwae train')
parser.add_argument('--iwae-test', type=int, default=10, help='Number of samples for iwae test')

# SDE GAN args
parser.add_argument('--d-hidden-layers', type=int, default=1, help='Number of hidden layers of discriminator')
parser.add_argument('--d-hidden-dim', type=int, default=16, help='Size of hidden layer of discriminator')
parser.add_argument('--initial-noise-size', type=int, default=5,  help='initial noise size')
parser.add_argument('--noise-size', type=int, default=3,  help='noise size')
parser.add_argument('--hidden-state-dim', type=int, default=5,  help='dim of sde state')
parser.add_argument('--swa-step-start', type=int, default=100,  help='which epoch does swa start')
parser.add_argument('--g-lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--d-lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--g-weight-decay', type=float, default=0, help='Weight decay of generator (regularization)')
parser.add_argument('--d-weight-decay', type=float, default=0, help='Weight decay of discriminator (regularization)')

# Flow Gan args
parser.add_argument('--init-mult1', type=float, default=1., help='Multiplication factor for initial weights of _initial')
parser.add_argument('--init-mult2', type=float, default=1., help='Multiplication factor for initial weights of flow')
parser.add_argument('--update-g-every-n-iter', type=int, default=1,  help='update generator every n iterations')
parser.add_argument('--d-hidden-state-dim', type=int, default=16,  help='dim of sde state')
parser.add_argument('--d-model', type=str, default='CDE', help='Discriminator model name', choices=['CDE', 'flow'])
parser.add_argument('--joint-training', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Joint training')

# Base SDE args
parser.add_argument('--base-sde', type=str, default='ou', help='Base SDE', choices=['brownian', 'ou'])
parser.add_argument('--train-base-sde', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Train SDE param')
parser.add_argument('--theta', type=float, default=0.1, help='sigma of base sde')
parser.add_argument('--mu', type=float, default=0., help='sigma of base sde')
parser.add_argument('--sigma', type=float, default=1., help='sigma of base sde')

# Logs
parser.add_argument('--log-dir', type=str, help='log path', default=os.getcwd() + '/result')
parser.add_argument('--data-dir', type=str, help='data path', default=None)
parser.add_argument('--generator-path', type=str, help='generator path')

#training
parser.add_argument('--t-split', type=float, default=0.8, help='time split')

parser.add_argument('--a', type=float, default=1., help='a')
parser.add_argument('--b', type=float, default=1., help='b')
args = parser.parse_args()

def get_module(args):
    if args.model == 'sde-gan':
        return SDEGAN
    elif args.model == 'flow-mc':
        return FlowMC
    elif args.model == 'flow-mcz':
        return FlowMCZ
    elif args.model in ['ctfp-ode', 'ctfp-flow', 'ctfp']:
        return CTFPModule
    elif args.model == 'flow-gan':
        return FlowGAN
    elif args.model == 'flow-vae':
        return FlowVAE
    elif args.model == 'latent-sde':
        return LSDE
    elif args.model == 'clpf':
        return CLPFModule


def get_data(args):
    if args.data in ['energy', 'baqd', 'hopper']:
        return get_real_data_loaders(args.data, args.batch_size, args.data_dir,
                            train_size=args.train_size, val_size=args.val_size, add_noise=args.add_noise)

    return get_data_loaders(args.data, args.batch_size, args.data_dir,
                            train_size=args.train_size, val_size=args.val_size, add_noise=args.add_noise)


if __name__ == '__main__':
    logging.info('Received the following configuration:')
    logging.info(f'Args: {args}')
    pl.seed_everything(args.seed, workers=True)

    data = get_data(args)
    module_cl = get_module(args)
    generator = module_cl.load_from_checkpoint(args.generator_path, data=data)
    module = SynthPredictReal(args, data, generator)

    callbacks = []
    callbacks.append(MetricsCallback(args))
    modelcpt = ModelCheckpoint(every_n_epochs=1, filename="checkpoint-{epoch:04d}", save_top_k=1, save_last=True, monitor='val_loss')
    callbacks.append(modelcpt)
    if args.early_stop:
        callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience))
    if args.swa:
        callbacks.append(pl.callbacks.StochasticWeightAveraging())
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="auto",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.clip,
        callbacks=callbacks,
        default_root_dir="result/tstr/" +args.model + "/" + args.data + "/" + datetime.now().strftime('%Y%m%d-%H%M%S'),
        deterministic=True
    )
    trainer.fit(model=module)

    trainer.test(ckpt_path='best')
    logging.info(f'test loss: {callbacks[0].get_test_loss()}')