import argparse
import logging
import torch
import numpy as np

from nfsde.modules.sde_gan import SDEGAN
from nfsde.modules.flow_gan import FlowGAN
from nfsde.modules.flow_mc import FlowMC
from nfsde.modules.flow_mc_z import FlowMCZ
from nfsde.modules.ctfp import CTFPModule
from nfsde.modules.clpf import CLPFModule
from nfsde.modules.flow_vae import FlowVAE
from nfsde.modules.lsde import LSDE
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.experiments.real.data import get_real_data_loaders

from nfsde.util import dotdict, MetricsCallback, SavePlotCallback, ResumeTrainingCallback
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
import os
from datetime import datetime

from sacred import Experiment
import seml


def get_data(args):
    if args.data in ['energy', 'baqd', 'hopper']:
        return get_real_data_loaders(args.data, args.batch_size, args.data_dir,
                            train_size=args.train_size, val_size=args.val_size, add_noise=args.add_noise)

    return get_data_loaders(args.data, args.batch_size, args.data_dir,
                            train_size=args.train_size, val_size=args.val_size, add_noise=args.add_noise)


def get_module(args, data):
    if args.model == 'sde-gan':
        return SDEGAN(args, data)
    elif args.model == 'flow-mc':
        return FlowMC(args, data)
    elif args.model == 'flow-mcz':
        return FlowMCZ(args, data)
    elif args.model in ['ctfp-ode', 'ctfp-flow', 'ctfp']:
        return CTFPModule(args, data)
    elif args.model == 'flow-gan':
        return FlowGAN(args, data)
    elif args.model == 'flow-vae':
        return FlowVAE(args, data)
    elif args.model == 'clpf':
        return CLPFModule(args, data)
    elif args.model == 'latent-sde':
        return LSDE(args, data)


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(args):
    logging.info('Received the following configuration:')
    logging.info(f'Args: {args}')
    args = dotdict(args)
    pl.seed_everything(args.seed, workers=True)
    data = get_data(args)
    module = get_module(args, data)
    callbacks = []
    callbacks.append(MetricsCallback(args))
    callbacks.append(SavePlotCallback(args))
    if args.model in ['flow-gan', 'sde-gan']:
        modelcpt = ModelCheckpoint(every_n_epochs=int(args.epochs / 10), filename="checkpoint-{epoch:04d}",
                                   save_top_k=-1, save_last=True)
        callbacks.append(modelcpt)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            devices=1,
            accelerator="auto",
            check_val_every_n_epoch=int(args.epochs / 10),
            callbacks=callbacks,
            default_root_dir="result/" + args.model + "/" + args.data,
            deterministic=True
        )
        trainer.fit(model=module)
        trainer.test()
    else:
        modelcpt = ModelCheckpoint(every_n_epochs=1, filename="checkpoint-{epoch:04d}",
                                   save_top_k=1, save_last=True, monitor='val_loss')
        callbacks.append(modelcpt)
        if args.early_stop:
            callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience))

        if args.checkpoint_path is not None:
            resumecpt = ResumeTrainingCallback()
            callbacks.append(resumecpt)
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                devices=1,
                accelerator="auto",
                check_val_every_n_epoch=1,
                callbacks=callbacks,
                default_root_dir="result/" +args.model + "/" + args.data,
                resume_from_checkpoint=args.checkpoint_path,
                deterministic=True
            )
        else:

            trainer = pl.Trainer(
                max_epochs=args.epochs,
                devices=1,
                accelerator="auto",
                check_val_every_n_epoch=1,
                callbacks=callbacks,
                default_root_dir="result/" +args.model + "/" + args.data,
                deterministic=True
            )
        trainer.fit(model=module)
        trainer.test(ckpt_path='best')
    test_loss = callbacks[0].get_test_loss()
    logging.info(f'test loss: {test_loss}')

    return {'test_loss': test_loss}
