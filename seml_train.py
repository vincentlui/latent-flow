import argparse
import logging

import numpy as np
import torch

from nfsde.experiments.synthetic.experiment import Synthetic
from nfsde.experiments.synthetic.experiment_ctfp import Synthetic_CTFP
from nfsde.experiments.synthetic.experiment_qv import Synthetic_qv
from nfsde.experiments.synthetic.experiment_mc import Synthetic_mc
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.modules.sde_gan import SDEGAN
from nfsde.modules.flow_gan import FlowGAN
from nfsde.util import dotdict, MetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
import os

from sacred import Experiment
import seml


def get_module(args, data):
    if args.model == 'sde-gan':
        return SDEGAN(args, data)
    else:
        return FlowGAN(args, data)

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
    data = get_data_loaders(args.data, args.batch_size, args.data_dir, train_size=args.train_size, val_size=args.val_size)
    module = get_module(args, data)
    callbacks = []
    callbacks.append(MetricsCallback(args))
    modelcpt = ModelCheckpoint(every_n_epochs=int(args.epochs / 10), filename="checkpoint-{epoch:04d}",
                               save_top_k=-1, save_last=True)
    callbacks.append(modelcpt)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator="auto",
        check_val_every_n_epoch=10,
        callbacks=callbacks
    )
    trainer.fit(model=module)

    trainer.test()
    test_loss = callbacks[0].get_test_loss()
    logging.info(f'test loss: {test_loss}')

    return {'test_loss': test_loss}
