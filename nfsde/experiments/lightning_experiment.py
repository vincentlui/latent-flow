import time
from argparse import Namespace
from copy import deepcopy
from logging import Logger
from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from nfsde.experiments.synthetic.data import get_data_loaders


import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



class LitExperiment:
    """ Base experiment class """
    def __init__(self, args: Namespace, logger: Logger, model):
        super().__init__()
        self.logger = logger
        self.args = args
        self.epochs = args.epochs
        self.patience = args.patience

        # self.logger2.info(f'Device: {self.device}')

        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, self.mean, self.std \
            = self.get_data_loaders(args.data, args.batch_size)
        # self.model = self.get_model(args)
        # self.logger.info(f'num_params={sum(p.numel() for p in self.model.parameters())}')

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = None
        if args.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, args.lr_scheduler_step, args.lr_decay)

        early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False,
                                            mode="max")
        self.trainer = pl.Trainer(callbacks=[early_stop_callback])

    def train(self):
        self.trainer.fit(model=self.model, train_dataloaders=self.dltrain)

    def test(self):
        return




