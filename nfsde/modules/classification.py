import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow
from nfsde.models.discriminator import CDEClassificationNet, CDE
from nfsde.util import dotdict

class Classification(pl.LightningModule):
    def __init__(self, args, data, generator):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.model = self.get_model(args, self.dim)
        self.generator = generator

        self.save_hyperparameters(args)


    def get_model(self, args, dim):
        return CDE(dim, [args.hidden_dim] * args.hidden_layers, args.hidden_state_dim, 1, args.activation,
                                    args.final_activation)

    def forward(self, x, t):
        return self.model(torch.cat([t, x], dim=-1))

    def _get_loss(self, batch):
        x, t, y_true = batch
        criterion = nn.BCEWithLogitsLoss()
        x_gen = self.generator(t, averaged=True)
        p_real = self.model(torch.cat([t, y_true], dim=-1))
        p_gen = self.model(torch.cat([t, x_gen], dim=-1))
        loss = criterion(p_real, torch.ones(p_real.shape)) + criterion(p_gen, torch.zeros(p_real.shape))
        return loss

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler_step > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, self.args.lr_decay)
            return [optimizer], [scheduler]

        return optimizer

    def train_dataloader(self):
        return self.dltrain

    def val_dataloader(self):
        return self.dlval

    def test_dataloader(self):
        return self.dltest

