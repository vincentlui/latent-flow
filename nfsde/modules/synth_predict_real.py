import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow
from nfsde.models.latent_ode import LatentODE
from nfsde.models.variational import PosteriorCDE
from nfsde.models.discriminator import CDE
from nfsde.util import dotdict

class SynthPredictReal(pl.LightningModule):
    def __init__(self, args, data, generator):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.model = self.get_model(args)
        self.encoder = self.get_encoder(args)
        self.generator = generator
        self.readout = nn.Linear(args.encoder_hidden_state_dim, self.dim)
        if generator.__class__.__name__ in ['FlowVAE']:
            self.generator_model = 0
        elif generator.__class__.__name__ in ['CTFPModule', 'FlowMCZ', 'FlowMC']:
            self.generator_model = 1
        elif generator.__class__.__name__ in ['LSDE', 'CLPFModule']:
            self.generator_model = 2
        else:
            self.generator_model = 3
        self.save_hyperparameters(args)


    def get_model(self, args):
        return LatentODE(args.encoder_hidden_state_dim, args.z_dim, [args.hidden_dim] * args.hidden_layers,
                         activation=args.activation, final_activation=args.final_activation,)

    def get_encoder(self, args):
        return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, 0,
                            [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                            [],
                            0, args.z_dim + args.encoder_hidden_state_dim)


    def forward(self, x, t, z=None):
        return self.readout(self.model(x, t, latent=z))

    def _get_loss(self, batch):
        x, t, y_true = batch
        with torch.no_grad():
            if self.generator_model == 0:
                y_gen = self.generator(x, t)
            elif self.generator_model == 1:
                y_gen = self.generator(t)
            elif self.generator_model == 2:
                y_gen = self.generator(t[0].flatten(), x.shape[0])
            else:
                y_gen = self.generator(t, averaged=True)
        split = int(self.args.t_split * t.shape[1])
        _, _, _, _, z, _, _, _ = self.encoder(y_gen[:, :split], t[:, :split], return_logp=False)
        y_pred = self(z[..., :self.args.encoder_hidden_state_dim], t[0, split:].flatten(), z=z[..., self.args.encoder_hidden_state_dim:])
        loss = torch.sum((y_gen[:, split:] - y_pred)**2, dim=-1)
        return loss.mean()

    def _get_test_loss(self, batch):
        x, t, y_true = batch
        split = int(self.args.t_split * t.shape[1])
        _, _, _, _, z, _, _, _ = self.encoder(y_true[:, :split], t[:, :split], return_logp=False)
        y_pred = self(z[..., :self.args.encoder_hidden_state_dim], t[0, split:].flatten(), z=z[..., self.args.encoder_hidden_state_dim:])
        loss = torch.sum((y_true[:, split:] - y_pred)**2, dim=-1)
        return loss.mean()

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self._get_test_loss(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.args.lr,
        #                                      weight_decay=self.args.weight_decay)
        optimizer = optim.Adam(list(self.encoder.parameters())+list(self.model.parameters())+ list(self.readout.parameters()), lr=self.args.lr,
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

