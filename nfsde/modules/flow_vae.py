import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow, LatentCouplingFlow
from nfsde.base_sde import OU2, Brownian2
from nfsde.util import dotdict
from nfsde.models.variational import PosteriorCDE

class FlowVAE(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.base_sde = self.get_base_sde(args)
        self.initial_layer = None
        self.readout = None
        self.flow_dim = self.dim
        if args.hidden_state_dim != -1:
            self.initial_layer = nn.Linear(self.dim, args.hidden_state_dim)
            self.readout = nn.Linear(args.hidden_state_dim, self.dim)
            self.flow_dim = args.hidden_state_dim
        self.model = self.get_model(args, self.flow_dim)
        self.model.multiply_latent_weights(args.init_mult2)
        self.posterior = self.get_posterior(args)
        torch.autograd.set_detect_anomaly(True)

        self.save_hyperparameters(args)


    def get_model(self, args, dim):
        if args.flow_model == "resnet":
            return LatentResnetFlow(dim, args.w_dim+args.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                args.time_net, args.time_hidden_dim, args.activation)
        elif args.flow_model == "coupling":
            return LatentCouplingFlow(dim, args.w_dim + args.z_dim, args.flow_layers,
                                    [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim, args.activation)
        else:
            raise NotImplementedError

    def get_base_sde(self, args):
        if args.base_sde == "brownian":
            return Brownian2(args.w_dim, args.sigma, train_param=False)
        elif args.base_sde == "ou":
            return OU2(args.w_dim, args.theta, args.mu, args.sigma, train_param=False)
        else:
            raise NotImplementedError

    def get_posterior(self, args):
        return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, args.decoder_hidden_state_dim, args.flow_layers,
                            [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                            [args.decoder_hidden_dim] * args.decoder_hidden_layers,
                            args.w_dim, args.z_dim, time_net=args.time_net, time_hidden_dim=args.time_hidden_dim,)


    def forward(self, x, t, w=None, z=None):
        initial = x
        if self.initial_layer is not None:
            initial = self.initial_layer(x)
        if z is None:
            z = torch.randn(x.shape[0], self.args.z_dim, device=self.device)
        if w is None:
            w = self.base_sde(t)
        flow_output = self.model(initial, t, w, z)
        if self.readout is not None:
            flow_output = self.readout(flow_output)
        return flow_output

    def _get_loss(self, batch, tag):
        x, t, y_true = batch
        t_0 = torch.zeros((len(t), 1, 1), device=self.device)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        dt = t - t_i

        w, z, mu_w, log_std_w, mu_z, log_std_z, *_ = self.posterior(y_true, t)

        y = self(x, t, w, z)

        mse = (y - y_true) ** 2
        mse_loss = torch.mean(torch.sum(mse, dim=-2))

        w_prior_mean, w_prior_std = self.base_sde.get_path_mean_and_std(w, t)
        kl_w = torch.sum(
            torch.log(w_prior_std + 1e-8) - 2 * log_std_w + ((mu_w - w_prior_mean) ** 2 + log_std_w.exp() ** 2) / (
                        w_prior_std + 1e-8) - 1,
            dim=-1, keepdim=True)

        kl_w[dt <= 0] = 0
        kl_loss_w = torch.mean(.5 * (torch.sum(kl_w, dim=-2)))

        kl_loss_z = torch.mean(.5 * (torch.sum(- 2*log_std_z + (mu_z**2 + log_std_z.exp()**2) - 1, dim=-1)))

        loss = mse_loss + kl_loss_w + kl_loss_z
        self.log(tag, loss)
        self.log(tag + '_mse', mse_loss)
        self.log(tag + '_kl_w', kl_loss_w)
        self.log(tag + '_kl_z', kl_loss_z)
        return loss

    # def backward(self, loss, optimizer, optimizer_idx):
    #     loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    #     print('test')

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'val_loss')
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

