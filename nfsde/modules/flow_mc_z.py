import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow, LatentCouplingFlow
from nfsde.base_sde import OU2, Brownian2
from nfsde.util import dotdict
from nfsde.models.variational import PosteriorCDE
from stribor.net import MLP

class FlowMCZ(pl.LightningModule):
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
            self.initial_layer = nn.Linear(args.initial_noise_size, args.hidden_state_dim)
            self.readout = nn.Linear(args.hidden_state_dim, self.dim)
            self.flow_dim = args.hidden_state_dim
        if args.learn_std:
            self.alpha = MLP(self.dim + 1, [64, 64], self.dim, activation='ReLU', final_activation='Softplus')
            # self.alpha = MLP(1, [32, 32], self.dim, activation='ReLU', final_activation='Softplus')
            # self.alpha_previous = copy.deepcopy(self.alpha)
        else:
            self.alpha = lambda x: args.std_likelihood
        self.model = self.get_model(args, self.flow_dim)
        self.model.multiply_latent_weights(args.init_mult2)
        self.posterior = self.get_posterior(args)
        self.x0_layer = nn.Linear(args.encoder_hidden_state_dim, args.initial_noise_size * 2)

        self.save_hyperparameters(args)


    def get_model(self, args, dim):
        if args.flow_model == "resnet":
            return LatentResnetFlow(dim, args.w_dim, args.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                args.time_net, args.time_hidden_dim, args.activation)
        elif args.flow_model == "coupling":
            return LatentCouplingFlow(dim, args.w_dim, args.z_dim, args.flow_layers,
                                    [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim, args.activation)
        else:
            raise NotImplementedError

    def get_posterior(self, args):
        return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, args.flow_layers,
                            [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                            [],
                            0, args.z_dim)

    def get_base_sde(self, args):
        if args.base_sde == "brownian":
            return Brownian2(args.w_dim, args.sigma, train_param=args.train_base_sde)
        elif args.base_sde == "ou":
            return OU2(args.w_dim, args.theta, args.mu, args.sigma, train_param=args.train_base_sde)
        else:
            raise NotImplementedError


    def forward(self, t, x=None, z=None):
        ws = self.base_sde(t)
        if x is None:
            x = torch.randn((t.shape[0], 1, self.args.initial_noise_size), device=self.device)
        x = self.initial_layer(x)
        if self.args.z_dim > 0:
            if z is None:
                z = torch.randn(x.shape[0], self.args.z_dim, device=self.device)
        flow_output = self.model(x, t, ws, z)
        if self.readout is not None:
            flow_output = self.readout(flow_output)
        return flow_output

    def _get_loss(self, batch, tag, k=20):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        if self.args.kl_loss:
            logk = torch.log(torch.tensor(k, device=self.device))
            x = x.repeat_interleave(k, dim=0)
            t = t.repeat_interleave(k, dim=0)
            nan_ind = torch.isnan(y_true)
            y_true[nan_ind] = 0.
            y_true = y_true.repeat_interleave(k, dim=0)
            nan_ind = nan_ind.repeat_interleave(k, dim=0) #y_true[(~nan_ind.view(-1)).nonzero()[:, 0]]
            _, z, _, _, mu_z, log_std_z, encoded, _ = self.posterior(y_true, t)
            mu_and_log_std_x0 = self.x0_layer(encoded).unsqueeze(-2)
            mu_x, log_std_x = torch.chunk(mu_and_log_std_x0, 2, -1)
            std_x = torch.exp(log_std_x)
            distribution_x = torch.distributions.Normal(mu_x, std_x)
            x0 = distribution_x.rsample()
            kl_loss_z = torch.mean(.5 * (torch.sum(- 2 * log_std_z + (mu_z ** 2 + log_std_z.exp() ** 2) - 1, dim=-1)))
            kl_loss_x0 = torch.mean(.5 * (torch.sum(- 2 * log_std_x + (mu_x ** 2 + log_std_x.exp() ** 2) - 1, dim=-1)))

            y = self(t, x=x0, z=z)
            distr = torch.distributions.Normal(y_true, self.alpha(torch.cat([x.repeat_interleave(t.shape[-2], dim=-2), t],dim=-1)) * torch.ones_like(y_true, device=self.device))
            log_pyz = distr.log_prob(y)
            # if self.training:
            #     log_pyz = torch.clamp(log_pyz, -20, 2)
            log_pyz[nan_ind] = 0.
            log_pyz = log_pyz.view(-1, k, num_seq * dim).sum(dim=-1)
            log_pyz = torch.mean(torch.logsumexp(log_pyz, dim=-1) - logk)
            loss = -log_pyz + self.args.a * kl_loss_z + self.args.b * kl_loss_x0
            self.log(tag + '_kl_x', kl_loss_x0)
            self.log(tag + '_kl_z', kl_loss_z)
            self.log(tag + '_log_pyz', log_pyz)
        else:

            logk = torch.log(torch.tensor(k, device=self.device))
            x = x.repeat_interleave(k, dim=0)
            t = t.repeat_interleave(k, dim=0)
            y_true = y_true.repeat_interleave(k, dim=0)
            _, z, _, _, mu_z, log_std_z, encoded, log_qz = self.posterior(y_true, t, return_logp=True)
            mu_and_log_std_x0 = self.x0_layer(encoded).unsqueeze(-2)
            mu_x, log_std_x = torch.chunk(mu_and_log_std_x0, 2, -1)
            log_std_x = torch.clamp(log_std_x, -20, 2)
            std_x = torch.exp(log_std_x)
            distribution_x = torch.distributions.Normal(mu_x, std_x)
            x0 = distribution_x.rsample()
            kl_loss_x0 = torch.mean(.5 * (torch.sum(- 2 * log_std_x + (mu_x ** 2 + log_std_x.exp() ** 2) - 1, dim=-1)))
            y = self(t, x=x0, z=z)
            log_pz = torch.distributions.Normal(torch.ones_like(z, device=self.device), torch.ones_like(z, device=self.device)).log_prob(z).sum(dim=-1)
            if self.alpha is not None:
                distr = torch.distributions.Normal(y_true, self.alpha(torch.cat([x.repeat_interleave(t.shape[-2], dim=-2), t],dim=-1)) * torch.ones_like(y_true, device=self.device))
            else:
                distr = torch.distributions.Normal(y_true, torch.ones_like(y_true, device=self.device))
            log_pyz = distr.log_prob(y).view(-1, k, num_seq * dim).sum(dim=-1) - log_qz.view(-1, k) + log_pz.view(-1, k)
            log_py = torch.mean(torch.logsumexp(log_pyz, dim=-1) - logk)
            loss = -log_py + kl_loss_x0
            self.log(tag + '_kl_x', kl_loss_x0)
            self.log(tag + '_log_py', log_py)
        self.log(tag, loss)
        return loss

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'train', self.args.iwae_train)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'val_loss', self.args.iwae_test)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'test', self.args.iwae_test)
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

