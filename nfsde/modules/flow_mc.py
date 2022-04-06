import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow, LatentCouplingFlow
from nfsde.base_sde import OU2, Brownian2, Brownian_OU
from nfsde.models.variational import PosteriorCDE
from nfsde.models.discriminator import CDE
from nfsde.util import dotdict, calc_KL
from stribor.net import MLP
import copy

class FlowMC(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.mask = False
        if self.mask:
            self.get_loss = self._get_loss_masked
        else:
            self.get_loss = self._get_loss
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.base_sde = self.get_base_sde(args)
        self.initial_layer = None
        self.readout = None
        self.flow_dim = self.dim
        self.py0_mean = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        self.py0_logstd = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        # self.register_buffer('py0_mean', torch.zeros(1, args.hidden_state_dim))
        # self.register_buffer('py0_logstd', torch.zeros(1, args.hidden_state_dim))
        if args.hidden_state_dim != -1:
            # self.initial_layer = nn.Linear(self.dim, args.hidden_state_dim)
            # self.initial_layer = nn.Linear(args.initial_noise_size, args.hidden_state_dim)
            self.readout = nn.Linear(args.hidden_state_dim, self.dim)
            self.flow_dim = args.hidden_state_dim
        self.initial_encoder = self.get_encoder(args)
        if args.learn_std:
            # self.alpha = torch.nn.Parameter(torch.randn(self.dim, requires_grad=True))
            self.alpha = MLP(args.hidden_state_dim + 1, [64, 64], self.dim, activation='ReLU', final_activation='Softplus')
            # self.alpha = MLP(1, [32, 32], self.dim, activation='ReLU', final_activation='Softplus')
            self.alpha_previous = copy.deepcopy(self.alpha)
        else:
            self.alpha = lambda x: 1.
        self.model = self.get_model(args, self.flow_dim)
        self.model.multiply_latent_weights(args.init_mult2)

        self.save_hyperparameters(args)


    def get_model(self, args, dim):
        if args.flow_model == "resnet":
            return LatentResnetFlow(dim, args.w_dim, args.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                args.time_net, args.time_hidden_dim, args.activation, invertible=args.invertible)
        elif args.flow_model == "coupling":
            return LatentCouplingFlow(dim, args.w_dim, args.z_dim, args.flow_layers,
                                    [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim, args.activation)
        else:
            raise NotImplementedError

    def get_encoder(self, args):
        if args.posterior_model == 'CDE':
            return Encoder(self.dim, args.encoder_hidden_state_dim, args.hidden_state_dim - args.flow_dim, args.flow_dim,
                           [args.encoder_hidden_dim] * args.encoder_hidden_layers)

        return MLP(self.dim, [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                   args.hidden_state_dim * 2, activation='ReLU')

    # return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, 0,
    #                     [args.encoder_hidden_dim] * args.encoder_hidden_layers,
    #                     [],
    #                     0, args.initial_noise_size)

    def get_base_sde(self, args):
        if args.base_sde == "brownian":
            return Brownian2(args.w_dim, args.sigma, train_param=args.train_base_sde)
        elif args.base_sde == "ou":
            return OU2(args.w_dim, args.theta, args.mu, args.sigma, train_param=args.train_base_sde)
        elif args.base_sde == "combine":
            return Brownian_OU(args.w_dim, args.theta, args.mu, args.sigma, train_param=args.train_base_sde)
        else:
            raise NotImplementedError


    def forward(self, t, x=None, w=None, z=None):
        if self.args.w_dim > 0 and w is None:
            w = self.base_sde(t)
        if x is None:
            y0_dist = torch.distributions.Normal(self.py0_mean, self.py0_logstd.exp())
            x = y0_dist.sample(torch.Size([t.shape[0]]))
            # x = torch.randn((t.shape[0], 1, self.args.initial_noise_size), device=self.device)
        initial = x
        if self.initial_layer is not None:
            initial = self.initial_layer(x)
        if self.args.z_dim > 0 and z is None:
            z = torch.randn(x.shape[0], self.args.z_dim, device=self.device)
        flow_output = self.model(initial, t, w, z)
        if self.readout is not None:
            flow_output = self.readout(flow_output)
        return flow_output

    # def _get_loss(self, batch, tag, k=20):
    #     x, t, y_true = batch
    #     n, num_seq, dim = y_true.shape
    #     y_true_clone = y_true.clone()
    #     logk = torch.log(torch.tensor(k, device=self.device))
    #     x0 = x.repeat_interleave(k, dim=0)
    #     t = t.repeat_interleave(k, dim=0)
    #     y_true = y_true.repeat_interleave(k, dim=0)
    #
    #     y = self(x0, t)
    #     if self.alpha is not None:
    #         distr = torch.distributions.Normal(y_true, self.alpha(t) * torch.ones_like(y_true, device=self.device))
    #     else:
    #         distr = torch.distributions.Normal(y_true, torch.ones_like(y_true, device=self.device))
    #     log_py = distr.log_prob(y).view(-1, k, num_seq*dim).sum(dim=-1)
    #
    #     loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)
    #     self.log(tag, loss)
    #     return loss

    def _get_loss(self, batch, tag, k=20):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape

        x0, mu_x, log_std_x = self.initial_encoder(y_true, t)
        x0 = x0.unsqueeze(1)

        # mu_and_log_std_x = self.initial_encoder(x)
        # mu_x, log_std_x = torch.chunk(mu_and_log_std_x, 2, -1)
        # log_std_x = torch.clamp(log_std_x, -20, 2)
        # std_x = torch.exp(log_std_x)
        # distribution_x = torch.distributions.Normal(mu_x, std_x)
        # x0 = distribution_x.rsample()
        # kl_loss_x0 = torch.mean(.5 * (torch.sum(- 2 * log_std_x + (mu_x ** 2 + log_std_x.exp() ** 2) - 1, dim=-1)))
        # _, x0, _, _, mu_x, log_std_x, _, _ = self.initial_encoder(y_true, t)
        #x0 = x0.unsqueeze(1)
        kl_loss_x0 = torch.mean(calc_KL(mu_x, log_std_x, self.py0_mean.repeat_interleave(mu_x.shape[0], dim=0),
                                        self.py0_logstd.repeat_interleave(mu_x.shape[0], dim=0)))

        logk = torch.log(torch.tensor(k, device=self.device))
        x0 = x0.repeat_interleave(k, dim=0)
        t_repeat = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)
        # nan_ind = torch.isnan(y_true)
        # y_true[nan_ind] = 0.

        y = self(t_repeat, x=x0)
        # distr = torch.distributions.Normal(y_true, self.alpha(torch.cat([x0.repeat_interleave(t.shape[-2], dim=-2), t_repeat],dim=-1)) * torch.ones_like(y_true, device=self.device))
        distr = torch.distributions.Laplace(y_true, self.alpha(
                torch.cat([x0.repeat_interleave(t.shape[-2], dim=-2), t_repeat], dim=-1)) * torch.ones_like(y_true,
                                                                                                            device=self.device))
        log_py = distr.log_prob(y)
        # if self.training:
        #     log_py = torch.clamp(log_py, -20, 20)
        # log_py[nan_ind] = 0.
        log_py = log_py.view(-1, k, num_seq*dim).sum(dim=-1)
        if tag != 'train':
            loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk) + kl_loss_x0
        else:
            loss = -torch.mean(torch.sum(
                torch.nn.functional.softmax(log_py, -1).detach() * log_py, dim=-1)) + kl_loss_x0
        loss /= num_seq
        self.log(tag+'_loss', loss)
        return loss

    # def _get_loss_masked(self, batch, tag, k=20):
    #     x, t, y_true, mask = batch
    #     n, num_seq, dim = y_true.shape
    #
    #     y_true_en = copy.deepcopy(y_true)
    #     y_true_en[~mask.bool()] = torch.nan
    #
    #     x0, mu_x, log_std_x = self.initial_encoder(y_true_en, t)
    #     x0 = x0.unsqueeze(1)
    #
    #     # mu_and_log_std_x = self.initial_encoder(x)
    #     # mu_x, log_std_x = torch.chunk(mu_and_log_std_x, 2, -1)
    #     # log_std_x = torch.clamp(log_std_x, -20, 2)
    #     # std_x = torch.exp(log_std_x)
    #     # distribution_x = torch.distributions.Normal(mu_x, std_x)
    #     # x0 = distribution_x.rsample()
    #     # kl_loss_x0 = torch.mean(.5 * (torch.sum(- 2 * log_std_x + (mu_x ** 2 + log_std_x.exp() ** 2) - 1, dim=-1)))
    #     # _, x0, _, _, mu_x, log_std_x, _, _ = self.initial_encoder(y_true, t)
    #     #x0 = x0.unsqueeze(1)
    #     kl_loss_x0 = torch.mean(calc_KL(mu_x, log_std_x, self.py0_mean.repeat_interleave(mu_x.shape[0], dim=0),
    #                                     self.py0_logstd.repeat_interleave(mu_x.shape[0], dim=0)))
    #
    #     logk = torch.log(torch.tensor(k, device=self.device))
    #     x0 = x0.repeat_interleave(k, dim=0)
    #     t_repeat = t.repeat_interleave(k, dim=0)
    #     y_true = y_true.repeat_interleave(k, dim=0)
    #     # nan_ind = torch.isnan(y_true)
    #     # y_true[nan_ind] = 0.
    #
    #     y = self(t_repeat, x=x0)
    #     # distr = torch.distributions.Normal(y_true, self.alpha(torch.cat([x0.repeat_interleave(t.shape[-2], dim=-2), t_repeat],dim=-1)) * torch.ones_like(y_true, device=self.device))
    #     distr = torch.distributions.Laplace(y_true, self.alpha(
    #             torch.cat([x0.repeat_interleave(t.shape[-2], dim=-2), t_repeat], dim=-1)) * torch.ones_like(y_true,
    #                                                                                                         device=self.device))
    #     log_py = distr.log_prob(y) * mask
    #     log_py = log_py.view(-1, k, num_seq*dim).sum(dim=-1)
    #     loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk) + kl_loss_x0
    #     self.log(tag+'_loss', loss)
    #     return loss


    def training_step(self, batch, batch_idx,):
        loss = self.get_loss(batch, 'train', self.args.iwae_train)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self.get_loss(batch, 'val', self.args.iwae_test)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self.get_loss(batch, 'test', self.args.iwae_test)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.args.learn_std and self.args.std_ema_factor > 0:
            soft_update(self.alpha, self.alpha_previous, self.args.std_ema_factor)
            self.alpha_previous.load_state_dict(self.alpha.state_dict())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay)
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

def soft_update(target, src, factor=0.95):
  with torch.no_grad():
    for target_param, param in zip(target.parameters(), src.parameters()):
      target_param.data.mul_(1.0 - factor)
      target_param.data.add_(factor * param.data)


class Encoder(nn.Module):
    def __init__(self, dim, encoder_hidden_state_dim, z0_dim, z_all_dim, hidden_dims):
        super().__init__()
        self.encoder = CDE(dim, hidden_dims, encoder_hidden_state_dim, -1)
        self.z0_net = nn.Linear(encoder_hidden_state_dim, 2 * z0_dim)
        self.z_all_net = nn.Linear(encoder_hidden_state_dim, 2 * z_all_dim)

    def forward(self, x, t):
        zs = self.encoder(torch.cat([x, t], dim=-1), intermediate=True)
        mu_and_log_std_z0 = self.z0_net(zs[:, 0])
        mu_z0, log_std_z0 = torch.chunk(mu_and_log_std_z0, 2, -1)
        log_std_z0 = torch.clamp(log_std_z0, -20, 2)
        std_z0 = torch.exp(log_std_z0)
        distribution_z0 = torch.distributions.Normal(mu_z0, std_z0)
        z0 = distribution_z0.rsample()

        mu_and_log_std_za = self.z_all_net(zs[:, -1])
        mu_za, log_std_za = torch.chunk(mu_and_log_std_za, 2, -1)
        log_std_za = torch.clamp(log_std_za, -20, 2)
        std_za = torch.exp(log_std_za)
        distribution_za = torch.distributions.Normal(mu_za, std_za)
        za = distribution_za.rsample()

        z = torch.cat([z0, za], dim=-1)
        mu_z = torch.cat([mu_z0, mu_za], dim=-1)
        log_std_z = torch.cat([log_std_z0, log_std_za], dim=-1)

        return z, mu_z, log_std_z