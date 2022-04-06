import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow
from torch.distributions import Normal
from nfsde.base_sde import OU2, Brownian2
from nfsde.util import dotdict, calc_KL
from nfsde.models import CTFP
from nfsde.models.CTFP import CTFP2
from nfsde.models.variational import PosteriorZ, PosteriorCDE

from nfsde.models.latent_sde import latentSDE, RNNEncoder

class CLPFModule(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.base_sde = Brownian2(self.dim, train_param=False)
        self.model = self.get_model(args)
        self.prior = None
        if args.z_dim > 0:
            self.encoder = self.get_encoder(args)
        self.py0_mean = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        self.py0_logstd = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        # if args.hidden_state_dim != -1:
        self.py0_network = nn.Linear(self.dim, 2 * args.hidden_state_dim)
        self.qy0_network = nn.Linear(self.dim + 1, 2 * args.hidden_state_dim)
        self.y_proj_z = nn.Linear(args.hidden_state_dim, args.z_dim)
        self.flow_dim = args.hidden_state_dim
        self.latent_sde = latentSDE(args.hidden_state_dim, args.encoder_hidden_state_dim,
                                    [args.encoder_hidden_dim] * args.encoder_hidden_layers + [args.hidden_state_dim],
                                    [args.encoder_hidden_dim] * args.encoder_hidden_layers + [args.hidden_state_dim], args)
        self.save_hyperparameters(args)

    def get_model(self, args):
        if args.z_dim > 0:
            return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, latent_dim=args.z_dim)
        else:
            return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers)
        # elif args.model == 'ctfp-ode':
        #     if args.z_dim > 0:
        #         return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=args.z_dim+1)
        #     else:
        #         return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=1)

    def get_encoder(self, args):
        # return PosteriorZ(self.dim, args.posterior_hidden_dim, args.hidden_dim, args.z_dim, args.posterior_model, args.flow_model,
        #                   args.activation,
        #                   args.hidden_layers, args.time_net, args.time_hidden_dim)
        # return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, args.flow_layers,
        #                     [args.encoder_hidden_dim] * args.encoder_hidden_layers,
        #                     [],
        #                     0, args.z_dim)
        return RNNEncoder(self.dim + args.hidden_state_dim + 7, args.encoder_hidden_state_dim)


    def sample(self, t, num_sample):
        t_repeat = t.unsqueeze(0).repeat_interleave(num_sample, 0).unsqueeze(-1)
        y0_mean = self.py0_mean.squeeze(0)#.repeat_interleave(num_sample, dim=0)
        y0_std = self.py0_logstd.exp().squeeze(0)#.repeat_interleave(num_sample, dim=0)
        y0_distr = Normal(y0_mean, y0_std)
        y0 = y0_distr.sample(torch.Size([num_sample]))
        y_prior = self.latent_sde.sample_from_prior(y0, t).transpose(0, 1)
        z = self.y_proj_z(y_prior)
        ws = self.base_sde(t_repeat, normalize=False)
        x = self.model(ws, t_repeat, latent=z)
        return x

    def posterior_inference(self, x, t):
        t_repeat = t.unsqueeze(0).repeat_interleave(x.shape[0], 0).unsqueeze(-1)
        num_obs = len(t)
        y0_mean_logstd = self.qy0_network(torch.cat([x[:, 0], t_repeat[:, 0]], dim=-1))
        mu_y0, log_std_y0 = torch.chunk(y0_mean_logstd, 2, -1)
        std_y0 = torch.exp(log_std_y0)
        distribution_qy0 = Normal(mu_y0, std_y0)
        y = distribution_qy0.rsample()
        distribution_py0 = Normal(self.py0_mean, self.py0_logstd.exp())

        h = torch.zeros((x.shape[0], self.args.encoder_hidden_state_dim), device=self.device)
        t_prev = torch.tensor(0, device=self.device)
        log_pqs = distribution_py0.log_prob(y).sum(-1, keepdims=True) \
                  - distribution_qy0.log_prob(y).sum(-1, keepdims=True)

        ys = []

        for i in range(num_obs):
            t_curr = t[i]
            if t_curr == 0:
                ys.append(y.unsqueeze(1))
                continue
            h = self.encoder(h, x[:, i], y, t_curr, t_prev,)
            context = h
            y_logpq = self.latent_sde(y, t_curr.unsqueeze(0), (context, t_prev))
            y = y_logpq[:, :-1]
            ys.append(y.unsqueeze(1))
            log_pq = -y_logpq[:, -1:]
            log_pqs += log_pq
            t_prev = t_curr

        ys = torch.cat(ys, dim=1)
        zs = self.y_proj_z(ys)

        # ws, ljd = self.model.inverse(x, t_repeat, latent=zs, return_jaco=True)

        return zs, log_pqs

    def forward(self, t, num_sample):
        self.latent_sde.clear_context()
        return self.sample(t, num_sample)

    def _get_loss(self, batch, k=1):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        logk = torch.log(torch.tensor([k], device=self.device))
        self.latent_sde.clear_context()
        if self.args.z_dim > 0:
            t = t.repeat_interleave(k, dim=0)
            y_true = y_true.repeat_interleave(k, dim=0)
            z, log_pqs = self.posterior_inference(y_true, t[0].flatten())
            w, ljd = self.model.inverse(y_true, t, latent=z, return_jaco=True)
            log_pw = self.base_sde.log_prob(w, t)
            log_py = torch.sum((log_pw + ljd).view(-1, k, num_seq) , dim=-1) + log_pqs.view(-1, k)
            loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)

        return loss

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, self.args.iwae_train)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, self.args.iwae_test)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, self.args.iwae_test)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler_step > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, self.args.lr_decay)
            return [optimizer], [scheduler]

        return optimizer

    def on_fit_start(self):
        if self.args.z_dim > 0:
            self.prior = Normal(torch.zeros(self.args.z_dim, device=self.device), torch.ones(self.args.z_dim, device=self.device))


    def train_dataloader(self):
        return self.dltrain

    def val_dataloader(self):
        return self.dlval

    def test_dataloader(self):
        return self.dltest

