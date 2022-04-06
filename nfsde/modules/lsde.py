import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow
from torch.distributions import Normal
from nfsde.base_sde import OU2, Brownian2
from nfsde.util import dotdict, calc_KL
from stribor.net import MLP
from nfsde.models import CTFP
from nfsde.models.CTFP import CTFP2
from nfsde.models.variational import PosteriorZ, PosteriorCDE

from nfsde.models.latent_sde import latentSDE, RNNEncoder, LatentSDE2

class LSDE(pl.LightningModule):
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
        # self.encoder = self.get_encoder(args)
        self.py0_mean = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        self.py0_logstd = nn.Parameter(torch.zeros(1, args.hidden_state_dim))
        # if args.hidden_state_dim != -1:
        # self.py0_network = nn.Linear(args.encoder_hidden_state_dim, 2 * args.hidden_state_dim)
        self.qy0_network = nn.Linear(args.z_dim, 2 * args.hidden_state_dim)
        self.readout = nn.Linear(args.hidden_state_dim, self.dim)
        self.flow_dim = args.hidden_state_dim
        self.save_hyperparameters(args)

    def get_model(self, args):
        # return latentSDE(args.hidden_state_dim, args.z_dim,
        #           [args.hidden_dim] * args.hidden_layers + [args.hidden_state_dim],
        #           [args.hidden_dim] * args.hidden_layers + [args.hidden_state_dim], args)
        return LatentSDE2(self.dim, args.hidden_state_dim, args.z_dim, [args.hidden_dim] * args.hidden_layers,
                          args.encoder_hidden_state_dim, [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                          args.dt, args.method)

    def get_encoder(self, args):
        # return PosteriorZ(self.dim, args.posterior_hidden_dim, args.hidden_dim, args.z_dim, args.posterior_model, args.flow_model,
        #                   args.activation,
        #                   args.hidden_layers, args.time_net, args.time_hidden_dim)
        return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, args.flow_layers,
                            [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                            [],
                            0, args.z_dim)
        # return RNNEncoder(args.encoder_hidden_state_dim, args.encoder_hidden_state_dim,
        #                   encoder_network=MLP(self.dim + args.hidden_state_dim + 2,
        #                                       [args.encoder_hidden_dim] * args.encoder_hidden_layers,
        #                                       args.encoder_hidden_state_dim, activation='ReLU'))


    def sample(self, t, num_sample):
        y = self.model.sample_from_prior(num_sample, t)
        return y

    # def posterior_inference(self, x, t):
    #     t_repeat = t.unsqueeze(0).repeat_interleave(x.shape[0], 0).unsqueeze(-1)
    #     num_obs = len(t)
    #     y0_mean_logstd = self.qy0_network(torch.cat([x[:, 0], t_repeat[:, 0]], dim=-1))
    #     mu_y0, log_std_y0 = torch.chunk(y0_mean_logstd, 2, -1)
    #     std_y0 = torch.exp(log_std_y0)
    #     distribution_qy0 = Normal(mu_y0, std_y0)
    #     y = distribution_qy0.rsample()
    #     distribution_py0 = Normal(self.py0_mean, self.py0_logstd.exp())
    #
    #     h = torch.zeros((x.shape[0], self.args.encoder_hidden_state_dim), device=self.device)
    #     t_prev = torch.tensor(0, device=self.device)
    #     log_pqs = distribution_py0.log_prob(y).sum(-1, keepdims=True) \
    #               - distribution_qy0.log_prob(y).sum(-1, keepdims=True)
    #
    #     ys = []
    #
    #     for i in range(num_obs):
    #         t_curr = t[i]
    #         if t_curr == 0:
    #             ys.append(y.unsqueeze(1))
    #             continue
    #         # h = self.encoder(h, x[:, i], y, t_curr, t_prev,)
    #         context = h
    #         y_logpq = self.model(y, t_curr.unsqueeze(0), (context, t_prev))
    #         y = y_logpq[:, :-1]
    #         ys.append(y.unsqueeze(1))
    #         log_pq = -y_logpq[:, -1:]
    #         log_pqs += log_pq
    #         t_prev = t_curr
    #
    #     ys = torch.cat(ys, dim=1)
    #     ys = self.readout(ys)
    #
    #     # ws, ljd = self.model.inverse(x, t_repeat, latent=zs, return_jaco=True)
    #
    #     return ys, log_pqs

    # def posterior_inference(self, x, t):
    #     t_repeat = t.unsqueeze(0).repeat_interleave(x.shape[0], 0).unsqueeze(-1)
    #     num_obs = len(t)
    #
    #     _, z, _, _, mu_z, log_std_z, _, _ = self.encoder(x, t_repeat)
    #     y0_mean_logstd = self.qy0_network(mu_z)
    #     mu_y0, log_std_y0 = torch.chunk(y0_mean_logstd, 2, -1)
    #     std_y0 = torch.exp(log_std_y0)
    #     distribution_qy0 = Normal(mu_y0, std_y0)
    #     y = distribution_qy0.rsample()
    #     distribution_py0 = Normal(self.py0_mean, self.py0_logstd.exp())
    #
    #     t_prev = torch.tensor(0, device=self.device)
    #     log_pqs = distribution_py0.log_prob(y).sum(-1, keepdims=True) \
    #               - distribution_qy0.log_prob(y).sum(-1, keepdims=True)
    #     y_logpq = self.model.run(y, t, context_tensor_time=(mu_z, t_prev))
    #     ys = y_logpq[:, :, :-1]
    #     log_pq = -y_logpq[:, :, -1:].sum(dim=-2)
    #     log_pqs += log_pq
    #
    #     ys = self.readout(ys)
    #
    #     # ws, ljd = self.model.inverse(x, t_repeat, latent=zs, return_jaco=True)
    #
    #     return ys, log_pqs

    def posterior_inference(self, x, t):

        ys, log_pqs = self.model(x, t, adjoint=True, method=self.args.method)

        return ys, log_pqs




    def forward(self, t, num_sample):
        # self.model.clear_context()
        return self.sample(t, num_sample)

    def _get_loss(self, batch, tag,  k=1):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        logk = torch.log(torch.tensor([k], device=self.device))
        # self.model.clear_context()
        t = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)
        y, log_pqs = self.posterior_inference(y_true, t[0].flatten())
        distr = torch.distributions.Laplace(y_true, torch.ones_like(y_true))
        log_py = distr.log_prob(y)
        log_py = torch.sum(log_py.view(-1, k, num_seq*dim) , dim=-1) + log_pqs.view(-1, k)
        if tag != 'train':
            loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)
        else:
            loss = -torch.mean(torch.sum(torch.nn.functional.softmax(log_py, -1).detach() * log_py, dim=-1))

        return loss / num_seq

    def training_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'train', self.args.iwae_train)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'val', self.args.iwae_test)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx,):
        loss = self._get_loss(batch, 'test', self.args.iwae_test)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr,
                                             weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler_step > 0:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_scheduler_step, self.args.lr_decay)
            return [optimizer], [scheduler]

        return optimizer

    # def on_fit_start(self):
    #     if self.args.z_dim > 0:
    #         self.prior = Normal(torch.zeros(self.args.z_dim, device=self.device), torch.ones(self.args.z_dim, device=self.device))


    def train_dataloader(self):
        return self.dltrain

    def val_dataloader(self):
        return self.dlval

    def test_dataloader(self):
        return self.dltest

