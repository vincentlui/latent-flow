import torch
from torch import optim, nn
import pytorch_lightning as pl
from nfsde.models.flow import LatentResnetFlow
from torch.distributions import Normal
from nfsde.base_sde import OU2, Brownian2
from nfsde.util import dotdict
from nfsde.models import CTFP
from nfsde.models.CTFP import CTFP2
from nfsde.models.variational import PosteriorZ, PosteriorCDE

class CTFPModule(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.optim = getattr(optim, 'Adam')#args.optim)
        self.base_sde = Brownian2(self.dim, train_param=False)
        self.model = self.get_model(args)
        self.prior = None
        if args.z_dim > 0:
            self.posterior = self.get_posterior(args)
        self.save_hyperparameters(args)

    def get_model(self, args):
        if args.flow_model == 'coupling':
            if args.z_dim > 0:
                return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, latent_dim=args.z_dim,
                            activation=args.activation, final_activation=args.final_activation)
            else:
                return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                            activation=args.activation, final_activation=args.final_activation)
        elif args.flow_model == 'ode':
            if args.z_dim > 0:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=args.z_dim+1,
                             activation=args.activation, final_activation=args.final_activation)
            else:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=1,
                             activation=args.activation, final_activation=args.final_activation)
        else:
            raise NotImplementedError

    def get_posterior(self, args):
        # return PosteriorZ(self.dim, args.posterior_hidden_dim, args.hidden_dim, args.z_dim, args.posterior_model, args.flow_model,
        #                   args.activation,
        #                   args.hidden_layers, args.time_net, args.time_hidden_dim)
        return PosteriorCDE(self.dim, args.encoder_hidden_state_dim, 0, args.flow_layers,
                            [args.encoder_hidden_dim] * args.encoder_hidden_layers,
                            [],
                            0, args.z_dim)


    def forward(self, t):
        ws = self.base_sde(t, normalize=False)
        z = None
        if self.args.z_dim > 0:
            z = torch.randn(t.shape[0], 1, self.args.z_dim, device=self.device)
        return self.model(ws, t, latent=z)

    def _get_loss(self, batch, tag, k=1):
        x, t, y_true = batch
        t = t[:, 1:]
        y_true = y_true[:, 1:]
        n, num_seq, dim = y_true.shape
        logk = torch.log(torch.tensor([k], device=self.device))

        if self.args.z_dim > 0:
            # t = t.repeat_interleave(k, dim=0)
            # y_true = y_true.repeat_interleave(k, dim=0)
            # mask = torch.ones_like(y_true)
            # z, mu, std, log_qz = self.posterior(y_true, t)
            # w, ljd = self.model.inverse(y_true, t, latent=z.unsqueeze(-2), return_jaco=True)
            # # assert w.shape == y_true.shape
            # log_pw = self.base_sde.log_prob(w, t)
            # log_pyz = torch.sum((log_pw + ljd).view(-1, k, num_seq * dim), dim=-1)
            # log_pz = self.prior.log_prob(z).sum(dim=-1, keepdims=True)
            # log_py = log_pyz + log_pz.view(-1, k) - log_qz.view(-1, k)
            # loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)

            t = t.repeat_interleave(k, dim=0)
            y_true = y_true.repeat_interleave(k, dim=0)
            _, z, _, _, _, _, _, log_qz = self.posterior(y_true, t, return_logp=True)
            w, ljd = self.model.inverse(y_true, t, latent=z.unsqueeze(-2), return_jaco=True)
            log_pw = self.base_sde.log_prob(w, t)
            log_pyz = torch.sum((log_pw + ljd).view(-1, k, num_seq) , dim=-1)
            log_pz = self.prior.log_prob(z).sum(dim=-1, keepdims=True)
            log_py = log_pyz + log_pz.view(-1, k) - log_qz.view(-1, k)
            if tag != 'train':
                loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk) / num_seq
            else:
                loss = -torch.mean(
                    torch.sum(torch.nn.functional.softmax(log_py, -1).detach() * log_py, dim=-1)) / num_seq

        else:
            w, ljd = self.model.inverse(y_true, t, return_jaco=True)
            assert w.shape == y_true.shape
            log_pw = self.base_sde.log_prob(w, t)
            loss = -torch.mean(torch.sum(log_pw + ljd, dim=-2) / num_seq)
        return loss

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
        optimizer = self.optim(self.parameters(), lr=self.args.lr,
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

