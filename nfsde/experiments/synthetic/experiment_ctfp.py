import time

import numpy as np
import torch
from copy import deepcopy
from torch.distributions.normal import Normal
from nfsde.models import CTFP
from nfsde.models.CTFP import CTFP2


from nfsde.experiments import BaseExperiment
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.models.variational import PosteriorZ
from nfsde.base_sde import Brownian


class Synthetic_CTFP(BaseExperiment):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.posterior = None
        self.base_sde = Brownian(self.dim)
        if args.is_latent:
            self.posterior = self.get_posterior(args).to(self.device)
            self.optim_posterior = torch.optim.Adam(self.posterior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler_posterior = None
            if args.lr_scheduler_step > 0:
                self.scheduler_posterior = torch.optim.lr_scheduler.StepLR(self.optim_posterior, args.lr_scheduler_step,
                                                                           args.lr_decay)
            self.prior = Normal(torch.zeros(args.z_dim).to(self.device), torch.ones(args.z_dim).to(self.device))
            self.prior_sampler = lambda N: self.prior.sample(torch.Size([N,1]))


    def get_model(self, args):
        if args.model == 'flow':
            if args.is_latent:
                return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, latent_dim=args.z_dim)
            else:
                return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers)
        elif args.model == 'ode':
            if args.is_latent:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=args.z_dim+1)
            else:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=1)

    def get_posterior(self, args):
        return PosteriorZ(self.dim, args.posterior_hidden_dim, args.hidden_dim, args.z_dim, args.model, args.flow_model,
                          args.activation,
                          args.hidden_layers, args.time_net, args.time_hidden_dim)


    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size)

    def _get_loss_latent(self, batch, k=1):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        y_true_clone = y_true.clone()
        t_0 = torch.zeros((len(t), 1, 1)).to(t)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        t_j = t.clone()
        x = x.repeat_interleave(k, dim=0)
        t = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)
        dt = (t_j - t_i + 1e-8).repeat_interleave(k, dim=0)
        logk = torch.log(torch.Tensor([k]).to(x))

        z, mu, std, log_qz = self.posterior(y_true, dt)
        w, ljd = self.model.inverse(y_true, t, latent=z.unsqueeze(-2), return_jaco=True)
        # assert w.shape == y_true.shape
        mean_martingale = w.clone()
        mean_martingale[:, 1:] = w.clone()[:, :-1]
        mean_martingale[:, 0:1] = 0.
        distr_p = Normal(mean_martingale, dt.sqrt())
        log_pyz = torch.sum((distr_p.log_prob(w) + ljd).view(-1, k, num_seq* dim), dim=-1)
        log_pz = self.prior.log_prob(z).sum(dim=-1, keepdims=True)
        log_py = log_pyz + log_pz.view(-1, k) - log_qz.view(-1, k)
        # loss = torch.exp(log_py / t.shape[-2]).sum(dim=-2)
        # loss = -torch.mean(torch.log(loss + 1e-8) - 1. / t.shape[-2] - logk)
        loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)
        w_sample = self.base_sde.sample(t)
        if self.args.is_latent:
            y = self.model(w_sample, t, latent=z.unsqueeze(-2))
        else:
            y = self.model(w_sample, t)
        mse = torch.mean((y.view(-1, k, num_seq, dim).mean(dim=1) - y_true_clone) ** 2)
        return loss, mse

    def _get_loss(self, batch):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        y_true_clone = y_true.clone
        t_0 = torch.zeros((len(t), 1, 1)).to(t)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        dt = t - t_i + 1e-8
        w, ljd = self.model.inverse(y_true, t, return_jaco=True)
        assert w.shape == y_true.shape
        mean_martingale = w.clone()
        mean_martingale[:, 1:] = w.clone()[:, :-1]
        mean_martingale[:, 0:1] = 0.
        distr_p = Normal(mean_martingale, dt.sqrt())
        log_py = distr_p.log_prob(w)

        loss = -torch.mean(torch.sum(log_py + ljd, dim=-2))
        # mse = torch.mean((y.view(-1, k, num_seq, dim).mean(dim=1) - y_true_clone) ** 2)
        return loss, loss


    def _get_loss_on_dl(self, dl):
        losses = []
        data_losses = []
        for batch in dl:
            if self.args.is_latent:
                loss = self._get_loss_latent(batch)
            else:
                loss = self._get_loss(batch)
            losses.append(loss[0].item())
            data_losses.append(loss[1].item())
        return np.mean(losses), np.mean(data_losses)

    def training_step(self, batch):
        if self.args.is_latent:
            return self._get_loss_latent(batch)[0]
        return self._get_loss(batch)[0]

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)

    def _sample_trajectories(self, path):
        N, M, T = 201, 100, 10
        x = torch.linspace(.1, 2., N).view(N, 1, 1).to(self.device)
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1).to(self.device)
        ws = self.base_sde.sample(t)
        if self.args.is_latent:
            z = self.prior_sampler(N)
            y = self.model(ws, t, latent=z)
        else:
            y = self.model(ws, t)

        y = self.std * y + self.mean
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())
        # np.savez(self.args.log_dir+'/ws.npz', x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=ws.detach().cpu().numpy())

    def finish(self):
        # dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        # dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)
        #
        # loss_time, data_loss_time  = self._get_loss_on_dl(dl_extrap_time)
        # loss_space, data_loss_space  = self._get_loss_on_dl(dl_extrap_space)
        #
        # self.logger.info(f'loss_extrap_time={loss_time:.5f} data_loss={data_loss_time:.5f}')
        # self.logger.info(f'loss_extrap_space={loss_space:.5f} data_loss={data_loss_space:.5f}')
        self._sample_trajectories(self.args.log_dir+'/traj.npz')
        self.logger.info('Finished')

        ## Uncomment to save models
        OUT_DIR = self.args.log_dir
        torch.save(self.model.state_dict(), OUT_DIR + '/model.pt')
        if self.posterior is not None:
            torch.save(self.posterior.state_dict(), OUT_DIR + '/posterior.pt')


    def train(self) -> None:
        # Training loop parameters
        best_loss = float('inf')
        waiting = 0
        durations = []
        best_model = deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            iteration = 0

            self.model.train()
            start_time = time.time()

            for batch in self.dltrain:
                # Single training step
                self.optim.zero_grad()
                if self.posterior is not None:
                    self.optim_posterior.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                ## Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()
                if self.posterior is not None:
                    self.optim_posterior.step()

                self.logger.info(f'[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}')
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(f'[epoch={epoch+1:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            val_loss, data_loss = self.validation_step()
            self.logger.info(f'[epoch={epoch+1:04d}] val_loss={val_loss:.5f} data_loss={data_loss:.5f}')

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                if self.posterior is not None:
                    self.scheduler_posterior.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
                waiting = 0
            elif waiting > self.patience:
                break
            else:
                waiting += 1

        self.logger.info(f'epoch_duration_mean={np.mean(durations):.5f}')

        # Load best model
        self.model.load_state_dict(best_model)

        # Held-out test set step
        test_loss, data_loss = self.test_step()
        self.logger.info(f'test_loss={test_loss:.5f}, data_loss={data_loss:.5f}')
