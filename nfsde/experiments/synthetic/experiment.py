import time

import numpy as np
import torch
from copy import deepcopy
from torch.distributions.normal import Normal
from nfsde.models.flow import LatentCouplingFlow, LatentResnetFlow


from nfsde.experiments import BaseExperiment
from nfsde.experiments.synthetic.data import get_data_loaders
# from nfsde.models import ODEModel, CouplingFlow, ResNetFlow
from nfsde.models.variational import PosteriorLSTM
from nfsde.base_sde import Brownian

class Synthetic(BaseExperiment):
    def __init__(self, args, logger):
        self.z_dim = 0
        if args.is_latent:
            self.z_dim = args.z_dim
        super().__init__(args, logger)
        self.base_sde = Brownian(args.w_dim)
        self.prior = Normal(torch.zeros(args.z_dim).to(self.device), torch.ones(args.z_dim).to(self.device))
        self.prior_sampler = lambda N, num_seq: self.prior.sample(torch.Size([N, 1])).repeat_interleave(num_seq, dim=-2)
        self.posterior = self.get_posterior(args).to(self.device)
        self.optim_posterior = torch.optim.Adam(self.posterior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler_posterior = None
        if args.lr_scheduler_step > 0:
            self.scheduler_posterior = torch.optim.lr_scheduler.StepLR(self.optim_posterior, args.lr_scheduler_step,
                                                                         args.lr_decay)




    def get_model(self, args):
        if args.model == 'ode':
            return NotImplementedError
        elif args.model == 'flow':
            if args.flow_model == 'resnet':
                return LatentResnetFlow(self.dim, args.w_dim + self.z_dim, args.flow_layers,
                                      [args.hidden_dim] * args.hidden_layers,
                                      args.time_net, args.time_hidden_dim)
            else:
                return LatentCouplingFlow(self.dim, args.w_dim+self.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)

        raise NotImplementedError

    def get_posterior(self, args):
        return PosteriorLSTM(args.w_dim, self.dim, args.posterior_hidden_dim, args.hidden_dim, args.model, self.z_dim, args.flow_model, args.activation,
                             args.hidden_layers, args.time_net, args.time_hidden_dim )
        # return Posterior(self.dim, [args.hidden_dim] * args.hidden_layers, self.dim, args.activation)


    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size)


    # def _get_loss(self, batch, k=1):
    #     x, t, y_true = batch
    #     n, num_seq, dim = y_true.shape
    #     t_0 = torch.zeros((len(t), 1, 1)).to(t)
    #     t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
    #     t_j = t.clone()
    #     y_true_clone = y_true.clone()
    #     logk = torch.log(torch.Tensor([k]).to(x))
    #     log_prior = 0.
    #     x = x.repeat_interleave(k, dim=0)
    #     t = t.repeat_interleave(k, dim=0)
    #     y_true = y_true.repeat_interleave(k, dim=0)
    #     dt = (t_j - t_i + 1e-8).repeat_interleave(k, dim=0)
    #
    #     ws, z, log_q = self.posterior(y_true, dt, x)
    #     if z is not None:
    #         y = self.model(x, t, ws, z)
    #         log_prior += self.prior.log_prob(z).view(-1, k, self.z_dim).sum(dim=-1)
    #     else:
    #         y = self.model(x, t, ws)
    #     # assert y.shape == y_true.shape
    #     mean_martingale = ws.clone()
    #     mean_martingale[:, 1:] = ws.clone()[:, :-1]
    #     mean_martingale[:, 0:1] = 0.
    #     distr_p = Normal(mean_martingale, dt.sqrt())
    #     log_prior += distr_p.log_prob(ws).view(-1, k, num_seq * dim).sum(dim=-1)
    #
    #     data_distr = Normal(y_true, torch.ones_like(y).to(y))
    #     log_pyz = data_distr.log_prob(y).view(-1, k, num_seq * dim).sum(dim=-1)
    #     log_py = log_pyz + log_prior - log_q.view(-1, k)
    #     # loss = torch.exp(log_py / t.shape[-2] /100).sum(dim=-1)
    #     loss = - torch.mean(torch.logsumexp(log_py, dim=-1) - logk)
    #     obs_loss = (torch.mean(y.view(-1, k, num_seq, dim), dim=1) - y_true_clone) ** 2
    #
    #     # loss = -torch.mean(torch.log(loss+1e-8) - 1./t.shape[-2]/100 - logk)
    #     return loss, obs_loss.mean()

    def _get_loss(self, batch, k=1):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        t_0 = torch.zeros((len(t), 1, 1)).to(t)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        t_j = t.clone()
        y_true_clone = y_true.clone()
        logk = torch.log(torch.Tensor([k]).to(x))
        x = x.repeat_interleave(k, dim=0)
        t = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)
        dt = (t_j - t_i + 1e-8).repeat_interleave(k, dim=0)

        ws, log_q = self.posterior(y_true, dt, x)
        y = self.model(x, t, ws)
        # assert y.shape == y_true.shape
        log_prior = self.base_sde.log_prob(ws, t).view(-1, k)

        data_distr = Normal(y_true, torch.ones_like(y).to(y))
        log_pyz = data_distr.log_prob(y).view(-1, k, num_seq * dim).sum(dim=-1)
        log_py = log_pyz + log_prior - log_q.view(-1, k)
        # loss = torch.exp(log_py / t.shape[-2] /100).sum(dim=-1)
        loss = - torch.mean(torch.logsumexp(log_py, dim=-1) - logk)
        obs_loss = (torch.mean(y.view(-1, k, num_seq, dim), dim=1) - y_true_clone) ** 2

        return loss, obs_loss.mean()


    def _get_loss_on_dl(self, dl):
        losses = []
        data_losses = []
        for batch in dl:
            loss = self._get_loss(batch, self.args.iwae_test)
            losses.append(loss[0].item())
            data_losses.append(loss[1].item())
        return np.mean(losses), np.mean(data_losses)

    def training_step(self, batch):
        return self._get_loss(batch,self.args.iwae_train)[0]

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)


    def _sample_trajectories(self, path):
        N, M, T = 201, 100, 10
        x = torch.linspace(.1, 2., N).view(N, 1, 1).to(self.device)
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1).to(self.device)
        ws = self.base_sde.sample(t)
        if self.z_dim > 0:
            z = self.prior_sampler(N, M)
            y = self.model(x,t,torch.cat([ws,z], dim=-1))
        else:
            y = self.model(x, t, ws)
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())
        # np.savez(self.args.log_dir+'/ws.npz', x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=ws.detach().cpu().numpy())

    def finish(self):
        # dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        # dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)
        #
        # loss_time, data_loss_time  = self._get_loss_on_dl(dl_extrap_time)
        # loss_space, data_loss_space  = self._get_loss_on_dl(dl_extrap_space)

        # self.logger.info(f'loss_extrap_time={loss_time:.5f} data_loss={data_loss_time:.5f}')
        # self.logger.info(f'loss_extrap_space={loss_space:.5f} data_loss={data_loss_space:.5f}')
        self._sample_trajectories(self.args.log_dir+'/traj.npz')
        self.logger.info('Finished')

        ## Uncomment to save models
        OUT_DIR = self.args.log_dir
        torch.save(self.model.state_dict(), OUT_DIR + '/model.pt')
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
                self.optim_posterior.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                ## Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm_(self.posterior.parameters(), self.args.clip)
                self.optim.step()
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
        self.logger.info(f'test_loss={test_loss:.5f} data_loss={data_loss:.5f}')
