import time

import numpy as np
import torch
from copy import deepcopy
from torch.distributions.normal import Normal


from nfsde.experiments import BaseExperiment
from nfsde.experiments.synthetic.data import get_data_loaders, get_single_loader
# from nfsde.models import ODEModel, CouplingFlow, ResNetFlow
from nfsde.models.variational import PosteriorLSTM
from nfsde.models.stochastic_flow import StochasticFlow2
from nfsde.base_sde import sample_brownian

class Synthetic(BaseExperiment):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.posterior = self.get_posterior(args)
        self.optim_posterior = torch.optim.Adam(self.posterior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler_posterior = None
        if args.lr_scheduler_step > 0:
            self.scheduler_posterior = torch.optim.lr_scheduler.StepLR(self.optim_posterior, args.lr_scheduler_step,
                                                                         args.lr_decay)



    def get_model(self, args):
        if args.model == 'ode':
            return NotImplementedError
        elif args.model == 'flow':
            return StochasticFlow2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)

        raise NotImplementedError

    def get_posterior(self, args):
        return PosteriorLSTM(self.dim*2, args.posterior_hidden_dim, args.hidden_dim, args.model, args.flow_model, args.activation,
                             args.hidden_layers, args.time_net, args.time_hidden_dim )
        # return Posterior(self.dim, [args.hidden_dim] * args.hidden_layers, self.dim, args.activation)


    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size)

    # def _get_loss(self, batch):
    #     x, t, y_true = batch
    #     ws, log_prob = sample_brownian(t, log_prob=True)
    #     y = self.model(x, t, ws)
    #     assert y.shape == y_true.shape
    #     observation_loss = torch.mean((y - y_true)**2)
    #     loss = torch.mean((y - y_true)**2 )#+ log_prob)
    #     return loss, observation_loss
    #
    # def _get_loss(self, batch):
    #     x, t, y_true = batch
    #     x_i = torch.cat([x, y_true[:, :-1]], dim=-2)
    #     x_j = y_true
    #     t_0 = torch.zeros((len(t), 1, 1), device=t.device)
    #     t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
    #     t_j = t
    #
    #     dw, mu, std, log_std, _ = self.posterior(x_i, x_j-x_i, t_i, t_j-t_i)
    #     # dw, mu, std, log_std, _ = self.posterior(x_j - x_i, t_j - t_i)
    #     ws = torch.cumsum(dw, dim=-2)#.detach()
    #     # ws = dw.clone()
    #     # ws[:, 1:] += dw_sum[:, :-1]
    #     y = self.model(x, t, ws)
    #     assert y.shape == y_true.shape
    #     dt = t_j - t_i
    #     kl_loss = torch.mean(.5 * (torch.sum(torch.log(dt + 1e-8) - 2*log_std + (mu**2 + std**2) / (dt+1e-8) - 1, dim=-2)))
    #     data_loss = torch.mean(torch.sum((y - y_true)**2, dim=-2))
    #     loss = data_loss + kl_loss
    #     return loss, data_loss, kl_loss

    # def _get_loss(self, batch):
    #     x, t, y_true = batch
    #     x_i = torch.cat([x, y_true[:, :-1]], dim=-2)
    #     x_j = y_true
    #     t_0 = torch.zeros((len(t), 1, 1), device=t.device)
    #     t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
    #     t_j = t
    #
    #     ws, mu, std, log_std, log_q = self.posterior(y_true, t_j - t_i, x)
    #     y = self.model(x, t, ws)
    #     assert y.shape == y_true.shape
    #     dt = t_j - t_i + 1e-8
    #     dw = ws
    #     dw[:, 1:] -= dw[:, :-1]
    #     distr_p = Normal(torch.zeros_like(ws), dt.sqrt())
    #     log_p = distr_p.log_prob(dw)
    #     # kl_loss = torch.mean(torch.sum(log_q - log_p, dim=-2))
    #     # data_loss = torch.mean(torch.sum((y - y_true) ** 2, dim=-2))
    #     kl_loss = torch.mean(log_q - log_p)
    #     data_loss = torch.mean((y - y_true) ** 2)
    #     loss = data_loss + kl_loss
    #     return loss, data_loss, kl_loss

    def _get_loss(self, batch):
        k = 1
        x, t, y_true = batch
        t_0 = torch.zeros((len(t), 1, 1), device=t.device)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        t_j = t

        loss=0
        obs_loss=0
        for i in range(k):
            # ws, mu, std, log_std, log_q = self.posterior(y_true, t, x)
            ws, log_q = self.posterior(y_true, t, x)
            y = self.model(x, t, ws)
            assert y.shape == y_true.shape
            dt = t_j - t_i + 1e-8
            dw = ws
            dw[:, 1:] -= dw[:, :-1]
            distr_p = Normal(torch.zeros_like(ws), dt.sqrt())
            log_p = distr_p.log_prob(dw)
            data_distr = Normal(y_true, torch.ones_like(y, device=y.device))
            log_py = data_distr.log_prob(y)
            loss += torch.exp(log_py + log_p - log_q)
            obs_loss += (y - y_true) ** 2

        loss = -torch.mean(torch.log(loss+1e-8) - torch.log(torch.ones_like(loss)*k))
        return loss, obs_loss.mean()


    # def _get_loss(self, batch):
    #     repeat_num = 3
    #     x, t, y_true = batch
    #     x_i = torch.cat([x, y_true[:, :-1]], dim=-2)
    #     x_j = y_true
    #     t_0 = torch.zeros((len(t), 1, 1), device=t.device)
    #     t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
    #     t_j = t
    #
    #     x = x.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     t = t.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     y_true = y_true.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     x_i = x_i.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     x_j = x_j.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     t_i = t_i.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     t_j = t_j.unsqueeze(-2).repeat_interleave(repeat_num, dim=-2)
    #     dt = t_j - t_i
    #     dw, mu, std, log_std, log_q = self.posterior(x_i, x_j-x_i, t_i, t_j-t_i)
    #     # dw, mu, std, log_std, log_q = self.posterior(x_j - x_i, t_j - t_i)
    #     prior_distr = torch.distributions.Normal(torch.zeros_like(mu, device=mu.device), dt.sqrt())
    #     log_p = prior_distr.log_prob(dw)
    #     ws = torch.cumsum(dw, dim=-3)#.detach()
    #     y = self.model(x, t, ws)
    #     assert y.shape == y_true.shape
    #     data_distr = torch.distributions.Normal(y_true, torch.ones_like(y, device=y.device))
    #     log_py = -(y - y_true)**2 #data_distr.log_prob(y)
    #     observation_error = torch.mean((y - y_true)**2)
    #     loss = -torch.mean(torch.logsumexp(log_py + log_p - log_q, dim=-2))
    #     return loss, observation_error

    def _get_loss_on_dl(self, dl):
        losses = []
        data_losses = []
        for batch in dl:
            losses.append(self._get_loss(batch)[0].item())
            data_losses.append(self._get_loss(batch)[1].item())
        return np.mean(losses), np.mean(data_losses)

    def training_step(self, batch):
        return self._get_loss(batch)[0]

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)

    def _sample_trajectories(self, path):
        N, M, T = 21, 100, 10
        x = torch.linspace(-2, 2, N).view(N, 1, 1)
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1)
        ws = sample_brownian(t)
        y = self.model(x, t, ws)
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())

    def finish(self):
        dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)

        loss_time, data_loss_time  = self._get_loss_on_dl(dl_extrap_time)
        loss_space, data_loss_space  = self._get_loss_on_dl(dl_extrap_space)

        self.logger.info(f'loss_extrap_time={loss_time:.5f} data_loss={data_loss_time:.5f}')
        self.logger.info(f'loss_extrap_space={loss_space:.5f} data_loss={data_loss_space:.5f}')
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
