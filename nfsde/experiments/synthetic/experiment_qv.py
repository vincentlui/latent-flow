import time

import numpy as np
import torch
from copy import deepcopy
from torch.distributions.normal import Normal


from nfsde.experiments import BaseExperiment
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.models.flow import LatentResnetFlow
from nfsde.models.variational import PosteriorZ
from nfsde.base_sde import OU

def calc_qv(x0, t, y):
    y_i = torch.cat([x0, y[:, :-1]], dim=-2)
    qv = (y-y_i)**2
    qv = torch.sum((y - y_i) ** 2, dim=-2, keepdim=True)

    return qv

class Synthetic_qv(BaseExperiment):
    def __init__(self, args, logger):
        self.z_dim = 0
        if args.is_latent:
            self.z_dim = args.z_dim
        super().__init__(args, logger)
        # self.base_sde = OU(self.dim, .5, .5, 0.)
        self.base_sde = OU(args.w_dim)
        # self.qv = self.get_qv_true()
        # self.logger.info(f'[QV={self.qv.item()}]')
        # self.posterior = self.get_posterior(args)
        # self.optim_posterior = torch.optim.Adam(self.posterior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optim_sde = torch.optim.Adam(self.base_sde.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.scheduler_posterior = None
        # if args.lr_scheduler_step > 0:
        #     self.scheduler_posterior = torch.optim.lr_scheduler.StepLR(self.optim_posterior, args.lr_scheduler_step,
        #                                                                args.lr_decay)
        self.prior = Normal(torch.zeros(args.z_dim).to(self.device), torch.ones(args.z_dim).to(self.device))
        self.prior_sampler = lambda N: self.prior.sample(torch.Size([N,1]))


    def get_qv_true(self):
        qv = 0.
        data_size = 0.
        for batch in self.dltrain:
            x, t, y_true = batch
            batch_size = x.shape[0]
            if data_size == 0:
                qv = torch.mean(calc_qv(x, t, y_true), 0, keepdim=True)
            else:
                ratio = batch_size/data_size
                qv = (1-ratio) * qv + ratio * torch.mean(calc_qv(x, t, y_true), 0, keepdim=True)
            data_size += batch_size
        return qv


    def get_model(self, args):
        if args.model == 'ode':
            return NotImplementedError
        elif args.model == 'flow':
            # return CouplingFlow(
            #     dim=self.dim,
            #     n_layers=args.flow_layers,
            #     hidden_dims=[args.hidden_dim] * args.hidden_layers,
            #     time_net=args.time_net,
            #     time_hidden_dim=args.time_hidden_dim
            # )
            return LatentResnetFlow(self.dim, args.w_dim+self.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)
            # return LatentCouplingFlow(self.dim, 1+self.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
            #                         args.time_net, args.time_hidden_dim)

        raise NotImplementedError


    def get_posterior(self, args):
        return PosteriorZ(self.dim, args.posterior_hidden_dim, args.hidden_dim, args.z_dim, args.model, args.flow_model, args.activation,
                             args.hidden_layers, args.time_net, args.time_hidden_dim)


    def get_data(self, args):
        return get_data_loaders(args.data, args.batch_size)

    def _get_loss(self, batch, k=20):
        x, t, y_true = batch
        n, num_seq, dim = y_true.shape
        y_true_clone = y_true.clone()
        logk = torch.log(torch.Tensor([k]).to(x))
        x0 = x.repeat_interleave(k, dim=0)
        t = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)

        ws = self.base_sde(t)
        z = None
        if self.z_dim > 0:
            z = self.prior_sampler(k*n)
        y = self.model(x0, t, ws, z)
        assert y.shape == y_true.shape
        # t_qv = torch.linspace(0, 9, 100).reshape(1,100,1).repeat_interleave(x.shape[0], dim=0)
        # ws_qv= self.base_sde.sample(t)(t_qv, log_prob=False)
        # qv = calc_qv(x, t_qv, self.model(x, t_qv, ws_qv))
        distr = torch.distributions.Normal(y_true, torch.ones_like(y_true).to(x))
        log_py = distr.log_prob(y).view(-1, k, num_seq*dim).sum(dim=-1)
        mse = torch.mean((y.view(-1, k, num_seq, dim).mean(dim=1) - y_true_clone) ** 2)
        # qv_loss = torch.mean((qv - self.qv) ** 2)
        loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk) #+ torch.mean(qv_loss)
        return loss, mse, torch.zeros(1)

    # def _get_loss(self, batch, k=20):
    #     x, t, y_true = batch
    #     n, num_seq, dim = y_true.shape
    #     y_true_clone = y_true.clone()
    #
    #     y = self.model(x, t)
    #     assert y.shape == y_true.shape
    #     distr = torch.distributions.Normal(y_true, torch.ones_like(y_true).to(x))
    #     observation_loss = distr.log_prob(y).view(-1, num_seq * dim).sum(dim=-1)
    #     mse = torch.mean((y - y_true_clone) ** 2)
    #     loss = -torch.mean(observation_loss)  # + torch.mean(qv_loss)
    #     return loss, mse, torch.zeros(1)


    def _get_loss_on_dl(self, dl):
        losses = []
        data_losses = []
        qv_losses = []
        for batch in dl:
            loss = self._get_loss(batch, self.args.iwae_test)
            losses.append(loss[0].item())
            data_losses.append(loss[1].item())
            qv_losses.append(loss[2].item())
        return np.mean(losses), np.mean(data_losses), np.mean(qv_losses)

    def training_step(self, batch):
        return self._get_loss(batch, self.args.iwae_train)[0]

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)

    def _sample_trajectories(self, path):
        N, M, T = 201, 100, 10
        x = torch.linspace(.1, 2., N).view(N, 1, 1).to(self.device)
        x = (x-self.mean)/self.std
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1).to(self.device)
        z = None
        if self.z_dim > 0:
            z = self.prior_sampler(N).to(self.device)
        ws = self.base_sde(t).to(x)
        y = self.model(x, t, ws, z)
        y = self.std * y + self.mean
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())
        # np.savez(self.args.log_dir+'/ws.npz', x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=ws.detach().cpu().numpy())

    def finish(self):
        # dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        # dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)

        # loss_time, data_loss_time  = self._get_loss_on_dl(dl_extrap_time)
        # loss_space, data_loss_space  = self._get_loss_on_dl(dl_extrap_space)

        # self.logger.info(f'loss_extrap_time={loss_time:.5f} data_loss={data_loss_time:.5f}')
        # self.logger.info(f'loss_extrap_space={loss_space:.5f} data_loss={data_loss_space:.5f}')
        self.logger.info('Finished')

        ## Uncomment to save models
        OUT_DIR = self.args.log_dir
        torch.save(self.model.state_dict(), OUT_DIR + '/model.pt')
        torch.save(self.base_sde.state_dict(), OUT_DIR + '/sde.pt')
        self._sample_trajectories(self.args.log_dir + '/traj.npz')


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
                # self.optim_posterior.zero_grad()
                self.optim_sde.zero_grad()
                train_loss = self.training_step(batch)
                train_loss.backward()
                ## Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()
                # self.optim_posterior.step()
                self.optim_sde.step()

                self.logger.info(f'[epoch={epoch+1:04d}|iter={iteration+1:04d}] train_loss={train_loss:.5f}')
                iteration += 1

            epoch_duration = time.time() - start_time
            durations.append(epoch_duration)
            self.logger.info(f'[epoch={epoch+1:04d}] epoch_duration={epoch_duration:5f}')

            # Validation step
            self.model.eval()
            val_loss, data_loss, qv_loss = self.validation_step()
            self.logger.info(f'[epoch={epoch+1:04d}] val_loss={val_loss:.5f} data_loss={data_loss:.5f} qv_loss={qv_loss:.5f}')

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                # self.scheduler_posterior.step()

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
        test_loss, data_loss, qv_loss = self.test_step()
        self.logger.info(f'test_loss={test_loss:.5f} data_loss={data_loss:.5f} qv_loss={qv_loss:.5f}')
