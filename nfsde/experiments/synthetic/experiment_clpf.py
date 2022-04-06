import time

import numpy as np
import torch
from copy import deepcopy
from torch.distributions.normal import Normal


from nfsde.experiments import BaseExperiment
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.models.flow import LatentCouplingFlow
from nfsde.models.CTFP import CTFP, CTFP2

from nfsde.models.variational import PosteriorZ
from nfsde.base_sde import Brownian


class Synthetic_clpf(BaseExperiment):
    def __init__(self, args, logger):
        self.z_dim = args.z_dim
        super().__init__(args, logger)
        self.model2 = self.get_ctfp_model(args).to(self.device)
        # self.base_sde = OU(10, .5, .5, 0.)
        self.base_sde = Brownian(1)
        self.base_sde2 = Brownian(1)
        self.posterior = self.get_posterior(args)
        self.optim_posterior = torch.optim.Adam(self.posterior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler_posterior = None
        if args.lr_scheduler_step > 0:
            self.scheduler_posterior = torch.optim.lr_scheduler.StepLR(self.optim_posterior, args.lr_scheduler_step,
                                                                       args.lr_decay)
        self.prior = Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))
        self.prior_sampler = lambda N: self.prior.sample(torch.Size([N,1]))


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
            # return LatentResnetFlow(args.z_dim, 1, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
            #                         args.time_net, args.time_hidden_dim)
            return LatentCouplingFlow(args.z_dim, 1+self.z_dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)

        raise NotImplementedError

    def get_ctfp_model(self, args):
        if args.model == 'flow':
            return CTFP(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, latent_dim=args.z_dim)

        elif args.model == 'ode':
            if args.is_latent:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                             anode_dim=args.z_dim + 1)
            else:
                return CTFP2(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers, anode_dim=1)

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

        x0 = torch.zeros((n*k, 1, self.z_dim))
        t = t.repeat_interleave(k, dim=0)
        y_true = y_true.repeat_interleave(k, dim=0)
        logk = torch.log(torch.Tensor([k]).to(x))

        ws = self.base_sde.sample(t)
        z = self.model(x0, t, ws)

        w, ljd = self.model2(y_true, t, latent=z, return_jaco=True)
        log_pyz = self.base_sde.log_prob(w, t)
        log_py = log_pyz.view(-1, k) + torch.sum(ljd.view(-1, k, num_seq * dim), dim=-1)
        loss = -torch.mean(torch.logsumexp(log_py, dim=-1) - logk)

        ws_z = self.base_sde.sample(t)
        z_mse = self.model(x0, t, ws_z)
        ws_mse = self.base_sde2.sample(t)
        y_mse = self.model2.inverse(ws_mse, t, latent=z_mse)
        mse = torch.mean((y_mse.view(-1, k, num_seq, dim).mean(dim=1) - y_true_clone) ** 2)
        return loss, mse



    def _get_loss_on_dl(self, dl):
        losses = []
        data_losses = []
        for batch in dl:
            loss = self._get_loss(batch, self.args.iwae_test)
            losses.append(loss[0].item())
            data_losses.append(loss[1].item())
        return np.mean(losses), np.mean(data_losses)

    def training_step(self, batch):
        return self._get_loss(batch, self.args.iwae_train)[0]

    def validation_step(self):
        return self._get_loss_on_dl(self.dlval)

    def test_step(self):
        return self._get_loss_on_dl(self.dltest)

    def _sample_trajectories(self, path):
        N, M, T = 201, 100, 10
        x = torch.linspace(.1, 2., N).view(N, 1, 1).to(self.device)
        t = torch.linspace(0, T, M).view(1, M, 1).repeat(N, 1, 1).to(self.device)
        z = None
        if self.z_dim > 0:
            z = self.prior_sampler(N).to(self.device)
        ws = self.base_sde.sample(t).to(x)
        y = self.model(x, t, ws, z)
        np.savez(path, x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=y.detach().cpu().numpy())
        # np.savez(self.args.log_dir+'/ws.npz', x=x.detach().cpu().numpy(), t=t.detach().cpu().numpy(), y=ws.detach().cpu().numpy())

    def finish(self):
        # dl_extrap_time = get_single_loader(f'{self.args.data}_extrap_time', self.args.batch_size)
        # dl_extrap_space = get_single_loader(f'{self.args.data}_extrap_space', self.args.batch_size)

        # loss_time, data_loss_time  = self._get_loss_on_dl(dl_extrap_time)
        # loss_space, data_loss_space  = self._get_loss_on_dl(dl_extrap_space)

        # self.logger.info(f'loss_extrap_time={loss_time:.5f} data_loss={data_loss_time:.5f}')
        # self.logger.info(f'loss_extrap_space={loss_space:.5f} data_loss={data_loss_space:.5f}')
        self._sample_trajectories(self.args.log_dir+'/traj.npz')
        self.logger.info('Finished')

        ## Uncomment to save models
        OUT_DIR = self.args.log_dir
        torch.save(self.model.state_dict(), OUT_DIR + '/model.pt')


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
