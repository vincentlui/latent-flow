from torch import optim, nn
import pytorch_lightning as pl
from nfsde.experiments.synthetic.data import get_data_loaders
import stribor as st

import torch
import torch.optim.swa_utils as swa_utils
from nfsde.models.flow import LatentResnetFlow, LatentCouplingFlow
from nfsde.models.discriminator import CDEDiscriminator, LSTMFlowDiscriminator
from nfsde.base_sde import OU2, Brownian2, Brownian_OU
from nfsde.util import dotdict


def get_flow(flow_model_name, dim, w_dim, z_dim, flow_layers, hidden_dims, time_net, time_hidden_dim, args, activation=None, final_activation=None):

    if flow_model_name == 'resnet':
        flow = LatentResnetFlow(
            dim,
            w_dim,
            z_dim,
            flow_layers,
            hidden_dims,
            time_net=time_net,
            time_hidden_dim=time_hidden_dim,
            activation=activation,
            final_activation=final_activation,
            invertible=args.invertible
        )
    elif flow_model_name == 'coupling':
        flow = LatentCouplingFlow(
            dim,
            w_dim,
            z_dim,
            flow_layers,
            hidden_dims,
            time_net=time_net,
            time_hidden_dim=time_hidden_dim,
            activation=activation,
            final_activation=final_activation,
        )
    else:
        raise NotImplementedError
    return flow

class Generator(nn.Module):
    def __init__(self, dim, n_classes,  args):
        super().__init__()
        self.args = args
        if args.hidden_state_dim > 0:
            self.flow = get_flow(
                args.flow_model,
                args.hidden_state_dim,
                args.w_dim,
                args.z_dim,
                args.flow_layers,
                [args.hidden_dim] * args.hidden_layers,
                time_net=args.time_net,
                time_hidden_dim=args.time_hidden_dim,
                args=args,
                activation=args.activation
            )
            self._initial = st.net.MLP(
                args.initial_noise_size,
                [args.d_hidden_dim] * args.d_hidden_layers,
                args.hidden_state_dim,
                activation=args.activation,
            )
            if args.base_sde == 'ou':
                self.base_sde = OU2(args.w_dim)
            elif args.base_sde == 'combine':
                self.base_sde = Brownian_OU(args.w_dim, theta=args.theta, sigma=args.sigma)
            else:
                self.base_sde = Brownian2(args.w_dim)
            self._readout = torch.nn.Linear(args.hidden_state_dim, dim)
        # else:
        #     self.flow = get_flow(
        #         args.flow_model,
        #         dim,
        #         args.w_dim + args.z_dim + n_classes,
        #         args.flow_layers,
        #         [args.hidden_dim] * args.hidden_layers,
        #         time_net=args.time_net,
        #         time_hidden_dim=args.time_hidden_dim,
        #         activation=args.activation
        #     )
        #     self._initial = st.net.MLP(
        #         args.initial_noise_size,
        #         [args.d_hidden_dim] * args.d_hidden_layers,
        #         dim,
        #         activation=args.activation,
        #     )
        #     self.base_sde = OU2(args.w_dim)
        #     self._readout = torch.nn.Identity()

    def forward(self, ts, label=None):
        init_noise = torch.randn(ts.shape[0], 1, self.args.initial_noise_size).type_as(ts)
        x0 = self._initial(init_noise)
        z = torch.randn(ts.shape[0], 1, self.args.z_dim).type_as(ts)
        if label is not None:
            z = torch.cat([z, label], dim=-1)
        w = self.base_sde(ts, normalize=True)
        hidden_state = self.flow(x0, ts, w, latent=z)
        return self._readout(hidden_state)

class FlowGAN(pl.LightningModule):
    def __init__(
            self, args, data
    ):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.generator = Generator(self.dim, self.n_classes, args)
        if args.d_model == 'CDE':
            self.discriminator = CDEDiscriminator(
                self.dim + self.n_classes,
                args.d_hidden_state_dim,
                args.d_hidden_dim,
                args.d_hidden_layers
            )
        else:
            self.discriminator = LSTMFlowDiscriminator(
                self.dim + self.n_classes,
                args.d_hidden_state_dim,
                args.d_hidden_dim,
                args.d_hidden_layers,
                args.time_net,
                args.time_hidden_dim,
            )
        self.optim = getattr(optim, args.optim)
        self.swa_step_start = self.args.swa_step_start
        self.averaged_generator = swa_utils.AveragedModel(self.generator)
        self.averaged_discriminator = swa_utils.AveragedModel(self.discriminator)
        self.automatic_optimization = False

        with torch.no_grad():
            for param in self.generator._initial.parameters():
                param *= args.init_mult1
            # make weights related to latent vectors larger gives better diffusion
            self.generator.flow.multiply_latent_weights(args.init_mult2)


        self.save_hyperparameters(args)
        self.step = 0

    def forward(self, ts, label=None, averaged=False):
        if not averaged:
            return self.generator(ts, label)
        self.train()
        with torch.no_grad():
            out = self.averaged_generator(ts, label)
        self.eval()
        return out

    def training_step(self, batch, batch_idx,):
        g_opt, d_opt = self.optimizers()

        d_opt.zero_grad()
        g_opt.zero_grad()
        loss, g_score, real_score = self._get_loss(batch, batch_idx, 'train')
        self.manual_backward(loss)
        for param in self.generator.parameters():
            param.grad *= -1
        # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip)
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.clip)
        if self.args.joint_training:
            d_opt.step()
            if self.step % self.args.update_g_every_n_iter == 0:
                g_opt.step()
        else:
            if g_score > real_score:
                d_opt.step()
            else:
                g_opt.step()
        self.step += 1
        self.log('train_loss', loss)
        assert not torch.isnan(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, *_ = self._get_loss(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss, *_ = self._get_loss(batch, batch_idx, 'test')
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.args.d_weight_clip > 0:
            with torch.no_grad():
                for module in self.discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features * self.args.d_weight_clip
                        module.weight.clamp_(-lim, lim)

        if self.current_epoch > self.swa_step_start:
            self.averaged_generator.update_parameters(self.generator)
            self.averaged_discriminator.update_parameters(self.discriminator)


    def configure_optimizers(self):
        generator_optimizer = self.optim(self.generator.parameters(), lr=self.args.g_lr,
                                             weight_decay=self.args.g_weight_decay)
        discriminator_optimizer = self.optim(self.discriminator.parameters(), lr=self.args.d_lr,
                                                 weight_decay=self.args.d_weight_decay)

        return [generator_optimizer, discriminator_optimizer]
    
    def _get_loss(self, batch, batch_idx, tag):
        x, ts, y_true = batch
        label = x[..., self.dim:] if self.n_classes > 0 else None
        generated_samples = self(ts, label=label)
        generated_score = self.discriminator(torch.cat([ts, generated_samples], dim=-1), label=label)
        real_score = self.discriminator(torch.cat([ts, y_true],dim=-1), label=label)
        loss = generated_score - real_score
        if tag == 'val':
            self.log(tag + '_generated_score', generated_score)
            self.log(tag + '_real_score', real_score)
            with torch.no_grad():
                mean_dist = torch.mean((torch.mean(generated_samples, dim=0) - torch.mean(y_true, dim=0)) ** 2)
            self.log('mean_dist', mean_dist)
        return loss, generated_score, real_score

    def train_dataloader(self):
        return self.dltrain

    def val_dataloader(self):
        return self.dlval

    def test_dataloader(self):
        return self.dltest