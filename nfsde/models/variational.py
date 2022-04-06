from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch.distributions import Normal
from torch import Tensor
from nfsde.models.lstm import ContinuousLSTMLayer, LSTMResNet, LSTMCoupling
from nfsde.models.ode_rnn_encoder import create_ode_rnn_encoder
from nfsde.models.discriminator import CDE
from nfsde.models.flow import ResNetFlow


class PosteriorZ(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_state_dim: int,
            hidden_dim: int,
            out_dim,
            model: str,
            flow_model: Optional[str] = None,
            activation: Optional[str] = None,
            # final_activation: Optional[str] = None,
            # solver: Optional[str] = None,
            # solver_step: Optional[int] = None,
            hidden_layers: Optional[int] = None,
            time_net: Optional[str] = None,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        self.model = model
        if model == 'flow':
            self.encoder = ContinuousLSTMLayer(
                dim,
                hidden_state_dim,
                hidden_dim,
                model,
                flow_model,
                activation,
                # final_activation,
                hidden_layers=hidden_layers,
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
            )
        elif model == 'ode-rnn':
            self.encoder = create_ode_rnn_encoder(
                dim,
                hidden_state_dim,
                out_dim,
                hidden_dim,
                hidden_layers
            )

        self.final_layer = nn.Linear(hidden_state_dim, 2*out_dim)
        self.min_log_std, self.max_log_std = (-20, 2)


    def forward(
            self,
            x: Tensor,  # Initial conditions, (..., seq_len, dim)
            t: Tensor,  # Times to solve at, (..., seq_len, dim)
    ):

        if self.model == 'flow':
            t_prev = t.clone()
            t_prev[:, 0] = 0
            t_prev[:, 1:] = t[:, :-1]
            dt = (t - t_prev + 1e-8)
            h = self.encoder(x, dt)

            mu_and_log_std = self.final_layer(h[:, -1])
            mu, log_std = torch.chunk(mu_and_log_std, 2, -1)
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)
        else:
            # Assume same t for all rows
            mask = torch.ones_like(x)
            mu, std = self.encoder(torch.cat([x, mask],dim=-1), t[0])
            mu, std = mu[0], std[0]

        distribution = Normal(mu, std)
        z = distribution.rsample()
        log_p = distribution.log_prob(z).sum(dim=-1, keepdims=True)

        return z, mu, std, log_p


class PosteriorLSTM(Module):
    def __init__(
            self,
            dim: int,
            obs_dim,
            hidden_state_dim: int,
            hidden_dim: int,
            model: str,
            z_dim: int = 0,
            flow_model: Optional[str] = None,
            activation: Optional[str] = None,
            # final_activation: Optional[str] = None,
            # solver: Optional[str] = None,
            # solver_step: Optional[int] = None,
            hidden_layers: Optional[int] = None,
            time_net: Optional[str] = None,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.hidden_state_dim = hidden_state_dim
        self.z_dim = z_dim

        if model == 'flow':
            FlowModel = LSTMResNet if flow_model == 'resnet' else LSTMCoupling
            self.lstm = FlowModel(dim+obs_dim, hidden_state_dim, hidden_dim, hidden_layers, time_net, time_hidden_dim)
        self.w_layer = nn.Linear(hidden_state_dim, dim*2)
        self.initial_layer = nn.Linear(obs_dim, hidden_state_dim*2)
        self.z_layer = None
        if z_dim > 0:
            self.z_layer = nn.Linear(hidden_state_dim, z_dim * 2)
        self.min_log_std, self.max_log_std = (-20, 2)


    def forward(
            self,
            x: Tensor,  # Initial conditions, (..., seq_len, dim)
            t: Tensor,  # Times to solve at, (..., seq_len, dim)
            x0: Tensor
    ):

        # Initialize hidden states
        # c, h = torch.zeros(x.shape[0], self.hidden_state_dim * 2).to(x).chunk(2, dim=-1)
        c, h = self.initial_layer(x0.squeeze(-2)).to(x).chunk(2, dim=-1)
        # hiddens = torch.zeros(*x.shape[:-1], self.hidden_state_dim).to(x)
        log_p = 0# torch.empty(*x.shape[:-1], 1).to(x)
        ws = torch.empty(x.shape[:-1] + (self.dim,)).to(x)
        # t0 = torch.zeros(x.shape[0], 1)
        w = torch.zeros(x.shape[0], self.dim).to(x)
        # h_pre, c, h = self.lstm(torch.cat([x[:,0], w[:, 0]],dim=-1), c, h, t0)

        for i in range(t.shape[1]):
            # Get pre- and post-jump states and cells
            h_pre, c, h = self.lstm(torch.cat([x[:,i], w],dim=-1), c, h, t[:, i])
            # hiddens[:, i] = h

            mu_and_log_std = self.w_layer(h)
            mu, log_std = torch.chunk(mu_and_log_std, 2, -1)

            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)
            distribution = Normal(mu, std)
            w = distribution.rsample()
            # log_p[:, i] = distribution.log_prob(w).sum(axis=-1, keepdims=True)
            log_p += distribution.log_prob(w).sum(dim=-1, keepdims=True)
            ws[:, i] = w

        # z = None
        # if self.z_layer is not None:
        #     mu_and_log_std_z = self.z_layer(h)
        #     mu_z, log_std_z = torch.chunk(mu_and_log_std_z, 2, -1)
        #
        #     log_std_z = torch.clamp(log_std_z, self.min_log_std, self.max_log_std)
        #     std_z = torch.exp(log_std_z)
        #     distribution_z = Normal(mu_z, std_z)
        #     z = distribution_z.rsample()
        #     log_p += distribution_z.log_prob(z).sum(axis=-1, keepdims=True)

        # return ws, z, log_p
        return ws, log_p


class PosteriorCDE(Module):
    def __init__(
            self,
            dim: int,
            cde_hidden_state_dim: int,
            flow_hidden_state_dim: int,
            flow_layers: int,
            cde_hidden_dims,
            flow_hidden_dims,
            w_dim,
            z_dim,
            flow_model: Optional[str] = None,
            activation: Optional[str] = None,
            # final_activation: Optional[str] = None,

            hidden_layers: Optional[int] = None,
            time_net: Optional[str] = None,
            time_hidden_dim: Optional[int] = None,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim

        if w_dim <= 0:
            encoder_out_dim = 0
        else:
            encoder_out_dim = flow_hidden_state_dim
        self.encoder = CDE(dim, cde_hidden_dims, cde_hidden_state_dim, encoder_out_dim)
        self.decoder = None
        self.layer_w = None
        self.layer_z = None
        if w_dim > 0:
            self.decoder = ResNetFlow(flow_hidden_state_dim, flow_layers, flow_hidden_dims, time_net=time_net, time_hidden_dim=time_hidden_dim)
            self.layer_w = nn.Linear(flow_hidden_state_dim, w_dim * 2)
        if z_dim > 0:
            self.layer_z = nn.Linear(cde_hidden_state_dim, z_dim * 2)
        self.min_log_std, self.max_log_std = (-20, 2)


    def forward(
            self,
            x: Tensor,  # Initial conditions, (..., seq_len, dim)
            t: Tensor,  # Times to solve at, (..., seq_len, dim)
            return_logp=False,
    ):

        # Initialize hidden states
        encoded = self.encoder(torch.concat([t, x], dim=-1))
        z, mu_z, log_std_z, log_pz = None, None, None, None
        if self.layer_z is not None:
            mu_and_log_std_z = self.layer_z(encoded)
            mu_z, log_std_z = torch.chunk(mu_and_log_std_z, 2, -1)
            log_std_z = torch.clamp(log_std_z, self.min_log_std, self.max_log_std)
            std_z = torch.exp(log_std_z)
            distribution_z = Normal(mu_z, std_z)
            z = distribution_z.rsample()
            if return_logp:
                log_pz = distribution_z.log_prob(z).sum(dim=-1)

        w, mu_w, log_std_w = None, None, None
        if self.layer_w is not None:
            decoded = self.decoder(encoded.unsqueeze(-2), t)
            mu_and_log_std_w = self.layer_w(decoded)
            mu_w, log_std_w = torch.chunk(mu_and_log_std_w, 2, -1)
            log_std_w = torch.clamp(log_std_w, self.min_log_std, self.max_log_std)
            std_w = torch.exp(log_std_w)
            distribution_w = Normal(mu_w, std_w)
            w = distribution_w.rsample()

        return w, z, mu_w, log_std_w, mu_z, log_std_z, encoded, log_pz
