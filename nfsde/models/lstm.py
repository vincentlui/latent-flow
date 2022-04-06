from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from nfsde.models import ResNetFlow, CouplingFlow
from nfsde.models.flow import ContinuousIResNet, LatentResnetFlow


class BaseContinuousLSTM(Module):
    """
    Base continuous LSTM class
    Other classes inherit and define `odeint` function

    Args:
        dim: Data dimension
        hidden_dim: Hidden state of LSTM
        odeint: Generic IVP solver, ODE or flow-based model
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        odeint: Module
    ):
        super().__init__()
        self.lstm = nn.LSTMCell(dim, hidden_dim)
        self.odeint = odeint

    def forward(
        self,
        x: Tensor, # Input data
        c: Tensor, # Previous `c` cell
        h: Tensor, # Previous `h` cell
        t: Tensor, # Input time
    ) -> Tuple[Tensor, Tensor, Tensor]: # Pre-jump state `h`, cell `c`, post-jump state `h`

        # Evolve the hidden state in continuous time
        h_pre = self.odeint(h.unsqueeze(1), t.unsqueeze(1)).squeeze(1)

        # Update the hidden state
        c, h = self.lstm(x, (c, h_pre))

        return h_pre, c, h


class LSTMResNet(BaseContinuousLSTM):
    """
    LSTM-based ResNet flow

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        n_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        hidden_state_dim: int,
        hidden_dim: int,
        n_layers: int,
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        activation='ReLU',
        **kwargs
    ):
        super().__init__(
            dim,
            hidden_state_dim,
            ResNetFlow(
                dim=hidden_state_dim,
                n_layers = n_layers,
                hidden_dims=[hidden_dim],
                time_net = time_net,
                time_hidden_dim = time_hidden_dim,
                activation=activation,
            )
        )


class LSTMCoupling(BaseContinuousLSTM):
    """
    LSTM-based coupling flow

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        n_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """

    def __init__(
            self,
            dim: int,
            hidden_state_dim: int,
            hidden_dim: int,
            n_layers: int,
            time_net: str,
            time_hidden_dim: Optional[int] = None,
            activation='ReLU',
            **kwargs
    ):
        super().__init__(
            dim,
            hidden_state_dim,
            CouplingFlow(
                dim=hidden_state_dim,
                n_layers=n_layers,
                hidden_dims=[hidden_dim],
                time_net=time_net,
                time_hidden_dim=time_hidden_dim,
                activation=activation,
            )
        )


class ContinuousLSTMLayer(Module):
    """
    Continuous LSTM layer with ODE or flow-based state evolution

    Args:
        dim: Data dimension
        hidden_dim: LSTM hidden dimension
        model: Which model to use (`ode` or `flow`)
        flow_model: Which flow model to use (`resnet` or `coupling`)
        activation: Name of the activation function from `torch.nn`
        final_activation: Name of the activation function from `torch.nn`
        solver: Which numerical solver to use
        solver_step: How many solvers steps to take, only applicable for fixed step solvers
        hidden_layers: How many flow layers
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        hidden_state_dim: int,
        hidden_dim: int,
        model: str,
        flow_model: Optional[str] = None,
        activation: Optional[str] = None,
        final_activation: Optional[str] = None,
        solver: Optional[str] = None,
        solver_step: Optional[int] = None,
        hidden_layers: Optional[int] = None,
        time_net: Optional[str] = None,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.hidden_state_dim = hidden_state_dim

        FlowModel = LSTMResNet if flow_model == 'resnet' else LSTMCoupling
        self.lstm = FlowModel(dim, hidden_state_dim, hidden_dim, hidden_layers, time_net, time_hidden_dim,
                              activation)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., seq_len, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., seq_len, dim)

        # Initialize hidden states
        c, h = torch.zeros(x.shape[0], self.hidden_state_dim * 2).to(x).chunk(2, dim=-1)
        hiddens = torch.zeros(*x.shape[:-1], self.hidden_state_dim).to(x)

        for i in range(t.shape[1]):
            # Get pre- and post-jump states and cells
            h_pre, c, h = self.lstm(x[:,i], c, h, t[:,i])
            hiddens[:,i] = h

        return hiddens
