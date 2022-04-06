from typing import List, Optional

import torch
import torch.nn as nn
import stribor as st
from torch import Tensor
from torch.nn import Module

from nfsde.models.flow import ResNetFlow, CouplingFlow


class StochasticFlow(nn.Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        invertible: Optional[bool] = True,
        **kwargs
    ):
        super().__init__()
        self.f = ResNetFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, invertible, **kwargs)

        # self.g = ResNetFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, invertible, **kwargs)
        self.g = st.net.MLP(dim + 1, hidden_dims, dim)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        w: Tensor,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        return self.f(x, t) + self.g(w, t)


class StochasticFlow2(nn.Module):
    """
    ResNet flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the residual neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
        invertible: Whether to make ResNet invertible (necessary for proper flow)
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: str,
        time_hidden_dim: Optional[int] = None,
        invertible: Optional[bool] = True,
        **kwargs
    ):
        super().__init__()
        # self.f = ResNetFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, invertible, **kwargs)
        self.f = CouplingFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, **kwargs)
        # self.g = ResNetFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, **kwargs)
        # self.g = CouplingFlow(dim, n_layers, hidden_dims, time_net, time_hidden_dim, **kwargs)
        self.g = st.net.MLP(dim+1, hidden_dims, dim)
    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        w: Tensor,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)
        g_input = torch.cat([w, t], dim=-1)

        return self.f(x, t) + self.g(g_input)