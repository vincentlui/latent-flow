from typing import List, Optional
from torchtyping import TensorType

import torch
import torch.nn as nn
import stribor as st
from stribor import Transform
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import spectral_norm


class CouplingFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + 1, hidden_dims, 2 * dim),
                time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = st.NeuralFlow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0)

        return self.flow(x, t=t)


class LatentCouplingFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        w_dim: int,
        z_dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        activation='ReLU',
        final_activation=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.w_dim = w_dim if w_dim > 0 else 0
        self.z_dim = z_dim if z_dim > 0 else 0
        transforms = []
        for i in range(n_layers):
            transforms.append(st.ContinuousAffineCoupling(
                latent_net=st.net.MLP(dim + w_dim + z_dim + 1, hidden_dims, 2 * dim,
                                      activation=activation, final_activation=final_activation),
                time_net=getattr(st.net, time_net)(2 * dim, hidden_dim=time_hidden_dim),
                mask='none' if dim == 1 else f'ordered_{i % 2}'))

        self.flow = st.NeuralFlow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        w: Tensor,
        latent: Tensor = None, # Latent vector
        t0: Optional[Tensor] = None,
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        if latent is not None:
            if latent.dim() == 2:
                latent = latent.unsqueeze(-2)
            if latent.shape[-2] == 1:
                latent = latent.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)
            w = torch.concat([w, latent], dim=-1)

        # If t0 not 0, solve inverse first
        if t0 is not None:
            x = self.flow.inverse(x, t=t0, latent=w)

        return self.flow(x, t=t, latent=w)

    def multiply_latent_weights(self, mult):
        with torch.no_grad():
            for transform_net in self.flow.transforms:
                transform_net.latent_net.net[0].weight[:, self.dim:-1-self.z_dim] *= mult


class ResNetFlow(Module):
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

        transforms = []
        for _ in range(n_layers):
            transforms.append(ContinuousIResNet(
                dim,
                hidden_dims,
                activation='ReLU',
                final_activation=None,
                time_net=getattr(st.net, time_net)(dim, hidden_dim=time_hidden_dim),
            ))

        self.flow = st.NeuralFlow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2)

        return self.flow(x, t=t)

class LatentResnetFlow(Module):
    """
    Affine coupling flow

    Args:
        dim: Data dimension
        n_layers: Number of flow layers
        hidden_dims: Hidden dimensions of the flow neural network
        time_net: Time embedding module
        time_hidden_dim: Time embedding hidden dimension
    """
    def __init__(
        self,
        dim: int,
        w_dim: int,
        z_dim: int,
        n_layers: int,
        hidden_dims: List[int],
        time_net: Module,
        time_hidden_dim: Optional[int] = None,
        activation = 'ReLU',
        final_activation = None,
        invertible = True,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.w_dim = w_dim if w_dim > 0 else 0
        self.z_dim = z_dim if z_dim > 0 else 0
        transforms = []
        for _ in range(n_layers):
            transforms.append(ContinuousIResNet(
                dim,
                hidden_dims,
                w_dim + z_dim,
                activation=activation,
                final_activation=final_activation,
                time_net=getattr(st.net, time_net)(dim, hidden_dim=time_hidden_dim),
                invertible=invertible
            ))

        self.flow = NeuralFlow(transforms=transforms) #st.NeuralFlow(transforms=transforms)

    def forward(
        self,
        x: Tensor, # Initial conditions, (..., 1, dim)
        t: Tensor, # Times to solve at, (..., seq_len, dim)
        w: Tensor,
        latent: Tensor = None, # Latent vector
        reverse = False
    ) -> Tensor: # Solutions to IVP given x at t, (..., times, dim)

        if x.shape[-2] == 1:
            x = x.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)

        if latent is not None:
            if latent.dim() == 2:
                latent = latent.unsqueeze(-2)
            if latent.shape[-2] == 1:
                latent = latent.repeat_interleave(t.shape[-2], dim=-2) # (..., 1, dim) -> (..., seq_len, 1)
            w = torch.concat([w, latent], dim=-1)

        if reverse:
            return self.flow.inverse(x, t=t, latent=w)

        return self.flow(x, t=t, latent=w)

    def inverse(self, y, t, w, latent=None):
        return self.forward(y, t, w, latent=latent, reverse=True)

    def multiply_latent_weights(self, mult):
        with torch.no_grad():
            for transform_net in self.flow.transforms:
                transform_net.net.net[0].weight[:, self.dim+1: self.dim+1+self.w_dim] *= mult

#
class ContinuousIResNet(Transform):
    """
    Continuous time invertible ResNet transformation.

    Args:
        dim: Input dimension
        hidden_dims: Hidden dimensions
        activation: Activation between hidden layers
        final_activation: Final activation
        time_net: Time embedding network
        time_hidden_dim: Time embedding dimension
        n_power_iterations: How many iterations to perform in `spectral_norm`
    """
    def __init__(
        self,
        dim: int,
        hidden_dims: List[int],
        latent_dim = 0,
        *,
        activation: str ='ReLU',
        final_activation: str = None,
        time_net: torch.nn.Module = None,
        invertible: bool = True,
        n_power_iterations: int = 5,
        **kwargs,
    ):
        super().__init__()
        if invertible:
            wrapper = lambda layer: spectral_norm(layer, n_power_iterations=n_power_iterations)
            self.net = st.net.MLP(dim + latent_dim + 1, hidden_dims, dim, activation, final_activation, nn_linear_wrapper_func=wrapper)
        else:
            self.net = st.net.MLP(dim + latent_dim + 1, hidden_dims, dim, activation, final_activation)
        self.time_net = time_net

    def forward(self, x: TensorType[..., 'dim'], t: TensorType[..., 1],
                latent: Optional[TensorType[..., 'latent']] = None) -> TensorType[..., 'dim']:
        z = t
        if latent is not None:
            z = torch.cat([z, latent], dim=-1)
        return x + self.time_net(t) * self.net(torch.cat([x, z], dim=-1))

    def inverse(self, y: TensorType[..., 'dim'], t: TensorType[..., 1], latent=None, iterations=100, **kwargs) -> TensorType[..., 'dim']:
        # Fixed-point iteration inverse
        x = y
        z = t
        if latent is not None:
            z = torch.cat([z, latent], dim=-1)
        for _ in range(iterations):
            residual = self.time_net(t) * self.net(torch.cat([x, z], dim=-1))
            x = y - residual
        return x

    def log_det_jacobian(self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 1]:
        return NotImplementedError


class NeuralFlow(nn.Module):
    """
    Neural flow model.
    https://arxiv.org/abs/2110.13040

    Example:
    >>> import stribor as st
    >>>

    Args:
        transforms (Transform): List of invertible transformations
            that satisfy initial condition F(x, t=0)=x.
    """
    def __init__(self, transforms: List[Transform]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(
        self,
        x: TensorType[..., 'dim'],
        t: TensorType[..., 1],
        t0: Optional[TensorType[..., 1]] = None,
        **kwargs,
    ) -> TensorType[..., 'dim']:
        if t0 is not None:
            for transform in reversed(self.transforms):
                x = transform.inverse(x, t=t0, **kwargs)
        for transform in self.transforms:
            x = transform(x, t=t, **kwargs)
        return x

    def inverse(
        self,
        x: TensorType[..., 'dim'],
        t: TensorType[..., 1],
        **kwargs,
    ):
        for transform in reversed(self.transforms):
            x = transform.inverse(x, t=t, **kwargs)
        return x