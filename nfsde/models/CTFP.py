from typing import List, Optional, Tuple, Union
from torchtyping import TensorType

import stribor as st
from stribor.flows import Transform
from torch.nn import Module
from torch import nn
from torch import Tensor
import torch

class CTFP(Module):
        def __init__(
                self,
                dim: int,
                n_layers: int,
                # flow_model: str,
                hidden_dims: List[int],
                latent_dim = 0,
                activation=None,
                final_activation=None,
                **kwargs
        ):
            super().__init__()

            transforms = []
            # if flow_model == 'coupling':
            for i in range(n_layers):
                transforms.append(st.Coupling(
                    st.Affine(dim, latent_net=st.net.MLP(dim + latent_dim + 1, hidden_dims, 2 * dim,
                    activation=activation, final_activation=final_activation)),
                    mask='none' if dim == 1 else f'ordered_{i % 2}'))

            self.flow = Flow(transforms=transforms)

        def forward(
                self,
                w: Tensor,  # Brownian motion, (..., 1, dim)
                t: Tensor,  # Times to solve at, (..., seq_len, dim)
                latent=None,
                reverse=False,
                return_jaco=False
        ) -> Tensor:  # Solutions to IVP given x at t, (..., times, dim)

            if w.shape[-2] == 1:
                w = w.repeat_interleave(t.shape[-2], dim=-2)  # (..., 1, dim) -> (..., seq_len, 1)

            if latent is not None:
                if latent.shape[-2] == 1:
                    latent = latent.repeat_interleave(t.shape[-2], dim=-2)  # (..., 1, dim) -> (..., seq_len, 1)
                latent = torch.cat([t, latent], dim=-1)
            else:
                latent = t

            if return_jaco:
                if reverse:
                    y, ljd = self.flow.inverse_and_log_det_jacobian(w, latent=latent)
                else:
                    y, ljd = self.flow.forward_and_log_det_jacobian(w, latent=latent)
                return y, ljd

            if reverse:
                y = self.flow.inverse(w, latent=latent)
            else:
                y = self.flow(w, latent=latent)

            # if reverse:
            #     y, ljd = self.flow.inverse(w, latent=latent)
            # else:
            #     y, ljd = self.flow(w, latent=latent)
            #
            # if return_jaco:
            #     return y, ljd

            return y

        def inverse(
                self,
                w: Tensor,  # Brownian motion, (..., 1, dim)
                t: Tensor,  # Times to solve at, (..., seq_len, dim)
                latent=None,
                return_jaco=False
        ):
            return self.forward(w, t, latent=latent, reverse=True, return_jaco=return_jaco)


class CTFP2(Module):
    def __init__(
            self,
            dim: int,
            n_layers: int,
            hidden_dims: List[int],
            anode_dim=1,
            activation=None,
            final_activation=None,
            **kwargs
    ):
        super().__init__()

        # transforms = []
        # for i in range(n_layers):
        #     transforms.append(st.flows.ContinuousTransform(
        #         st.Affine(dim, st.net.MLP(dim + latent_dim + 1, hidden_dims, 2 * dim)),
        #         mask='none' if dim == 1 else f'ordered_{i % 2}'))
        #
        # self.flow = st.Flow(transforms=transforms)
        self.dim = dim
        self.anode_dim = anode_dim
        transforms = []
        for i in range(n_layers):
            transforms.append(AnodeContinuosTransform(dim=dim, net=DiffeqAnode(dim, anode_dim, hidden_dims,
                activation=activation, final_activation=final_activation)))

        self.flow = Flow(transforms=transforms)

    def forward(
            self,
            w: Tensor,  # Brownian motion, (..., 1, dim)
            t: Tensor,  # Times to solve at, (..., seq_len, dim)
            latent=None,
            reverse=False,
            return_jaco=False
    ) -> Tensor:  # Solutions to IVP given x at t, (..., times, dim)

        if w.shape[-2] == 1:
            w = w.repeat_interleave(t.shape[-2], dim=-2)  # (..., 1, dim) -> (..., seq_len, 1)

        if latent is not None:
            if latent.shape[-2] == 1:
                latent = latent.repeat_interleave(t.shape[-2], dim=-2)  # (..., 1, dim) -> (..., seq_len, 1)
            w = torch.cat([w, latent], dim=-1)
        latent=None

        w = torch.cat([w, t], dim=-1)

        if return_jaco:
            if reverse:
                y, ljd = self.flow.inverse_and_log_det_jacobian(w)
            else:
                y, ljd = self.flow.forward_and_log_det_jacobian(w)
            return y[..., :self.dim], ljd[..., :self.dim].sum(-1, keepdim=True)

        if reverse:
            y = self.flow.inverse(w, latent=latent)
        else:
            y = self.flow(w, latent=latent)

        return y[..., :self.dim]

    def inverse(
            self,
            w: Tensor,  # Brownian motion, (..., 1, dim)
            t: Tensor,  # Times to solve at, (..., seq_len, dim)
            latent=None,
            return_jaco=False
    ):
        return self.forward(w, t, latent=latent, reverse=True, return_jaco=return_jaco)

class DiffeqAnode(Module):
    def __init__(
            self,
            dim: int,
            anode_dim: int,
            hidden_dims: List[int],
            activation: str = None,
            final_activation: str = None,
    ):
        super().__init__()
        self.dim = dim
        self.anode_dim = anode_dim
        self.f = st.net.DiffeqMLP(dim + 1 + anode_dim, hidden_dims, dim, activation=activation, final_activation=final_activation)
        self.g = st.net.DiffeqMLP(1 + anode_dim, hidden_dims, anode_dim, activation=activation, final_activation=final_activation)

    def forward(self, t, state, latent: TensorType[..., 'latent'] = None,
        **kwargs,):
        # x = state[..., :self.dim]
        z = state[..., self.dim:]
        dz = self.g(t, z)
        dx = self.f(t, state)

        return torch.cat([dx, dz], dim=-1)


class AnodeContinuosTransform(st.ContinuousTransform):
    def forward_and_log_det_jacobian(
            self,
            x: TensorType[..., 'dim'],
            latent: Optional[TensorType[..., 'latent']] = None,
            mask: Optional[Union[TensorType[..., 1], TensorType[..., 'dim']]] = None,
            *,
            reverse: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 1]]:
        # Set inputs
        log_trace_jacobian = torch.zeros_like(x)

        # Set integration times
        integration_times = torch.tensor([0.0, self.T]).to(x)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics
        self.odefunc.before_odeint()

        initial = (x, log_trace_jacobian)
        if latent is not None:
            initial += (latent,)
        if mask is not None:
            initial += (mask,)

        # Solve ODE
        state_t = self.integrate(
            self.odefunc,
            initial,
            integration_times,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver if self.training else self.test_solver,
            options=self.solver_options if self.training else self.test_solver_options,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        # Collect outputs with correct shape
        x, log_trace_jacobian = state_t[:2]
        return x, -log_trace_jacobian




class Flow(Transform):
    """
    Normalizing flow for density estimation and efficient sampling.

    Args:
        transforms (Transform): List of invertible transformations
    """
    def __init__(
        self,
        transforms: List[Transform],
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        for f in self.transforms:
            x = f(x, **kwargs)
        return x

    def inverse(self, y: TensorType[..., 'dim'], **kwargs) -> TensorType[..., 'dim']:
        for f in reversed(self.transforms):
            y = f.inverse(y, **kwargs)
        return y

    def forward_and_log_det_jacobian(
        self, x: TensorType[..., 'dim'], **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 1]]:
        log_det_jac = 0
        for f in self.transforms:
            x, ldj = f.forward_and_log_det_jacobian(x, **kwargs)
            log_det_jac += ldj
        return x, log_det_jac

    def inverse_and_log_det_jacobian(
        self, y: TensorType[..., 'dim'], **kwargs,
    ) -> Tuple[TensorType[..., 'dim'], TensorType[..., 1]]:
        log_det_jac = 0
        for f in reversed(self.transforms):
            y, ldj = f.inverse_and_log_det_jacobian(y, **kwargs)
            log_det_jac += ldj
        return y, log_det_jac

    def log_det_jacobian(
        self, x: TensorType[..., 'dim'], y: TensorType[..., 'dim'], **kwargs,
    ) -> TensorType[..., 1]:
        _, log_det_jacobian = self.forward_and_log_det_jacobian(x, **kwargs)
        return log_det_jacobian


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]