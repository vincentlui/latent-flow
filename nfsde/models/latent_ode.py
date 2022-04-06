from stribor.net import MLP, DiffeqMLP
from torchdiffeq import odeint_adjoint, odeint
import torch
from torch import nn
from typing import Dict, Optional, Tuple, Union, List
from torchtyping import TensorType


class DiffeqMLPLatent(DiffeqMLP):
    """
    Differential equation defined with MLP.

    Example:
    >>> batch, dim = 32, 3
    >>> net = stribor.net.DiffeqMLP(dim + 1, [64, 64], dim)
    >>> x = torch.randn(batch, dim)
    >>> t = torch.rand(batch, 1)
    >>> net(t, x).shape
    torch.Size([32, 3])

    Args:
        Same as in `st.net.MLP`
    """

    def forward(
            self,
            t: TensorType[1],
            x: TensorType[..., 'dim'],
            latent: TensorType[..., 'latent'] = None,
            **kwargs,
    ) -> TensorType[..., 'out']:
        t = torch.ones_like(x[..., :1]) * t
        input = torch.cat([t, x], -1)
        if latent is not None:
            input = torch.cat([input, latent], -1)
            dx = self.net(input, **kwargs)
            dlatent = torch.zeros_like(latent)
            return torch.cat([dx, dlatent], dim=-1)
        dx = self.net(input, **kwargs)
        return dx


class ODEfunc(nn.Module):
    def __init__(
            self,
            diffeq: nn.Module,
            has_latent: Optional[bool] = False,
            **kwargs,
    ):
        super().__init__()

        self.diffeq = diffeq
        self.has_latent = has_latent

        self.register_buffer('_num_evals', torch.tensor(0.))

    def before_odeint(self):
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(
            self,
            t: TensorType[()],
            states: Union[Tuple, Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]],
    ) -> Union[Tuple, Tuple[TensorType[..., 'dim'], TensorType[..., 'dim']]]:
        self._num_evals += 1

        y = states[0]
        # t = torch.Tensor([t]).to(y)

        latent = None
        if len(states) == 2:
            if self.has_latent:
                latent = states[1]

        dy = self.diffeq(t, y, latent=latent)
        return dy

class LatentODE(nn.Module):
    def __init__(
        self,
        dim: int,
        latent_dim: int,
        hidden_dims: List,
        activation: str = None,
        final_activation: str = None,
        use_adjoint: bool = True,
        solver: str = 'dopri5',
        solver_options: Optional[Dict] = {},
        test_solver: str = None,
        test_solver_options: Optional[Dict] = None,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim

        diffeq = DiffeqMLPLatent(dim + 1 + latent_dim, hidden_dims, dim, activation, final_activation)

        self.odefunc = ODEfunc(diffeq, True)

        self.integrate = odeint_adjoint if use_adjoint else odeint

        self.solver = solver
        self.solver_options = solver_options
        self.test_solver = test_solver or solver
        self.test_solver_options = solver_options if test_solver_options is None else test_solver_options

        self.atol = atol
        self.rtol = rtol

    def forward(self, x, ts, latent=None):
        self.odefunc.before_odeint()
        initial = (x,)
        if latent is not None:
            initial += (latent,)
        else:
            raise NotImplementedError

        # Solve ODE
        state_t = odeint_adjoint(
            self.odefunc,
            initial,
            ts,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
            options=self.solver_options,
        )

        return state_t[0].transpose(0, 1)