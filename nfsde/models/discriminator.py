import torch
import torchcde
from nfsde.models.lstm import ContinuousLSTMLayer
import stribor as st
from torch import nn

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)

class CDEDiffFunc(torch.nn.Module):
    def __init__(self, dim, hidden_dims, hidden_state_dim, activation="ReLU", final_activation=None):
        super().__init__()
        self._data_size = dim
        self._hidden_size = hidden_state_dim
        self._module = st.net.MLP(1 + hidden_state_dim, hidden_dims, hidden_state_dim * (1 + dim), activation=activation,
                                final_activation=final_activation)

    def forward(self, t, h):
        # t has shape (batch_size, )
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class CDEDiscriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs, label=None):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        if label is not None:
            if label.shape[-2] == 1:
                label = label.repeat_interleave(ys_coeffs.shape[-2], dim=-2)
            Y = torchcde.LinearInterpolation(
                torch.concat([ys_coeffs, label], dim=-1))
        else:
            Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()

class LSTMFlowDiscriminator(torch.nn.Module):
    def __init__(
            self,
            dim,
            hidden_state_dim,
            hidden_dim,
            hidden_layers,
            time_net,
            time_hidden_dim
    ):
        super().__init__()

        self.lstm = ContinuousLSTMLayer(
            dim+1,
            hidden_state_dim,
            hidden_dim,
            'flow',
            'resnet',
            LipSwish,
            # final_activation,
            hidden_layers=hidden_layers,
            time_net=time_net,
            time_hidden_dim=time_hidden_dim,
        )
        self._readout = torch.nn.Linear(hidden_state_dim, 1)

    def forward(self, y):
        t = y[..., :1]
        delta_t = t.clone()
        delta_t[:, 1:] = t[:, 1:] - t[:, :-1]
        hs = self.lstm(y, t)
        score = self._readout(hs[:, -1])
        return score.mean()


class CDE(torch.nn.Module):
    def __init__(self, dim, hidden_dims, hidden_state_dim, out_dim, activation="Softplus", final_activation="Tanh"):
        super().__init__()
        self._initial = torch.nn.Linear(1 + dim, hidden_state_dim)
        self._func = CDEDiffFunc(dim, hidden_dims, hidden_state_dim, activation=activation, final_activation=final_activation)
        self._readout = None
        if out_dim > 0:
            self._readout = torch.nn.Linear(hidden_state_dim, out_dim)

    def forward(self, ys_coeffs, label=None, intermediate=False):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        if label is not None:
            if label.shape[-2] == 1:
                label = label.repeat_interleave(ys_coeffs.shape[-2], dim=-2)
            Y = torchcde.LinearInterpolation(
                torch.concat([ys_coeffs, label], dim=-1))
        else:
            Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        t = Y.interval
        if intermediate:
            t = Y.grid_points
        hs = torchcde.cdeint(Y, self._func, h0, t, method='reversible_heun', backend='torchsde', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        if not intermediate:
            out = hs[:, -1]
        else:
            out = hs
        if self._readout is not None:
            out = self._readout(out)
        return out

class CDEClassificationNet(torch.nn.Module):
    def __init__(self, dim, hidden_dims, hidden_state_dim, activation="ReLU", final_activation=None):
        super().__init__()
        self.cde = CDE(dim, hidden_dims, hidden_state_dim, 1, activation=activation, final_activation=final_activation)

    def forward(self, ys_coeffs, label=None):
        out = self.cde(ys_coeffs, label)
        return torch.sigmoid(out)