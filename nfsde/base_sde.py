import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

def sample_brownian(t,d,o,l):
    return
class Brownian:
    def __init__(self, dim, delta=1., device='cpu'):
        self.dim = dim
        self.delta = torch.tensor(delta).to(device)

    def sample(self, ts, out=None, log_prob=False):
        """
        Adopted from https: // scipy - cookbook.readthedocs.io / items / BrownianMotion.html
        """

        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,))
        delta_ts[:, 0] = ts[:, 0]
        delta_ts[:, 1:] = ts[:, 1:] - ts[:, :-1]
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=self.delta*torch.sqrt(delta_ts))
        r = distr.rsample()
        r[invalid_index] = 0.

        # If `out` was not given, create an output array.
        # if out is None:
        #     out = torch.empty(r.shape).to(ts)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        out = torch.cumsum(r, dim=-2).to(ts)

        if log_prob:
            log_prob = distr.log_prob(r).sum(dim=-1, keepdim=True)
            return out, log_prob

        return out

    def log_prob(self, x, t):
        t_0 = torch.zeros((len(t), 1, 1)).to(t)
        t_i = torch.cat([t_0, t[:, :-1]], dim=-2)
        dt = (t - t_i + 1e-8)
        mean_martingale = x.clone()
        mean_martingale[:, 1:] = x.clone()[:, :-1]
        mean_martingale[:, 0:1] = 0.
        distr_p = Normal(mean_martingale, dt.sqrt())
        log_px = torch.sum((distr_p.log_prob(x)).view(len(t), -1), dim=-1)

        return log_px

class OU(nn.Module):
    def __init__(
            self,
            dim,
            device='cpu'
    ):
        super().__init__()
        self.dim = dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.theta = torch.nn.Parameter(torch.zeros(dim, requires_grad=True, device=self.device))
        self.mu = torch.nn.Parameter(torch.zeros(dim, requires_grad=True, device=self.device))
        self.sigma = torch.nn.Parameter(torch.ones(dim, requires_grad=True, device=self.device))

    def forward(self, ts, log_prob=False):
        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,)).double().to(self.device)
        tsd = ts.double()
        theta = torch.sigmoid(self.theta)
        sigma = torch.nn.functional.softplus(self.sigma)
        delta_ts[:, 0] = torch.exp(2 * theta * tsd[:, 0]) - 1
        delta_ts[:, 1:] = torch.exp(2 * theta * tsd[:, 1:]) - torch.exp(2 * theta * tsd[:, :-1])
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=sigma / torch.sqrt(2 * theta) * torch.sqrt(delta_ts))
        r = distr.rsample()
        r[invalid_index] = 0.

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        out = torch.cumsum(r, dim=-2)
        out = torch.exp(-theta * ts) * out + self.mu * (1 - torch.exp(-theta * ts))
        out = out.to(ts)

        if log_prob:
            log_prob = distr.log_prob(r).sum(dim=-1, keepdim=True).to(ts)
            return out, log_prob

        return out

    def sample(self, ts, out=None, log_prob=False):
        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,)).double()
        tsd = ts.double()
        delta_ts[:, 0] = torch.exp(2 * self.theta * tsd[:, 0]) - 1
        delta_ts[:, 1:] = torch.exp(2 * self.theta * tsd[:, 1:]) - torch.exp(2 * self.theta * tsd[:, :-1])
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=self.sigma / torch.sqrt(2 * self.theta) * torch.sqrt(delta_ts))
        r = distr.rsample()
        r[invalid_index] = 0.

        # If `out` was not given, create an output array.
        if out is None:
            out = torch.empty(r.shape)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        torch.cumsum(r, dim=-2, out=out)
        out = torch.exp(-self.theta * ts) * out + self.mu * (1 - torch.exp(-self.theta * ts))
        out = out.to(ts)

        if log_prob:
            log_prob = distr.log_prob(r).sum(dim=-1, keepdim=True).to(ts)
            return out, log_prob

        return out

class Brownian2(nn.Module):
    def __init__(
            self,
            dim,
            sigma=1.,
            train_param=False
    ):
        super().__init__()
        self.dim = dim
        if train_param:
            self.sigma = torch.nn.Parameter(torch.ones(dim, requires_grad=True))
            self.sigma_transform = torch.nn.functional.softplus
        else:
            self.register_buffer('sigma', torch.as_tensor(sigma))
            self.sigma_transform = torch.nn.Identity()

    def forward(self, ts, log_prob=False, normalize=True):
        """
        Adopted from https: // scipy - cookbook.readthedocs.io / items / BrownianMotion.html
        """
        sigma = self.sigma_transform(self.sigma)
        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,)).type_as(ts)
        delta_ts[:, 0] = ts[:, 0]
        delta_ts[:, 1:] = ts[:, 1:] - ts[:, :-1]
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=sigma*torch.sqrt(delta_ts))
        r = distr.rsample()
        r[invalid_index] = 0.

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        out = torch.cumsum(r, dim=-2).type_as(ts)

        if normalize:
            out /= sigma * torch.sqrt(ts) + 1e-8

        if log_prob:
            log_prob = distr.log_prob(r).sum(dim=-1, keepdim=True)
            return out, log_prob

        return out

    def log_prob(self, x, t):
        t_prev = t.clone()
        t_prev[:, 0] = 0
        t_prev[:, 1:] = t[:, :-1]
        dt = (t - t_prev + 1e-8)
        mean = x.clone()
        mean[:, 1:] = x.clone()[:, :-1]
        mean[:, 0] = 0.
        distr_p = Normal(mean, dt.sqrt())
        log_px = torch.sum((distr_p.log_prob(x)).view(t.shape[0], t.shape[1], -1), dim=-1, keepdim=True)

        return log_px

    def get_path_mean_and_std(self, x, t):
        t_prev = t.clone()
        t_prev[:, 0] = 0
        t_prev[:, 1:] = t[:, :-1]
        dt = t - t_prev

        x_prev = x.clone()
        x_prev[:, 1:] = x_prev[:, :-1]
        x_prev[:, 0] = 0.
        mean = x_prev

        std = dt
        return mean, std

class OU2(nn.Module):
    def __init__(
            self,
            dim,
            theta=0.1,
            mu=0.,
            sigma=1.,
            train_param=False
    ):
        super().__init__()
        self.dim = dim
        self.train_param = train_param
        if train_param:
            self.theta = torch.nn.Parameter(torch.randn(dim, requires_grad=True))
            self.mu = torch.nn.Parameter(torch.randn(dim, requires_grad=True))
            self.sigma = torch.nn.Parameter(torch.ones(dim, requires_grad=True))
            self.theta_transform = torch.sigmoid
            self.sigma_transform = torch.nn.functional.softplus
        else:
            self.register_buffer('mu', torch.as_tensor(mu))
            self.register_buffer('theta', torch.as_tensor(theta))
            self.register_buffer('sigma', torch.as_tensor(sigma))
            self.theta_transform = torch.nn.Identity()
            self.sigma_transform = torch.nn.Identity()

    def forward(self, ts, log_prob=False, normalize=True):
        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,)).type_as(ts).double()
        tsd = ts.double()
        theta = self.theta_transform(self.theta)
        sigma = self.sigma_transform(self.sigma)

        delta_ts[:, 0] = torch.exp(2 * theta * tsd[:, 0]) - 1
        delta_ts[:, 1:] = torch.exp(2 * theta * tsd[:, 1:]) - torch.exp(2 * theta * tsd[:, :-1])
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=torch.sqrt(delta_ts))
        r = distr.rsample()
        r[invalid_index] = 0.
        r *= sigma / torch.sqrt(2 * theta)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        out = torch.cumsum(r, dim=-2).type_as(ts)
        out = torch.exp(-theta * ts) * out + self.mu * (1 - torch.exp(-theta * ts))

        if normalize:
            out /= sigma * torch.sqrt((1 - torch.exp(-2*theta*tsd)) / (2 * theta)) + 1e-8

        if log_prob:
            raise NotImplementedError

        return out

    def get_path_mean_and_std(self, x, t):
        t_prev = t.clone()
        t_prev[:, 0] = 0
        t_prev[:, 1:] = t[:, :-1]
        dt = t - t_prev
        x_prev = x.clone()
        x_prev[:, 1:] = x_prev[:, :-1]
        x_prev[:, 0] = 0
        mean = x_prev * torch.exp(-self.theta * dt) + self.mu * (1 - torch.exp(-self.theta * dt))
        std = self.sigma / torch.sqrt(2 * self.theta) * torch.sqrt(1 - torch.exp(-2 * self.theta * dt))
        return mean, std


class Brownian_OU(nn.Module):
    def __init__(
            self,
            dim,
            theta=0.1,
            mu=0.,
            sigma=1.,
            train_param=False
    ):
        super().__init__()
        self.dim = dim
        if train_param:
            raise NotImplementedError
        else:
            self.register_buffer('sigma', torch.as_tensor(sigma))
            self.register_buffer('theta', torch.as_tensor(theta))
            self.register_buffer('mu', torch.as_tensor(mu))
            self.sigma_transform = torch.nn.Identity()

    def forward(self, ts, log_prob=False, normalize=True):
        delta_ts = torch.zeros(ts.shape[:-1] + (self.dim,)).type_as(ts).double()
        tsd = ts.double()
        slice_index = int(self.dim/2)
        delta_ts[:, 0, :slice_index] = torch.sqrt(ts[:, 0])
        delta_ts[:, 1:, :slice_index] = torch.sqrt(ts[:, 1:] - ts[:, :-1])
        delta_ts[:, 0, slice_index:] = torch.sqrt(torch.exp(2 * self.theta * tsd[:, 0]) - 1)
        delta_ts[:, 1:, slice_index:] = torch.sqrt(torch.exp(2 * self.theta * tsd[:, 1:]) - torch.exp(2 * self.theta * tsd[:, :-1]))
        invalid_index = delta_ts <= 0
        delta_ts[invalid_index] += 1e-4

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        distr = Normal(loc=torch.zeros_like(delta_ts), scale=delta_ts)
        r = distr.rsample()
        r[invalid_index] = 0.
        r[..., :slice_index] *= torch.sqrt(self.sigma)
        r[..., slice_index:] *= self.sigma / torch.sqrt(2 * self.theta)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        out = torch.cumsum(r, dim=-2).type_as(ts)
        out[..., slice_index:] = torch.exp(-self.theta * ts) * out[..., slice_index:] + self.mu * (1 - torch.exp(-self.theta * ts))

        if normalize:
            out[..., :slice_index] /= self.sigma * torch.sqrt(ts) + 1e-8
            out[..., slice_index:] /= self.sigma * torch.sqrt((1 - torch.exp(-2 * self.theta * ts)) / (2 * self.theta)) + 1e-8

        if log_prob:
            raise NotImplementedError

        return out

    def log_prob(self, x, t):
        t_prev = t.clone()
        t_prev[:, 0] = 0
        t_prev[:, 1:] = t[:, :-1]
        dt = (t - t_prev + 1e-8)
        mean_martingale = x.clone()
        mean_martingale[:, 1:] = x.clone()[:, :-1]
        mean_martingale[:, 0] = 0.
        distr_p = Normal(mean_martingale, dt.sqrt())
        log_px = torch.sum((distr_p.log_prob(x)).view(t.shape[0], t.shape[1], -1), dim=-1, keepdim=True)

        return log_px

    def get_path_mean_and_std(self, x, t):
        t_prev = t.clone()
        t_prev[:, 0] = 0
        t_prev[:, 1:] = t[:, :-1]
        dt = t - t_prev

        x_prev = x.clone()
        x_prev[:, 1:] = x_prev[:, :-1]
        x_prev[:, 0] = 0.
        mean = x_prev

        std = dt
        return mean, std