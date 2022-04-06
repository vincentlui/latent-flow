import torchsde
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from stribor.net import MLP
from nfsde.models.discriminator import CDE


def _stable_sign(b):
    b = b.sign()
    b[b == 0] = 1
    return b


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(
        b.abs().detach() > epsilon,
        b,
        torch.full_like(b, fill_value=epsilon) * _stable_sign(b),
    )

    return a / b


def network_factory(network_dims, non_linearity="softplus"):
    module_list = []
    network_dims = list(network_dims)
    if non_linearity.lower() == "softplus":
        non_linearity = nn.Softplus
    elif non_linearity.lower() == "sigmoid":
        non_linearity = nn.Sigmoid
    elif non_linearity.lower() == "relu":
        non_linearity = nn.ReLu
    for i in range(len(network_dims) - 2):
        module_list.append(nn.Linear(network_dims[i], network_dims[i + 1]))
        module_list.append(non_linearity())
    module_list.append(nn.Linear(network_dims[-2], network_dims[-1]))
    return nn.Sequential(*module_list)


def time_embedding(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    return torch.cat([t, sin_t, cos_t], 1)
    # return t


def sample_normal(mean, logvar, stdv=None):
    if stdv is None:
        stdv = torch.exp(0.5 * logvar)
    return torch.randn_like(mean) * stdv + mean


class ConstantVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(ConstantVariance, self).__init__()
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        self.variance = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, t, y):
        batch_size = y.shape[0]
        return self.activation(self.variance).repeat(batch_size, 1)


class DiagonalVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(DiagonalVariance, self).__init__()
        self.context_tensor = None
        self.variance_networks = nn.ModuleList(
            [
                network_factory([1 + time_embedding_dim] + model_dims)
                for i in range(input_dim)
            ]
        )
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, t, y):
        ## t: time index of size batch_size x time_embedding_dim,
        ## y: size batch_size x latent or data_size
        y_split = torch.split(y, split_size_or_sections=1, dim=1)
        result = torch.cat(
            [
                network(torch.cat([y_, t], 1))
                for (network, y_) in zip(self.variance_networks, y_split)
            ],
            1,
        )
        return self.activation(result)


class GeneralVariance(nn.Module):
    def __init__(
        self, input_dim, model_dims, time_embedding_dim=3, activation="softplus"
    ):
        super(GeneralVariance, self).__init__()
        self.context_tensor = None
        self.variance_networks = network_factory(
            [input_dim + time_embedding_dim] + model_dims
        )

        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, t, y):
        ## t: time index of size batch_size x time_embedding_dim,
        ## y: size batch_size x latent or data_size
        result = self.variance_networks(torch.cat([y, t], 1))
        return self.activation(result)


class latentSDE(torchsde.SDEIto):
    def __init__(
        self, latent_dim, context_dim, drift_network_dims, variance_network_dims, args
    ):
        super(latentSDE, self).__init__(noise_type=args.noise_type)
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.context_tensor_lst = []
        self.noise_type_original = args.noise_type
        self.prior_drift = network_factory(
            [latent_dim + args.time_embedding_dim] + drift_network_dims
        )
        self.posterior_drift = network_factory(
            [latent_dim + args.time_embedding_dim + context_dim] + drift_network_dims
        )
        self.variance_presoftplus = nn.Parameter(torch.rand(1, self.latent_dim))

        self.method = args.method
        self.dt = args.dt
        self.dt_min = args.dt_min
        self.dt_test = args.dt_test
        self.dt_min_test = args.dt_min_test
        self.adaptive = args.adaptive
        if args.noise_type == "general":
            self.diagonal_variance = GeneralVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        elif args.noise_type == "additive":
            self.diagonal_variance = ConstantVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        elif args.noise_type == "diagonal":
            self.diagonal_variance = DiagonalVariance(
                latent_dim, variance_network_dims, activation=args.variance_act
            )
        else:
            raise NotImplementedError

        self.rtol = args.rtol
        self.atol = args.atol
        ## Get the index for context
        self.context_index = 0
        self.sdeint_noadjoint = torchsde.sdeint_adjoint

        self.register_buffer("time_zero", torch.tensor([0]))

    def update_context(self, context_tensor, context_time):
        self.context_tensor_lst.append((context_tensor, context_time))

    def clear_context(self):
        self.context_tensor_lst = []
        self.context_index = 0

    def f(self, t, y):  # Approximate posterior drift.
        context_index = -1
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.posterior_drift(
            torch.cat([t, self.context_tensor_lst[context_index][0], y], 1)
        )

    def g(self, t, y):  # Shared diffusion.
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.diagonal_variance(t, y)

    def g_diag(self, t, y):
        # Return G as a diagonal matrix
        batch_size = y.shape[0]
        return torch.diag_embed(
            F.softplus(self.variance_presoftplus).repeat(batch_size, 1)
        )

    def h(self, t, y):  # Prior drift.
        t = t.unsqueeze(0).unsqueeze(0)
        t = time_embedding(t)
        t = t.repeat(y.shape[0], 1)
        return self.prior_drift(torch.cat([t, y], 1))

    def h_diag(self, t, y):
        return torch.diag_embed(self.h(t, y))

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def h_aug(self, t, y):
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([h, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, :-1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        g_logqp = _stable_division(f - h, g)
        g = torch.diag_embed(g)
        ## size of g is batch_size x latent x latent
        return torch.cat([g, g_logqp.unsqueeze(1)], dim=1)

    def g_prior(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        g = self.g(t, y)
        return torch.diag_embed(g)

    def forward(self, y_t0, t1, context_tensor_time=None, first_time_forward=False):
        if context_tensor_time is not None:
            self.update_context(*context_tensor_time)
        batch_size = y_t0.shape[0]
        bm = torchsde.BrownianInterval(
            t0=context_tensor_time[1], t1=t1, size=y_t0.shape, device=y_t0.device
        )
        dt = self.dt
        aug_adaptive = self.adaptive
        dt_min = self.dt_min

        ## Running the augmented SDE using euler maruyama scheme to get the logpq
        ## Always set noit type to general and use euler-maruyama method to run
        ## stochastic integral.

        self.noise_type = "general"
        aug_y0 = torch.cat([y_t0, torch.zeros(batch_size, 1, device=y_t0.device)], dim=1)
        aug_ys = self.sdeint_noadjoint(
            sde=self,
            y0=aug_y0,
            ts=torch.cat([context_tensor_time[1].unsqueeze(0), t1]),
            method=self.method,
            dt=dt,
            adaptive=aug_adaptive,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        self.noise_type = self.noise_type_original
        return aug_ys[1]

    def run(self, y_t0, t, context_tensor_time=None):
        if context_tensor_time is not None:
            self.update_context(*context_tensor_time)
        batch_size = y_t0.shape[0]
        dt = self.dt
        aug_adaptive = self.adaptive
        dt_min = self.dt_min

        ## Running the augmented SDE using euler maruyama scheme to get the logpq
        ## Always set noit type to general and use euler-maruyama method to run
        ## stochastic integral.

        self.noise_type = "general"
        aug_y0 = torch.cat([y_t0, torch.zeros(batch_size, 1, device=y_t0.device)], dim=1)
        aug_ys = self.sdeint_noadjoint(
            sde=self,
            y0=aug_y0,
            ts=t,
            method=self.method,
            dt=dt,
            adaptive=aug_adaptive,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        self.noise_type = self.noise_type_original
        return aug_ys.transpose(0, 1)

    def sample_from_posterior(self, y_t0, ts, context_tensor_time=None):
        if context_tensor_time is not None:
            self.update_context(*context_tensor_time)
        self.noise_type = "general"
        batch_size = y_t0.shape[0]
        dt = self.dt
        dt_min = self.dt_min

        aug_y0 = torch.cat([y_t0, torch.zeros(batch_size, 1, device=y_t0.device)], dim=1)
        aug_ys = self.sdeint_noadjoint(
            sde=self,
            y0=aug_y0,
            ts=torch.cat([context_tensor_time[1].unsqueeze(0), ts]),
            method="euler",
            dt=dt,
            adaptive=True,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )
        self.noise_type = self.noise_type_original

        return aug_ys[1:]

    def sample_from_prior(self, y_t0, ts):
        dt = self.dt
        dt_min = self.dt_min
        batch_size = y_t0.shape[0]
        bm = torchsde.BrownianInterval(
            t0=torch.zeros_like(ts[-1]), t1=ts[-1], size=y_t0.shape, device=y_t0.device
        )
        self.noise_type = "general"
        ys = self.sdeint_noadjoint(
            sde=self,
            y0=y_t0,
            ts=ts,
            method=self.method,
            dt=dt,
            adaptive=True,
            rtol=self.rtol,
            atol=self.atol,
            dt_min=dt_min,
            names={"drift": "h", "diffusion": "g_prior"},
        )
        self.noise_type = self.noise_type_original
        return ys


class RNNEncoder(nn.Module):
    def __init__(self, rnn_input_dim, hidden_dim, encoder_network=None):
        super(RNNEncoder, self).__init__()
        self.encoder_network = encoder_network
        self.rnn_cell = nn.GRUCell(rnn_input_dim, hidden_dim)

    def forward(self, h, x_current, y_prev, t_current, t_prev):
        t_current = torch.ones(x_current.shape[0], 1, device=t_current.device) * t_current
        t_prev = torch.ones_like(t_current) * t_prev

        if self.encoder_network is None:
            t_diff = t_current - t_prev
            t_current = time_embedding(t_current)
            t_prev = time_embedding(t_prev)
            input = torch.cat([x_current, y_prev, t_current, t_prev, t_diff], 1)
        else:
            input = self.encoder_network(torch.cat([x_current, y_prev, t_current, t_prev], 1))
        return self.rnn_cell(input, h)


class LatentSDE2(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_dims, encoder_hidden_state_dim, encoder_hidden_dims, dt, method):
        super(LatentSDE2, self).__init__()
        # Encoder.
        self.encoder = CDE(data_size, encoder_hidden_dims, encoder_hidden_state_dim, context_size)
        # self.encoder = Encoder(data_size, encoder_hidden_state_dim, context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = MLP(latent_size + context_size, hidden_dims, latent_size, activation='Softplus')

        self.h_net = MLP(latent_size, hidden_dims, latent_size, activation='Softplus')

        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                MLP(1, [hidden_dims[0]], 1, activation='Softplus', final_activation='Sigmoid')
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self.dt = dt
        self.method = method
        self.sde_type = 'ito' if method == 'euler' else 'stratonovich'

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[:, i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    # def forward(self, xs, ts, adjoint=False, method="euler"):
    #     # Contextualization is only needed for posterior inference.
    #     ctx = self.encoder(xs)
    #     self.contextualize((ts, ctx))
    #
    #     qz0_mean, qz0_logstd = self.qz0_net(ctx[:, 0]).chunk(chunks=2, dim=1)
    #     qz0_logstd = torch.clamp(qz0_logstd, -20, 2)
    #     z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
    #
    #     if adjoint:
    #         # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
    #         adjoint_params = (
    #                 (ctx,) +
    #                 tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
    #         )
    #         zs, log_ratio = torchsde.sdeint_adjoint(
    #             self, z0, ts, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=self.method)
    #     else:
    #         zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=self.dt, logqp=True, method=self.method)
    #
    #     _xs = self.projector(zs.transpose(0, 1))
    #     qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
    #     pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
    #     logqp0 = pz0.log_prob(z0).sum(-1, keepdims=True) \
    #                       - qz0.log_prob(z0).sum(-1, keepdims=True)
    #     logqp_path = log_ratio.transpose(0, 1).sum(dim=-1, keepdims=True)
    #     return _xs, logqp0 - logqp_path

    def forward(self, xs, ts, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        ts_repeat = ts.unsqueeze(0).unsqueeze(-1).repeat_interleave(xs.shape[0], dim=0)
        ctx = self.encoder(torch.cat([xs, ts_repeat], dim=-1), intermediate=True)
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[:, 0]).chunk(chunks=2, dim=1)
        qz0_logstd = torch.clamp(qz0_logstd, -20, 2)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=self.dt, logqp=True, method=self.method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=self.dt, logqp=True, method=self.method)

        _xs = self.projector(zs.transpose(0, 1))
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = pz0.log_prob(z0).sum(-1, keepdims=True) \
                          - qz0.log_prob(z0).sum(-1, keepdims=True)
        logqp_path = log_ratio.transpose(0, 1).sum(dim=-1, keepdims=True)
        return _xs, logqp0 - logqp_path

    @torch.no_grad()
    def sample_from_prior(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=self.dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs).transpose(0, 1)
        return _xs

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out