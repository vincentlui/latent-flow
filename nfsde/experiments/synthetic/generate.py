import numpy as np
import scipy.signal
import scipy.integrate
from pathlib import Path
from scipy.stats import norm
from numpy.random import default_rng
import torchsde
import torch


DATA_DIR = Path(__file__).parents[1] / 'data/synth'
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(123)

NUM_SEQUENCES = 10000
NUM_POINTS = 51
MAX_TIME = 63
EXTRAPOLATION_TIME = 0
NUM_LINSPACE_SAMPLE = 1001
NUM_LINSPACE_SAMPLE_NAN = 100

def get_inital_value(num, extrap_space, min=-1., max=1., dim=1):
    if extrap_space:
        return np.random.uniform(-4, -2, (num, dim,)) if np.random.rand() > 0.5 else np.random.uniform(2, 4, (dim,))
    else:
        return np.random.uniform(min, max, (num, dim,))

def get_inital_value2d(extrap_space):
    if extrap_space:
        return np.random.uniform(1, 2, (2,))
    else:
        return np.random.uniform(0, 1, (2,))

def get_data(sde, time_min, time_max, dt, dir, y0_min=-1., y0_max=1., dim=1,
             extrap_space=False, name=None, num_points=NUM_POINTS,
             num_linspace_sample=NUM_POINTS, random=False, random_nan=False):
    y0 = torch.Tensor(get_inital_value(NUM_SEQUENCES, extrap_space, y0_min, y0_max, dim))
    ts = torch.linspace(time_min,time_max,num_linspace_sample)
    y = torchsde.sdeint(sde, y0, ts, dt=dt)

    initial_values = y0.numpy()
    times = ts.unsqueeze(0).repeat_interleave(NUM_SEQUENCES, dim=0).numpy() / time_max
    sequences = y.transpose(0,1).numpy()
    if sde.__class__.__name__ == 'ContinuousAR4':
        sequences = sequences[..., :1]

    if random:
        rng = default_rng()
        t_ind = np.sort(np.apply_along_axis(rng.choice, arr=np.tile(np.arange(NUM_LINSPACE_SAMPLE),(NUM_SEQUENCES, 1)), size=num_points, axis=-1, replace=False), axis=-1)
        times = times[np.arange(NUM_SEQUENCES).repeat(num_points), t_ind.flatten()].reshape(NUM_SEQUENCES, num_points)
        sequences = sequences[np.arange(NUM_SEQUENCES).repeat(num_points), t_ind.flatten()].reshape(NUM_SEQUENCES, num_points, dim)
        name = name + '_random'

    if random_nan:
        rng = default_rng()
        t_ind = np.sort(np.apply_along_axis(rng.choice, arr=np.tile(np.arange(NUM_LINSPACE_SAMPLE_NAN),(NUM_SEQUENCES, 1)), size=num_points, axis=-1, replace=False), axis=-1)
        sequences_with_nan = np.empty_like(sequences)
        sequences_with_nan += np.nan
        sequences_with_nan[np.arange(NUM_SEQUENCES).repeat(num_points), t_ind.flatten()] = sequences[np.arange(NUM_SEQUENCES).repeat(num_points), t_ind.flatten()]
        sequences = sequences_with_nan
        name = name + '_nan'

    if name is None:
        return initial_values, times, sequences
    else:
        np.savez(dir / f'{name}.npz', init=initial_values, seq=sequences, time=times)

def generate(dir):
    # if not (dir / 'ou.npz').exists():
    #     ou = OU(mu=3., theta=0.4, sigma=0.3)
    #     get_data(ou, 0, 10, 0.01, dir, name='ou')

    if not (dir / 'ou.npz').exists():
        ou = OU(mu=4., theta=0.15, sigma=0.3)
        get_data(ou, 0, 30, 0.01, dir, name='ou')

    if not (dir / 'ou2.npz').exists():
        ou2 = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4)
        get_data(ou2, 0, 63, 1, dir, name='ou2')

    if not (dir / 'gbm.npz').exists():
        gbm = GBM(mu=0.1, sigma=0.2)
        get_data(gbm, 0, 30, 0.01, dir, name='gbm', y0_min=0.1, y0_max=1.)

    if not (dir / 'linear.npz').exists():
        linear = Linear()
        get_data(linear, 0, 30, 0.01, dir, name='linear', y0_min=-1., y0_max=1.)

    if not (dir / 'gompertzian.npz').exists():
        gompertzian = Gompertzian(0.2, 0.1)
        get_data(gompertzian, 0, 30, 0.01, dir, name='gompertzian', y0_min=0.1, y0_max=0.2)

    if not (dir / 'gompertzian2.npz').exists():
        gompertzian2 = Gompertzian2(0.2, 0.02)
        get_data(gompertzian2, 0, 30, 0.01, dir, name='gompertzian2', y0_min=0.1, y0_max=0.2)

    if not (dir / 'logistic.npz').exists():
        logistic = Logistic(0.4, 0.05, 100.)
        get_data(logistic, 0, 30, 0.01, dir, name='logistic', y0_min=0.1, y0_max=2.)

    if not (dir / 'car.npz').exists():
        car = ContinuousAR4(0.002, 0.005, -0.003, -0.002)
        get_data(car, 0, 30, 0.01, dir, name='car', dim=4, y0_min=-0.0001, y0_max=0.0001)

    if not (dir / 'lorenz.npz').exists():
        stochastic_lorenz = Lorenz(10., 28., 8./3, 0.3)
        get_data(stochastic_lorenz, 0, 2, 0.01, dir, dim=3, name='lorenz', y0_min=-1, y0_max=1, num_points=40)

    # if not (dir / 'ou_random.npz').exists():
    #     ou = OU(mu=3., theta=0.4, sigma=0.3)
    #     get_data(ou, 0, 10, 0.01, dir, name='ou', num_linspace_sample=NUM_LINSPACE_SAMPLE, random=True)

    # if not (dir / 'ou2_random.npz').exists():
    #     ou2 = OrnsteinUhlenbeckSDE(mu=0.02, theta=0.1, sigma=0.4)
    #     get_data(ou2, 0, 63, 0.01, dir, name='ou2', num_linspace_sample=NUM_LINSPACE_SAMPLE, random=True)

    # if not (dir / 'gbm_random.npz').exists():
    #     gbm = GBM(mu=0.1, sigma=0.2)
    #     get_data(gbm, 0, 30, 0.01, dir, name='gbm', y0_min=0.1, y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE, random=True)

    # if not (dir / 'linear_random.npz').exists():
    #     linear = Linear()
    #     get_data(linear, 0, 30, 0.01, dir, name='linear', y0_min=-1., y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE, random=True)

    # if not (dir / 'lorenz_random.npz').exists():
    #     stochastic_lorenz = Lorenz(10., 28., 8. / 3, 0.3)
    #     get_data(stochastic_lorenz, 0, 2, 0.001, dir, dim=3, name='lorenz', y0_min=-1, y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE, random=True)

    # if not (dir / 'ou_nan.npz').exists():
    #     ou = OU(mu=3., theta=0.4, sigma=0.3)
    #     get_data(ou, 0, 10, 0.01, dir, name='ou', num_linspace_sample=NUM_LINSPACE_SAMPLE_NAN, random_nan=True)

    # if not (dir / 'gbm_nan.npz').exists():
    #     gbm = GBM(mu=0.1, sigma=0.2)
    #     get_data(gbm, 0, 30, 0.01, dir, name='gbm', y0_min=0.1, y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE_NAN, random_nan=True)

    # if not (dir / 'linear_nan.npz').exists():
    #     linear = Linear()
    #     get_data(linear, 0, 30, 0.01, dir, name='linear', y0_min=-1., y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE_NAN, random_nan=True)

    # if not (dir / 'lorenz_nan.npz').exists():
    #     stochastic_lorenz = Lorenz(10., 28., 8. / 3, 0.3)
    #     get_data(stochastic_lorenz, 0, 2, 0.001, dir, dim=3, name='lorenz', y0_min=-1, y0_max=1., num_linspace_sample=NUM_LINSPACE_SAMPLE_NAN, random_nan=True)

    # if not (dir / 'ou3.npz').exists():
    #     ou = OU(mu=4., theta=0.15, sigma=0.15)
    #     get_data(ou, 0, 30, 0.1, dir, name='ou3')

def f_linear(x, t):
    return 0.5 * np.sin(t) * x + 0.5 * np.cos(t)

def g_linear(x, t):
    return 0.2 / (1 + np.exp(-t))


class OU(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, mu, theta, sigma):
        super().__init__()
        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('theta', torch.as_tensor(theta))
        self.register_buffer('sigma', torch.as_tensor(sigma))

    def f(self, t, y):
        return self.theta * (self.mu - y)

    def g(self, t, y):
        return self.sigma.expand(y.size(0), 1, 1)

class OrnsteinUhlenbeckSDE(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, mu, theta, sigma):
        super().__init__()
        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('theta', torch.as_tensor(theta))
        self.register_buffer('sigma', torch.as_tensor(sigma))

    def f(self, t, y):
        return self.mu * t - self.theta * y

    def g(self, t, y):
        return self.sigma.expand(y.size(0), 1, 1) * (2 * t / NUM_POINTS)

class GBM(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, mu, sigma):
        super().__init__()
        self.register_buffer('mu', torch.as_tensor(mu))
        self.register_buffer('sigma', torch.as_tensor(sigma))

    def f(self, t, y):
        return self.mu * y

    def g(self, t, y):
        return (self.sigma * y).unsqueeze(-1)

class Linear(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self):
        super().__init__()

    def f(self, t, y):
        return 0.5 * torch.sin(t) * y + 0.5 * torch.cos(t)

    def g(self, t, y):
        return 0.2 / (1 + torch.exp(-t)).expand(y.size(0), 1, 1)

class Gompertzian(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, b, c):
        super().__init__()
        self.register_buffer('b', torch.as_tensor(b))
        self.register_buffer('c', torch.as_tensor(c))

    def f(self, t, y):
        return - self.b * y * y.log()

    def g(self, t, y):
        return (self.c * y).unsqueeze(-1)

class Gompertzian2(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, b, c):
        super().__init__()
        self.register_buffer('b', torch.as_tensor(b))
        self.register_buffer('c', torch.as_tensor(c))

    def f(self, t, y):
        return - self.b * y * y.log()

    def g(self, t, y):
        return self.c.expand(y.size(0), 1, 1)

class Logistic(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, b, c, d, m=1):
        super().__init__()
        self.register_buffer('b', torch.as_tensor(b))
        self.register_buffer('c', torch.as_tensor(c))
        self.register_buffer('d', torch.as_tensor(d))
        self.register_buffer('m', torch.as_tensor(m))

    def f(self, t, y):
        return self.b * y * (1 - (y/self.d) ** self.m)

    def g(self, t, y):
        return (self.c * y).unsqueeze(-1)

class ContinuousAR4(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(self, a1, a2, a3, a4, e=torch.Tensor([0., 0., 0., 1.])):
        super().__init__()
        self.register_buffer('a1', torch.as_tensor(a1))
        self.register_buffer('a2', torch.as_tensor(a2))
        self.register_buffer('a3', torch.as_tensor(a3))
        self.register_buffer('a4', torch.as_tensor(a4))
        self.register_buffer('e', torch.as_tensor(e))
        a_m = torch.Tensor([
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [a1, a2, a3, a4]
        ])
        self.register_buffer('a_m', a_m)

    def f(self, t, y):
        return y @ self.a_m.T

    def g(self, t, y):
        return self.e.expand(y.size(0), -1)

class Lorenz(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(self, sigma, ro, beta, noise):
        super().__init__()
        self.register_buffer('sigma', torch.as_tensor(sigma))
        self.register_buffer('ro', torch.as_tensor(ro))
        self.register_buffer('beta', torch.as_tensor(beta))
        self.register_buffer('noise', torch.as_tensor(noise))

    def f(self, t, y):
        out = torch.empty_like(y)
        out[:, 0] = self.sigma * (y[:, 1] - y[:, 0])
        out[:, 1] = y[:, 0] * (self.ro - y[:, 2]) - y[:, 1]
        out[:, 2] = y[:, 0] * y[:, 1] - self.beta * y[:, 2]
        return out

    def g(self, t, y):
        return self.noise.expand(y.size(0), 3)



if __name__ == '__main__':
    generate(DATA_DIR)
