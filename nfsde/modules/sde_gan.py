import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from nfsde.experiments.synthetic.data import get_data_loaders
from nfsde.models.discriminator import CDEDiscriminator
from nfsde.util import dotdict

import torch
import matplotlib.pyplot as plt
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
import tqdm

###################
# First some standard helper objects.
###################

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 torch.nn.LeakyReLU()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(torch.nn.LeakyReLU())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


###################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
###################
class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size).type_as(ts)
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=ts[1] - ts[0],
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))



class SDEGAN(pl.LightningModule):
    def __init__(
            self, args, data
    ):
        super().__init__()
        if isinstance(args, dict):
            args = dotdict(args)
        self.args = args
        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, mean, std = data
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.generator = Generator(
            self.dim,
            args.initial_noise_size,
            args.noise_size,
            args.hidden_state_dim,
            args.hidden_dim,
            args.hidden_layers
        )
        self.discriminator = CDEDiscriminator(
            self.dim,
            args.d_hidden_state_dim,
            args.d_hidden_dim,
            args.d_hidden_layers
        )
        self.swa_step_start = self.args.swa_step_start
        self.averaged_generator = swa_utils.AveragedModel(self.generator)
        self.averaged_discriminator = swa_utils.AveragedModel(self.discriminator)
        self.automatic_optimization = False

        with torch.no_grad():
            for param in self.generator._initial.parameters():
                param *= args.init_mult1
            for param in self.generator._func.parameters():
                param *= args.init_mult2

        self.save_hyperparameters(args)

    def forward(self, ts, averaged=False):
        if not averaged:
            return self.generator(ts[0].squeeze(-1), ts.shape[0])
        self.train()
        with torch.no_grad():
            out = self.averaged_generator(ts[0].squeeze(-1), ts.shape[0])
        self.eval()
        return out[..., 1:]

    def training_step(self, batch, batch_idx,):
        g_opt, d_opt = self.optimizers()

        d_opt.zero_grad()
        g_opt.zero_grad()
        loss = self._get_loss(batch, batch_idx, 'train')
        self.manual_backward(loss)
        for param in self.generator.parameters():
            param.grad *= -1
        d_opt.step()
        g_opt.step()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, batch_idx, 'test')
        return loss

    def _get_loss(self, batch, batch_idx, tag):
        _, ts, y_true = batch
        generated_samples = self.generator(ts[0].squeeze(-1), self.args.batch_size)
        generated_score = self.discriminator(generated_samples)
        real_score = self.discriminator(torch.cat([ts, y_true], dim=-1))
        loss = generated_score - real_score
        if tag == 'val':
            self.log(tag + '_generated_score', generated_score)
            self.log(tag + '_real_score', real_score)
            mean_dist = torch.mean((torch.mean(generated_samples[:,:,1:], dim=0) - torch.mean(y_true, dim=0)) ** 2)
            self.log('mean_dist', mean_dist)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.args.d_weight_clip > 0:
            with torch.no_grad():
                for module in self.discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features * self.args.d_weight_clip
                        module.weight.clamp_(-lim, lim)

        if self.current_epoch > self.swa_step_start:
            self.averaged_generator.update_parameters(self.generator)
            self.averaged_discriminator.update_parameters(self.discriminator)

    def configure_optimizers(self):
        generator_optimizer = optim.Adadelta(self.generator.parameters(), lr=self.args.g_lr,
                                             weight_decay=self.args.weight_decay)
        discriminator_optimizer = optim.Adadelta(self.discriminator.parameters(), lr=self.args.d_lr, weight_decay=self.args.weight_decay)
        return [generator_optimizer, discriminator_optimizer]

    def prepare_data(self):
        # download
        return

    # def setup(self, stage=None):
    #     self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest, self.mean, self.std = \
    #         get_data_loaders(self.args.data, self.args.batch_size)

    def train_dataloader(self):
        return self.dltrain

    def val_dataloader(self):
        return self.dlval

    def test_dataloader(self):
        return self.dltest