from pytorch_lightning.callbacks import Callback
import numpy as np
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
# import iisignature


def calc_KL(mu1, log_std1, mu2=0., log_std2=0.):
    return .5 * torch.sum(2*(log_std2 - log_std1)
            + ((mu1 - mu2) ** 2 + log_std1.exp() ** 2) /
            log_std2.exp() ** 2 - 1, dim=-1, keepdim=True)


# def calc_sig(dl, module, t_target=None, num_levels=5, sample=50):
#     s1 = None
#     s2 = None
#     num_sample = 0.
#     num_data = 0.
#     for batch in dl:
#         x, t, y_true = batch
#         t_detach, y_true_detach = t.detach().cpu().numpy(), y_true.detach().cpu()
#         for i in range(sample):
#             num_sample += x.shape[0]
#             if module.__class__.__name__ in ['FlowMC', 'CTFPModule', 'FlowMCZ']:
#                 y = module(t)
#             elif module.__class__.__name__ in ['CLPFModule', 'LSDE']:
#                 y = module(t[0].flatten(), num_sample=t.shape[0])
#             y = y.detach().cpu().numpy()
#             streams = np.concatenate([t_detach, y], axis=-1)
#             sig_model = np.asarray([iisignature.sig(s, num_levels) for s in streams]).sum(axis=0)
#             if s1 is None:
#                 s1 = sig_model
#             else:
#                 s1 += sig_model

#         num_data += x.shape[0]
#         streams2 = np.concatenate([t_detach, y_true_detach], axis=-1)

#         sig_true = np.asarray([iisignature.sig(s, num_levels) for s in streams2]).sum(axis=0)
#         if s2 is None:
#             s2 = sig_true
#         else:
#             s2 += sig_true

#     sig = np.sum((s1 / num_sample - s2 / num_data) ** 2)
#     return sig


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __deepcopy__(self, memo=None):
        return dotdict(deepcopy(dict(self), memo=memo))


class ResumeTrainingCallback(Callback):
    def __init__(self):
        self.is_log = False

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_log:
            print('train_start')
            self.is_log = True
            # pl_module.eval()
            # i = 0
            # for batch in pl_module.val_dataloader():
            #     pl_module.validation_step(batch, i)
            # pl_module.train()
            pl_module.log('val_loss', torch.tensor(1000000.))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_log:
            print('train_end')
            self.is_log = True
            # pl_module.eval()
            # i = 0
            # for batch in pl_module.val_dataloader():
            #     pl_module.validation_step(batch, i)
            # pl_module.train()
            pl_module.log('val_loss', torch.tensor(1000000.))

class MetricsCallback(Callback):
    def __init__(self, args):
        self.losses = []
        self.args = args

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        loss = outputs
        self.losses.append(loss.item())

    def get_test_loss(self):
        return np.mean(self.losses)


class SavePlotCallback(Callback):
    def __init__(self, args):
        self.losses = []
        self.args = args

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        save_dir = trainer.log_dir + '/plot.png'
        t, y = self._sample_trajectories(trainer, pl_module)
        if y.shape[-1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            for k in range(t.shape[0]):
                # print(seq[k+i][:, 0])
                ax.plot(y[k][:, 0], y[k][:, 1], y[k][:, 2])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            for i in range(t.shape[0]):
                ax.plot(t[i], y[i])
        fig.savefig(save_dir)
        plt.close(fig)

    def _sample_trajectories(self, trainer, pl_module):
        for batch in pl_module.test_dataloader():
            x, t, y = batch
            x, t = x.to(pl_module.device), t.to(pl_module.device)
            time_max = t.max()
            t_space = torch.linspace(0., time_max, 51,
                               device=pl_module.device).unsqueeze(0) \
                .repeat_interleave(x.shape[0], dim=0).unsqueeze(-1)
            if pl_module.__class__.__name__ in ['FlowVAE']:
                t = t_space
                y = pl_module(x, t)
            elif pl_module.__class__.__name__ in ['FlowMC', 'CTFPModule', 'FlowMCZ']:
                t = t_space
                y = pl_module(t)
            elif pl_module.__class__.__name__ in ['CLPFModule', 'LSDE']:
                t = t_space
                y = pl_module(t_space[0].flatten(), num_sample=t_space.shape[0])
            else:
                y = pl_module(t, averaged=True)
            break
        if y.shape[0] > 10:
            y = y[:10]
            t = t[:10]
        if pl_module.mean is not None:
            y = pl_module.std * y + pl_module.mean
        return t.detach().cpu().numpy(), y.detach().cpu().numpy()