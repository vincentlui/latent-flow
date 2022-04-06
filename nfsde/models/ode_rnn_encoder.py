# Copyright (c) 2019-present Royal Bank of Canada
# Copyright (c) 2019 Yulia Rubanova
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import lib.utils as utils
# This code is based on latent ODE project which can be found at: https://github.com/YuliaRubanova/latent_ode copy
import torch.nn as nn
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN
from lib.ode_func import ODEFunc


def create_ode_rnn_encoder(
    dim,
    posterior_hidden_dim,
    z_dim,
    hidden_dim,
    hidden_layers,
):
    """
    return an ode-rnn model
    """
    enc_input_dim = dim * 2  ## concatenate the mask with input

    ode_func_net = utils.create_net(
        posterior_hidden_dim,
        posterior_hidden_dim,
        n_layers=hidden_layers,
        n_units=hidden_dim,
        nonlinear=nn.Tanh,
    )

    rec_ode_func = ODEFunc(
        input_dim=enc_input_dim,
        latent_dim=posterior_hidden_dim,
        ode_func_net=ode_func_net,
    )

    z0_diffeq_solver = DiffeqSolver(
        enc_input_dim,
        rec_ode_func,
        "euler",
        z_dim,
        odeint_rtol=1e-3,
        odeint_atol=1e-4,
    )

    encoder_z0 = Encoder_z0_ODE_RNN(
        posterior_hidden_dim,
        enc_input_dim,
        z0_diffeq_solver,
        z0_dim=z_dim,
        n_gru_units=hidden_dim,
    )
    return encoder_z0
