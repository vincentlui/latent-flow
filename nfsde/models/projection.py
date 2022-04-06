import torch
from torch import nn
import stribor as st
from torch.distributions import Normal

class ProjectionNetwork(nn.Module):
    def __init__(
            self,
            flow_dim,
            data_dim,
            hidden_dims,
            flow_model
    ):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.flow = flow_model
        self.encoder = st.net.MLP(data_dim, hidden_dims, flow_dim*2)
        self.decoder = st.net.MLP(flow_dim, hidden_dims, data_dim)
        self.prior_p = Normal(torch.zeros(flow_dim).to(device), torch.ones(flow_dim).to(device))
        self.min_log_std, self.max_log_std = (-20, 2)

    def forward(self, x, t, w, latent=None, return_iw=False):
        mu_and_log_std =  self.encoder(x)
        mu, log_std = torch.chunk(mu_and_log_std, 2, -1)

        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        distribution = Normal(mu, std)
        y = distribution.rsample()
        z = self.flow(y, t, w, latent)
        if return_iw:
            log_q = distribution.log_prob(y).sum(dim=-1, keepdims=True)
            log_p = self.prior_p.log_prob(y).sum(dim=-1, keepdims=True)
            return self.decoder(z), log_q-log_p

        return self.decoder(z)
