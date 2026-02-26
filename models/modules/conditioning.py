import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import scatter


class JointDropPath(nn.Module):
    # allows for a joint drop path on invariante (x) and equivariant (v) embeeddings
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x, v):
        if self.drop_prob == 0.0 or not self.training:
            return x, v
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return (x / keep_prob) * binary_mask, (v / keep_prob) * binary_mask.unsqueeze(-1)


class DyT(nn.Module):
    # Layer norm but fast
    # Publication: "Transformers without Normalization" https://arxiv.org/pdf/2503.10622v1
    def __init__(self, dim, init_alpha=0.2):
        super().__init__()
        self.norm_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.norm_gamma = nn.Parameter(torch.ones(dim)/init_alpha)
        self.norm_beta = nn.Parameter(torch.zeros(dim))

    def reset_parameters(self, init_alpha=0.2):
        nn.init.constant_(self.norm_alpha, init_alpha)
        nn.init.constant_(self.norm_gamma, 1.0/init_alpha)
        nn.init.constant_(self.norm_beta, 0.0)

    def forward(self, x):
        x = F.tanh(self.norm_alpha * x) * self.norm_gamma + self.norm_beta
        return x

class adaDyT2(nn.Module):
    # Layer norm but fast
    def __init__(self, dim, init_alpha=0.2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def reset_parameters(self, init_alpha=0.2):
        nn.init.constant_(self.alpha, init_alpha)

    def forward(self, x, scale, shift):
        alpha = torch.clamp(self.alpha.abs(), min=1e-5)
        gamma = (1+scale)/alpha # no tanh here
        x = F.tanh(alpha * x) * gamma + shift
        return x

class adaLN2(nn.Module):
    def __init__(self, dim, bias=True, init_alpha=0.2, norm_type='LN'):
        super().__init__()
        
        norm_class = DyT if norm_type == 'DyT' else nn.LayerNorm
        self.dim = dim
        self.x_norm = norm_class(dim)
        self.fc = nn.Sequential(
            nn.Linear(2*dim, dim, bias=bias),
            nn.SiLU(),
            norm_class(dim), 
            nn.Linear(dim, 3 * dim, bias=bias)
            )
        
        self.x_modulate = adaDyT2(dim, init_alpha=init_alpha)
        self.alpha_x = nn.Parameter(torch.tensor(init_alpha))
        self.init_params()
       
    def init_params(self, zero=False):
        if zero:
            nn.init.constant_(self.fc[-1].weight, 0)
            nn.init.constant_(self.fc[-1].bias, 0)
        else:
            nn.init.xavier_uniform_(self.fc[-1].weight)
            nn.init.constant_(self.fc[-1].bias, 0)
    
    def reset_parameters(self, norm_type='LN'):
        norm_class = DyT if norm_type == 'DyT' else nn.LayerNorm
        dim = self.dim
        self.x_norm = norm_class(dim)

        self.fc[0].reset_parameters()
        if isinstance(self.fc[2], (DyT, nn.LayerNorm)):
            self.fc[2] = norm_class(dim)
        else:
            raise ValueError("Unexpected layer type in fc")
        self.fc[3].reset_parameters()
        self.x_modulate.reset_parameters()
        nn.init.constant_(self.alpha_x, 0.2)
        self.init_params()
        
    def forward(self, x, c):
        if c is None:
            return self.x_norm(x), 1.0

        x = self.x_norm(x)
        c = torch.cat([c, x.detach()], dim=-1)
        shift_x, scale_x, gate_x = self.fc(c).chunk(3, dim=1)

        x = self.x_modulate(x, scale_x, shift_x)
        gate_x = F.tanh(gate_x * self.alpha_x)

        return x, gate_x


class DropCond(nn.Module):
    def __init__(self, dim, p_drop=0.25):
        super().__init__()
        self.p_drop_cond = p_drop
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.norm = nn.LayerNorm(dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_token)
        self.norm.reset_parameters()

    def forward(self, c, batch_idx):
        _num_graphs = batch_idx.max().item() + 1 if batch_idx.numel() > 0 else 0

        if c is None:
            c = self.mask_token.repeat(_num_graphs, 1)
        else:
            if self.training:
                drop_mask = torch.rand(c.size(0), device=c.device) < self.p_drop_cond
                drop_mask = drop_mask.float().unsqueeze(1) # (N, 1)
                c_mask = self.mask_token.repeat(_num_graphs, 1)
                c = c * (1 - drop_mask) + c_mask * drop_mask

        c = self.norm(c)
        #expand to node level
        c = c[batch_idx]
        return c


class ProjHead2(nn.Module):
    """Projection head for self-conditioning embedding."""
    def __init__(self, emb_dim, agg='sum'):
        super().__init__()
        self.emb_dim = emb_dim
        self.agg = agg

        self.pre_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.LayerNorm(emb_dim),
            )
        self.post_agg_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.pre_proj[1].weight)
        self.pre_proj[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.post_agg_mlp[1].weight)
        self.post_agg_mlp[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.post_agg_mlp[4].weight)
        self.post_agg_mlp[4].bias.data.fill_(0)

    def forward(self, x, batch):
        x = self.pre_proj(x)
        x = scatter(x, batch, dim=0, reduce=self.agg)  # sum pooled embeddings
        x = self.post_agg_mlp(x)
        return x

