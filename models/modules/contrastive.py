import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np

#### NOTE: a Contrastive Loss was not used for the published model, but is available for testing and future work.

def ctr_loss(q, k, tau):
    '''
    Simple similarity based contrastive loss
    bigger batch = better

    q shape: [N, D] or [B, N, D]
    k shape: [N, D] or [B, N, D]
    tau: temperature
    '''
    N, d = q.shape[-2], q.shape[-1]
    logits = torch.einsum('...Nd, ...Md-> ...NM', q, k)
    labels = torch.arange(N, device=q.device)
    loss = F.cross_entropy(logits / tau, labels)
    return 2*tau*loss

class ContrastiveLoss(nn.Module):
    def __init__(self, 
                 dim, 
                 proj_dim=128, 
                 hidden_dim=2048, 
                 tau=0.1,
                 tau_min=0.01,
                 tau_max=100.0, 
                 learn_tau=True,
                 dual_proj_head=False):
        super().__init__()
        self.proj_head = ProjectionHead(in_dim=dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.proj_head2 = None
        if dual_proj_head:
            self.proj_head2 = ProjectionHead(in_dim=dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        
        self.tau_min = tau_min
        self.tau_max = tau_max
        if learn_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(tau)))
        else:
            self.register_buffer('log_tau', torch.log(torch.tensor(tau)))
    def get_parameter_groups(self, weight_decay=0.01):
        wd_params, nowd_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if 'bias' in name or 'bn' in name or 'norm' in name or 'tau' in name or 'log_tau' in name:
                nowd_params.append(param)
            else:
                wd_params.append(param)
        return [{'params': wd_params, 'weight_decay': weight_decay},
                {'params': nowd_params, 'weight_decay': 0.0}]

    def get_tau(self):
        return self.log_tau.exp()
    
    def forward(self, q, k):
        q = self.proj_head(q)
        if self.proj_head2 is not None:
            k = self.proj_head2(k)
        else:
            k = self.proj_head(k)

        #clamp log_tau instead
        self.log_tau.data = self.log_tau.data.clamp(min=np.log(self.tau_min), 
                                                    max=np.log(self.tau_max) )
        tau = self.get_tau()

        return ctr_loss(q, k, tau)
