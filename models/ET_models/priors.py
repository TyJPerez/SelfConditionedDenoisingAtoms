# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from typing import Optional, Dict
import torch
from torch import nn, Tensor
from lightning_utilities.core.rank_zero import rank_zero_warn

__all__ = ["Atomref", "LearnableAtomref", "PositiveOutput"]

class BasePrior(nn.Module):
    r"""Base class for prior models.
    Derive this class to make custom prior models, which take some arguments and a dataset as input.
    As an example, have a look at the `torchmdnet.priors.atomref.Atomref` prior.
    """

    def __init__(self, dataset=None):
        super().__init__()

    def get_init_args(self):
        r"""A function that returns all required arguments to construct a prior object.
        The values should be returned inside a dict with the keys being the arguments' names.
        All values should also be saveable in a .yaml file as this is used to reconstruct the
        prior model from a checkpoint file.
        """
        return {}

    def pre_reduce(self, x, z, pos, batch, extra_args: Optional[Dict[str, Tensor]]):
        r"""Pre-reduce method of the prior model.

        Args:
            x (torch.Tensor): scalar atom-wise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.
            extra_args (dict): any addition fields provided by the dataset

        Returns:
            torch.Tensor: updated scalar atom-wise predictions
        """
        return x

    def post_reduce(
        self,
        y,
        z,
        pos,
        batch,
        box: Optional[Tensor],
        extra_args: Optional[Dict[str, Tensor]],
    ):
        r"""Post-reduce method of the prior model.

        Args:
            y (torch.Tensor): scalar molecule-wise predictions from the model.
            z (torch.Tensor): atom types of all atoms.
            pos (torch.Tensor): 3D atomic coordinates.
            batch (torch.Tensor): tensor containing the sample index for each atom.
            box (Optional[torch.Tensor]): box vectors of the system.
            extra_args (dict): any addition fields provided by the dataset

        Returns:
            torch.Tensor: updated scalar molecular-wise predictions
        """
        return y


# class PositiveOutput(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
    
#     def pre_reduce(self, x, z, pos, batch, extra_args: Optional[Dict[str, Tensor]]):
#         return x

#     def post_reduce(self, y, z, pos, batch, box: Optional[Tensor], extra_args: Optional[Dict[str, Tensor]]):
#         return y.abs()
    
    # def forward(self, x):
    #     return torch.relu(x)


class Atomref(BasePrior):
    r"""Atomref prior model.

    This prior model is used to add atomic reference values to the input features. The atomic reference values are stored in an embedding layer and are added to the input features as:

    .. math::

        x' = x + \\textrm{atomref}(z)

    where :math:`x` is the input feature tensor, :math:`z` is the atomic number tensor, and :math:`\\textrm{atomref}` is the embedding layer. The atomic reference values are stored in the embedding layer and can be trainable.

    When using this in combination with some dataset, the dataset class must implement the function `get_atomref`, which returns the atomic reference values as a tensor.

    Args:
        max_z (int, optional): Maximum atomic number to consider. If `dataset` is not `None`, this argument is ignored.
        dataset (torch_geometric.data.Dataset, optional): A dataset from which to extract the atomref values.
        trainable (bool, optional): If `False`, the atomref values are not trainable. (default: `False`)
        enable (bool, optional): If `False`, the prior is disabled. This is useful if you want to add the reference energies only during inference (or training) (default: `True`)
    """

    def __init__(self, max_z=None, dataset=None, trainable=False, enable=True):
        super().__init__()
        if max_z is None and dataset is None:
            raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
        if dataset is None:
            assert max_z is not None, "max_z must be provided if dataset is None."
            atomref = torch.zeros(max_z, 1)
        else:
            if hasattr(dataset, "get_atomref") and callable(getattr(dataset, "get_atomref")):
                atomref = dataset.get_atomref()
                # print(f"---> Successfully loaded atomref from dataset: {atomref.shape}")
            else:
                rank_zero_warn(
                    "The dataset does not have a 'get_atomref' method, defaulting to zeros with max. atomic number 99."
                )
                atomref = torch.zeros(100, 1)

            if atomref is None:
                rank_zero_warn(
                    "The atomref returned by the dataset is None, defaulting to zeros with max. "
                    "atomic number 99. Maybe atomref is not defined for the current target."
                )
                atomref = torch.zeros(100, 1)

        if atomref.ndim == 1:
            atomref = atomref.view(-1, 1)
        self.register_buffer("initial_atomref", atomref)
        self.atomref = nn.Embedding(
            len(atomref), 1, _freeze=not trainable, _weight=atomref
        )
        self.enable = enable

        #create embeddings to scale the atomrefs
        # self.atomref_scale = nn.Embedding(len(atomref), 1, _freeze=not trainable, _weight=atomref)  
        #initialize to 1
        # self.atomref_scale.weight.data.fill_(1.0)

    def reset_parameters(self):
        self.atomref.weight.data.copy_(self.initial_atomref)
        # self.atomref_scale.weight.data.fill_(1.0)

    def get_init_args(self):
        return dict(
            max_z=self.initial_atomref.size(0),
            trainable=self.atomref.weight.requires_grad,
            enable=self.enable,
        )

    def pre_reduce(
        self,
        x: Tensor,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ):
        """Applies the stored atomref to the input as:

        .. math::

            x' = x + \\textrm{atomref}(z)

        .. note:: The atomref operation is an embedding lookup that can be trainable if the `trainable` argument is set to `False`.

        .. note:: This call becomes a no-op if the `enable` argument is set to `False`.

        Args:
            x (Tensor): Input feature tensor.
            z (Tensor): Atomic number tensor.
            pos (Tensor): Atomic positions tensor. Unused.
            batch (Tensor, optional): Batch tensor. Unused. (default: `None`).
            extra_args (Dict[str, Tensor], optional): Extra arguments. Unused. (default: `None`)


        """
        if self.enable:
            return x + self.atomref(z)
            # return x*self.atomref_scale(z) + self.atomref(z) 
        else:
            return x


class LearnableAtomref(Atomref):
    r"""LearnableAtomref prior model.

    This prior model is used to add learned atomic reference values to the input features. The atomic reference values are learned as an embedding layer and are added to the input features as:

    .. math::

        x' = x + \\textrm{atomref}(z)

    where :math:`x` is the input feature tensor, :math:`z` is the atomic number tensor, and :math:`\\textrm{atomref}` is the embedding layer.


    Args:
        max_z (int, optional): Maximum atomic number to consider.
    """

    def __init__(self, max_z=None, **kwargs):
        super().__init__(max_z, trainable=True, enable=True, **kwargs)



from typing import Optional, Dict
from tqdm import tqdm
import wandb 
from torch_geometric.data import DataLoader

from torch_geometric.nn.pool import global_add_pool, global_mean_pool

#import torch_geometric data and loader



# class ElementNorm(BasePrior):
#     r"""
#     Shift and Scale model outputs by norm of target by element type 
    
#     Args:
#         max_z (int, optional): Maximum atomic number to consider. If `dataset` is not `None`, this argument is ignored.
#         dataset (torch_geometric.data.Dataset, optional): A dataset from which to extract the atomref values.
#         trainable (bool, optional): If `False`, the atomref values are not trainable. (default: `False`)
#         enable (bool, optional): If `False`, the prior is disabled. This is useful if you want to add the reference energies only during inference (or training) (default: `True`)
#     """

#     def __init__(self, max_z=118, dataset=None, trainable=False, enable=True):
#         super().__init__()
#         if max_z is None and dataset is None:
#             raise ValueError("Can't instantiate Atomref prior, all arguments are None.")
#         if dataset is None:
#             assert False, "ElementNorm prior requires a dataset to extract norms."
#             # assert max_z is not None, "max_z must be provided if dataset is None."
#             # atomref = torch.zeros(max_z, 1)
#         # else:
#         #     if hasattr(dataset, "get_atomref") and callable(getattr(dataset, "get_atomref")):
#         #         atomref = dataset.get_atomref()
#         #     else:
#         #         rank_zero_warn(
#         #             "The dataset does not have a 'get_atomref' method, defaulting to zeros with max. atomic number 99."
#         #         )
#         #         atomref = torch.zeros(100, 1)
#         #     if atomref is None:
#         #         rank_zero_warn(
#         #             "The atomref returned by the dataset is None, defaulting to zeros with max. "
#         #             "atomic number 99. Maybe atomref is not defined for the current target."
#         #         )
#         #         atomref = torch.zeros(100, 1)
#         # if atomref.ndim == 1:
#         #     atomref = atomref.view(-1, 1)
#         # self.register_buffer("initial_atomref", atomref)

#         # self.atomref = nn.Embedding(
#         #     len(atomref), 1, _freeze=not trainable, _weight=atomref
#         # )

#         self.enable = enable

#         #create embeddings to scale the atomrefs
#         # self.atomref_scale = nn.Embedding(len(atomref), 1, _freeze=not trainable, _weight=atomref)  
#         #initialize to 1
#         # self.atomref_scale.weight.data.fill_(1.0)

#         # self.element_norms = {} # dict of {z : (mean, std)} per element in dataset


#         self.use_global_norm = True
#         self.z_means = nn.Embedding(max_z, 1)
#         self.z_stds = nn.Embedding(max_z, 1)

#         self.register_buffer("global_mean", torch.tensor(0.0))
#         self.register_buffer("global_std", torch.tensor(1.0))
        

#         self.eps = 1e-12
#         self.compute_per_elem_norms(dataset, divby_natoms=False)

#         self.fit(dataset, device='cuda', batch_size=128, num_epochs=5, lr=0.5)

#         # this creates:
#         # self.element_norms : dict of {z: (mean, std)}
#         # self.initial_means : tensor of shape (max_z, 1)
#         # self.initial_stds : tensor of shape (max_z, 1)
#         # self.z_means = nn.Embedding
#         # self.z_stds = nn.Embedding
#     def fit(self, dataset, device, batch_size=16, num_epochs=2, lr=0.5, ):
        
#         self.reduce_op = 'sum'  # or 'mean' depending on how model outputs are reduced
#         reduce_fn = global_mean_pool if self.reduce_op == 'mean' else global_add_pool


#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         #set up wandb logging
#         proj_name = "qm9_norm_testing"
#         # time_stamp = wandb.util.generate_id()
#         wandb.init(project=proj_name, 
#                    name="element_norm_fit")
        
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
#         self.train()
#         self.freeze_embeddings(flag=False)
#         self.to(device)
#         for epoch in range(num_epochs):
#             for batch in tqdm(loader, desc=f"Fitting ElementNorm Epoch {epoch+1}/{num_epochs}"):
#                 batch = batch.to(device)
#                 z = batch.z
#                 y = batch.y
#                 x = torch.randn((z.size(0), 1), device=device)*0.5  # dummy input features
#                 out = self.pre_reduce(x, z, None, batch.batch, None)
#                 y_pred = reduce_fn(out, batch.batch)
#                 y_pred_unscale = self.post_reduce(y_pred, z)

#                 if self.use_global_norm:
#                     y_norm = (y - self.global_mean) / (self.global_std + self.eps)
#                 else:
#                     y_norm = y
                
#                 loss = nn.MSELoss()(y_norm, y_pred)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 loss_unscale = nn.MSELoss()(y, y_pred_unscale.detach())

#                 # log to wandb
#                 wandb.log({"train_loss": loss.item(),
#                            "unscaled_loss": loss_unscale.item(),
#                            'epoch': epoch})

#         #freeze embeddings after fitting
#         self.freeze_embeddings(flag=True)
    
#     def freeze_embeddings(self, flag=True):
#         self.z_means.weight.requires_grad = not flag
#         self.z_stds.weight.requires_grad = not flag

#     def compute_per_elem_norms(self, dataset, divby_natoms=True):
#         targets_dict = {}
#         max_sample = 20000

#         all_y_vals = []
#         for data in tqdm(dataset):
#             z = data.z # [N]
#             y = data.y # [1]

#             all_y_vals.append(y)

#             if divby_natoms:
#                 natoms = z.size(0)
#                 y = y / natoms

#             natoms = z.size(0)
#             for zi in z:
#                 zi = zi.item()

#                 if zi not in targets_dict:
#                     targets_dict[zi] = []
#                 targets_dict[zi].append(y)
        
#             if len(all_y_vals) >= max_sample:
#                 #DEBUGGING
#                 break

#         all_y_tensor = torch.stack(all_y_vals)
#         global_mean = all_y_tensor.mean().item()
#         global_std = all_y_tensor.std().item()
#         self.global_mean.fill_(global_mean)
#         self.global_std.fill_(global_std)

#         # print(f"Number of H atoms in dataset: {h_count}")
#         #compute mean and std per element
#         elem_norms = {}
#         max_z = max(targets_dict.keys()) + 1
#         means_ = torch.zeros(max_z, 1)
#         stds_ = torch.ones(max_z, 1)

#         for zi, y_list in targets_dict.items():
#             y_tensor = torch.tensor(y_list)

#             #scale the y_tensor by global mean and std
#             #TODO: debug
#             y_tensor = (y_tensor - global_mean) / (global_std + self.eps)

#             y_len = y_tensor.size(0)
#             mean = y_tensor.mean().item()
#             std = y_tensor.std().item()
#             means_[zi] = mean
#             stds_[zi] = std
#             elem_norms[zi] = (mean, std, y_len)
        
#         self.element_norms = elem_norms
#         self.register_buffer("initial_means", means_)
#         self.register_buffer("initial_stds", stds_)
#         #convert element_norms to nn.Embedding for faster lookup
#         freeze_vals = False
#         self.z_means = nn.Embedding(
#             len(means_), 1, _freeze=freeze_vals, _weight=means_
#         )
#         self.z_stds = nn.Embedding(
#             len(stds_), 1, _freeze=freeze_vals, _weight=stds_
#         )
        


#     # def reset_parameters(self):
#     #     self.atomref.weight.data.copy_(self.initial_atomref)
#         # self.atomref_scale.weight.data.fill_(1.0)

#     def get_init_args(self):
#         return dict(
#             max_z=self.initial_atomref.size(0),
#             trainable=self.atomref.weight.requires_grad,
#             enable=self.enable,
#         )

#     def pre_reduce(
#         self,
#         x: Tensor,
#         z: Tensor,
#         pos: Tensor = None,
#         batch: Optional[Tensor] = None,
#         extra_args: Optional[Dict[str, Tensor]] = None,
#     ):
#         """
#         Args:
#             x (Tensor): Input feature tensor.
#             z (Tensor): Atomic number tensor.
#             pos (Tensor): Atomic positions tensor. Unused.
#             batch (Tensor, optional): Batch tensor. Unused. (default: `None`).
#             extra_args (Dict[str, Tensor], optional): Extra arguments. Unused. (default: `None`)
        
#         """
#         if self.enable:
#             # natoms = torch.bincount(batch)
#             # mean = self.z_means(z)
#             # std = self.z_stds(z)
#             # unnormalize input x
#             x = x * self.z_stds(z).abs() + self.z_means(z)
#             # x = x + self.z_means(z)

#             # x = (x - mean) / (std + self.eps)  
#             return x
#             # return x + self.atomref(z)
#             # return x*self.atomref_scale(z) + self.atomref(z) 
#         else:
#             return x
    

#     def post_reduce(
#         self,
#         y: Tensor,
#         z: Tensor,
#         pos: Tensor = None,
#         batch: Optional[Tensor] = None,
#         box: Optional[Tensor] = None,
#         extra_args: Optional[Dict[str, Tensor]] = None,
#         ):
        
#         #apply global unnormalization
#         if self.enable:
#             if self.use_global_norm:
#                 y = y * (self.global_std + self.eps) + self.global_mean
#             return y
#         else:
#             return y
        
    