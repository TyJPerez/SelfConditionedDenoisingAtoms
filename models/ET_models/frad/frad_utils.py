import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph




class Distance(nn.Module):
    """
    Module for computing interatomic distances and corresponding vectors based on position data.

    Args:
        cutoff_lower (float): Lower cutoff distance for interatomic interactions.
        cutoff_upper (float): Upper cutoff distance for interatomic interactions.
        max_num_neighbors (int, optional): Maximum number of neighbors to return for each atom.
            Defaults to 32.
        return_vecs (bool, optional): Whether to return distance vectors. Defaults to False.
        loop (bool, optional): Whether to include self-loops in the graph. Defaults to False.

    Inputs:
        pos (torch.Tensor): Tensor of atom positions with shape (num_atoms, num_dimensions).
        batch (torch.Tensor): Tensor indicating the batch index for each atom.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.LongTensor): Index tensor of shape (2, num_edges) representing
            edges between atoms.
            - edge_weight (torch.Tensor): Tensor of shape (num_edges,) representing edge weights
            (distances or norms of distance vectors).
            - edge_vec (torch.Tensor or None): Tensor of shape (num_edges, num_dimensions) representing
            distance vectors if `return_vecs` is True, otherwise None.
    """
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None