# """
# Equivariant Transformer implementation.
# Extracted from torchmd-net for standalone use.

# Based on: Equivariant Transformers for Neural Network based Molecular Potentials
# P. Tholke and G. de Fabritiis. ICLR 2022.
# """

# from typing import Optional, Tuple
# import torch
# from torch import Tensor, nn
# from .utils import (
#     NeighborEmbedding,
#     CosineCutoff,
#     OptimizedDistance,
#     rbf_class_mapping,
#     act_class_mapping,
#     scatter,
#     EquivariantLayerNorm,
# )

# from models.modules.conditioning import adaLN, DropPath, DyT
# from models.modules.clsnodes import ClassNodes
# from models.modules.harmonic_emb import FourierPointEncoding

# class CondEquivMultiHeadAttention(nn.Module):
#     """Equivariant multi-head attention layer."""

#     def __init__(
#         self,
#         hidden_channels,
#         num_rbf,
#         distance_influence,
#         num_heads,
#         activation,
#         attn_activation,
#         cutoff_lower,
#         cutoff_upper,
#         vector_cutoff=False,
#         dtype=torch.float32,
#         p_droppath=0.1,
#     ):
#         super().__init__()
#         assert hidden_channels % num_heads == 0, (
#             f"The number of hidden channels ({hidden_channels}) "
#             f"must be evenly divisible by the number of "
#             f"attention heads ({num_heads})"
#         )

#         self.distance_influence = distance_influence
#         self.num_heads = num_heads
#         self.hidden_channels = hidden_channels
#         self.head_dim = hidden_channels // num_heads
#         # self.layernorm = nn.LayerNorm(hidden_channels, dtype=dtype)
#         self.act = activation()
#         self.attn_activation = act_class_mapping[attn_activation]()
#         self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

#         self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
#         self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
#         self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)
#         self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3, dtype=dtype)

#         self.vec_proj = nn.Linear(
#             hidden_channels, hidden_channels * 3, bias=False, dtype=dtype
#         )

#         self.dk_proj = None
#         if distance_influence in ["keys", "both"]:
#             self.dk_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)

#         self.dv_proj = None
#         if distance_influence in ["values", "both"]:
#             self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3, dtype=dtype)
#         self.vector_cutoff = vector_cutoff

#         #### MODIFICATIONS FOR SCD ####
#         self.conditional_layernorm = adaLN(dim=hidden_channels)
#         self.droppath = DropPath(drop_prob=p_droppath) if p_droppath > 0.0 else nn.Identity()
#         #### MODIFICATIONS FOR SCD ####

#         self.reset_parameters()

#     def reset_parameters(self):
#         # self.layernorm.reset_parameters()
#         nn.init.xavier_uniform_(self.q_proj.weight)
#         self.q_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.k_proj.weight)
#         self.k_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.v_proj.weight)
#         self.v_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.o_proj.weight)
#         self.o_proj.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.vec_proj.weight)
#         if self.dk_proj:
#             nn.init.xavier_uniform_(self.dk_proj.weight)
#             self.dk_proj.bias.data.fill_(0)
#         if self.dv_proj:
#             nn.init.xavier_uniform_(self.dv_proj.weight)
#             self.dv_proj.bias.data.fill_(0)

#     def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, cond):
#         #### MODIFICATIONS FOR SCD ####
#         # x = self.layernorm(x) # TODO replace with adaLN
#         x, gate_x = self.conditional_layernorm(x, cond) # apply conditional layernorm 
#         #### MODIFICATIONS FOR SCD ####

#         q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
#         k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
#         v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

#         vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
#         vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
#         vec_dot = (vec1 * vec2).sum(dim=1)

#         dk = (
#             self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
#             if self.dk_proj is not None
#             else None
#         )
#         dv = (
#             self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
#             if self.dv_proj is not None
#             else None
#         )
#         x, vec = self.propagate(
#             edge_index,
#             q=q,
#             k=k,
#             v=v,
#             vec=vec,
#             dk=dk,
#             dv=dv,
#             r_ij=r_ij,
#             d_ij=d_ij,
#             dim_size=None,
#         )
#         x = x.reshape(-1, self.hidden_channels)
#         vec = vec.reshape(-1, 3, self.hidden_channels)

#         o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
#         dx = vec_dot * o2 + o3
#         dvec = vec3 * o1.unsqueeze(1) + vec

#         #### MODIFICATIONS FOR SCD ####
#         dx = self.droppath(dx*gate_x) 
#         #### MODIFICATIONS FOR SCD ####

#         return dx, dvec

#     def propagate(
#         self,
#         edge_index: Tensor,
#         q: Tensor,
#         k: Tensor,
#         v: Tensor,
#         vec: Tensor,
#         dk: Optional[Tensor],
#         dv: Optional[Tensor],
#         r_ij: Tensor,
#         d_ij: Tensor,
#         dim_size: Optional[int],
#     ) -> Tuple[Tensor, Tensor]:
#         q_i = q.index_select(0, edge_index[1])
#         k_j = k.index_select(0, edge_index[0])
#         v_j = v.index_select(0, edge_index[0])
#         vec_j = vec.index_select(0, edge_index[0])
#         x, vec = self.message(q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij)
#         return self.aggregate((x, vec), edge_index[1], dim_size=dim_size)

#     def message(
#         self,
#         q_i: Tensor,
#         k_j: Tensor,
#         v_j: Tensor,
#         vec_j: Tensor,
#         dk: Optional[Tensor],
#         dv: Optional[Tensor],
#         r_ij: Tensor,
#         d_ij: Tensor,
#     ) -> Tuple[Tensor, Tensor]:
#         # attention mechanism
#         if dk is None:
#             attn = (q_i * k_j).sum(dim=-1)
#         else:
#             attn = (q_i * k_j * dk).sum(dim=-1)

#         # attention activation function
#         cutoff = self.cutoff(r_ij).unsqueeze(1)
#         attn = self.attn_activation(attn)

#         # The original ET architecture only weights the attention with the cutoff function,
#         # this causes a discontinuity in the energy at the cutoff, since the bias of the dv_proj
#         # layer might be non-zero.
#         # This option makes it so that both the scalar and vector features are weighted with the cutoff.
#         if self.vector_cutoff:
#             v_j = v_j * cutoff.unsqueeze(2)
#         else:
#             attn = attn * cutoff
#         # value pathway
#         if dv is not None:
#             v_j = v_j * dv
#         x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

#         # update scalar features
#         x = x * attn.unsqueeze(2)
#         # update vector features
#         vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
#             2
#         ).unsqueeze(3)
#         return x, vec

#     def aggregate(
#         self,
#         features: Tuple[torch.Tensor, torch.Tensor],
#         index: torch.Tensor,
#         dim_size: Optional[int],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         x, vec = features
#         x = scatter(x, index, dim=0, dim_size=dim_size)
#         vec = scatter(vec, index, dim=0, dim_size=dim_size)
#         return x, vec

#     def update(
#         self, inputs: Tuple[torch.Tensor, torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         return inputs


# class ConditionalET(nn.Module):
#     """Equivariant Transformer architecture.
    
#     Based on: Equivariant Transformers for Neural Network based Molecular Potentials
#     P. Tholke and G. de Fabritiis. ICLR 2022.

#     This function optionally supports periodic boundary conditions with arbitrary triclinic boxes.
#     For a given cutoff, :math:`r_c`, the box vectors :math:`\\vec{a},\\vec{b},\\vec{c}` must satisfy
#     certain requirements:

#     .. math::

#       \\begin{align*}
#       a_y = a_z = b_z &= 0 \\\\
#       a_x, b_y, c_z &\\geq 2 r_c \\\\
#       a_x &\\geq 2  b_x \\\\
#       a_x &\\geq 2  c_x \\\\
#       b_y &\\geq 2  c_y
#       \\end{align*}

#     These requirements correspond to a particular rotation of the system and reduced form of the vectors, as well as the requirement that the cutoff be no larger than half the box width.

#     Args:
#         hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
#         num_layers (int, optional): The number of attention layers. (default: :obj:`6`)
#         num_rbf (int, optional): The number of radial basis functions :math:`\\mu`. (default: :obj:`50`)
#         rbf_type (string, optional): The type of radial basis function to use. (default: :obj:`"expnorm"`)
#         trainable_rbf (bool, optional): Whether to train RBF parameters with backpropagation. (default: :obj:`True`)
#         activation (string, optional): The type of activation function to use. (default: :obj:`"silu"`)
#         attn_activation (string, optional): The type of activation function to use inside the attention mechanism. (default: :obj:`"silu"`)
#         neighbor_embedding (bool, optional): Whether to perform an initial neighbor embedding step. (default: :obj:`True`)
#         num_heads (int, optional): Number of attention heads. (default: :obj:`8`)
#         distance_influence (string, optional): Where distance information is used inside the attention mechanism. (default: :obj:`"both"`)
#         cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions. (default: :obj:`0.0`)
#         cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions. (default: :obj:`5.0`)
#         max_z (int, optional): Maximum atomic number. Used for initializing embeddings. (default: :obj:`100`)
#         max_num_neighbors (int, optional): Maximum number of neighbors to return for a given node/atom when constructing the molecular graph during forward passes. (default: :obj:`32`)
#         box_vecs (Tensor, optional): The vectors defining the periodic box. This must have shape `(3, 3)`, where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`. If this is omitted, periodic boundary conditions are not applied. (default: :obj:`None`)
#         vector_cutoff (bool, optional): Whether to apply the cutoff to the vector features. This prevents the energy from being discontinuous at the cutoff, but may hinder training. (default: :obj:`False`)
#         check_errors (bool, optional): Whether to check for errors in the distance module. (default: :obj:`True`)
#         dtype (torch.dtype, optional): Data type of the model. (default: :obj:`torch.float32`)
#     """

#     def __init__(
#         self,
#         hidden_channels=128,
#         num_layers=6,
#         num_rbf=50,
#         rbf_type="expnorm",
#         trainable_rbf=True,
#         activation="silu",
#         attn_activation="silu",
#         neighbor_embedding=True,
#         num_heads=8,
#         distance_influence="both",
#         cutoff_lower=0.0,
#         cutoff_upper=5.0,
#         max_z=100,
#         max_num_neighbors=32,
#         check_errors=True,
#         box_vecs=None,
#         vector_cutoff=True,
#         dtype=torch.float32,

#         layernorm_on_vec= False,
#         p_droppath=0.1,  # probability of dropping input conditioning
#         p_dropcond=0.2,  # probability of dropping input conditioning and replacing with mask token
        
#         use_cls_token = False,
#         norm_cls = True,
#         global_pos_emb = False,

#     ):
#         super().__init__()

#         assert distance_influence in ["keys", "values", "both", "none"]
#         assert rbf_type in rbf_class_mapping, (
#             f'Unknown RBF type "{rbf_type}". '
#             f'Choose from {", ".join(rbf_class_mapping.keys())}.'
#         )
#         assert activation in act_class_mapping, (
#             f'Unknown activation function "{activation}". '
#             f'Choose from {", ".join(act_class_mapping.keys())}.'
#         )
#         assert attn_activation in act_class_mapping, (
#             f'Unknown attention activation function "{attn_activation}". '
#             f'Choose from {", ".join(act_class_mapping.keys())}.'
#         )

#         self.hidden_channels = hidden_channels
#         self.num_layers = num_layers
#         self.num_rbf = num_rbf
#         self.rbf_type = rbf_type
#         self.trainable_rbf = trainable_rbf
#         self.activation = activation
#         self.attn_activation = attn_activation
#         self.neighbor_embedding = neighbor_embedding
#         self.num_heads = num_heads
#         self.distance_influence = distance_influence
#         self.cutoff_lower = cutoff_lower
#         self.cutoff_upper = cutoff_upper
#         self.max_z = max_z
#         self.dtype = dtype

#         act_class = act_class_mapping[activation]

#         self.embedding = nn.Embedding(self.max_z, hidden_channels, dtype=dtype)

#         self.distance = OptimizedDistance(
#             cutoff_lower,
#             cutoff_upper,
#             max_num_pairs=-max_num_neighbors,
#             return_vecs=True,
#             loop=True,
#             box=box_vecs,
#             long_edge_index=True,
#             check_errors=check_errors,
#         )
#         self.distance_expansion = rbf_class_mapping[rbf_type](
#             cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
#         )
#         self.neighbor_embedding_module = (
#             NeighborEmbedding(
#                 hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z, dtype
#             )
#             if neighbor_embedding
#             else None
#         )

#         self.attention_layers = nn.ModuleList()
#         for _ in range(num_layers):
#             layer = CondEquivMultiHeadAttention(
#                 hidden_channels,
#                 num_rbf,
#                 distance_influence,
#                 num_heads,
#                 act_class,
#                 attn_activation,
#                 cutoff_lower,
#                 cutoff_upper,
#                 vector_cutoff,
#                 dtype,
#                 p_droppath=p_droppath,
#             )
#             self.attention_layers.append(layer)

#         self.out_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
#         # self.out_norm = DyT(hidden_channels)
#         self.vec_norm = EquivariantLayerNorm(hidden_channels, dtype=dtype) if layernorm_on_vec else nn.Identity()
#         self.layernorm_on_vec = layernorm_on_vec

#         #### MODIFICATIONS FOR SCD ####
#         self.mask_token = nn.Parameter(torch.randn(1, hidden_channels)) # create a token to represent no conditioning
#         nn.init.normal_(self.mask_token.data, mean=0.0, std=0.02)
#         self.p_cond_dropout = p_dropcond # probability of dropping input conditioning and replacing with mask token
        
#         # self.clip_value = 100.0  # used to improve stability of torchnet architecture

#         #set clip value to inifinity
#         self.clip_value = float("inf") # FIXME: temprorary
#         #### MODIFICATIONS FOR SCD ####

#         ### CLS TOKEN MODIFICATION ###
#         self.cls_nodes = None
#         if use_cls_token:
#             self.cls_nodes = ClassNodes(
#                 node_dim=hidden_channels,
#                 edge_dim=num_rbf,
#                 init_edge_weight=1.0,
#                 init_alpha=1.0,
#                 normalize_cls=norm_cls,  # normalize cls by num nodes
#             )
#         ### CLS TOKEN MODIFICATION ###

#         ### global positional embedding ###
#         # a non-equiv positional encoding
#         if global_pos_emb:
#             print("-----USING GLOBAL POSITIONAL EMBEDDING-----")
#             self.global_pos_emb = FourierPointEncoding(hidden_channels)
#         else:
#             self.global_pos_emb = None
#         ### global positional embedding ###

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.embedding.reset_parameters()
#         self.distance_expansion.reset_parameters()
#         if self.neighbor_embedding_module is not None:
#             self.neighbor_embedding_module.reset_parameters()
#         for attn in self.attention_layers:
#             attn.reset_parameters()
#         self.out_norm.reset_parameters()
#         if self.layernorm_on_vec:
#             self.vec_norm.reset_parameters()

#     def forward(
#         self,
#         z: Tensor,
#         pos: Tensor,
#         batch: Tensor,
#         box: Optional[Tensor] = None,
#         q: Optional[Tensor] = None,
#         s: Optional[Tensor] = None,
#         cond : Optional[Tensor] = None,
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         """
#         Forward pass of the Equivariant Transformer.

#         Args:
#             z (Tensor): Atomic numbers of shape (N,)
#             pos (Tensor): Atomic positions of shape (N, 3)
#             batch (Tensor): Batch indices of shape (N,)
#             box (Tensor, optional): Box vectors for periodic boundary conditions of shape (3, 3)
#             q (Tensor, optional): Atomic charges of shape (N,)
#             s (Tensor, optional): Atomic spins of shape (N,)

#         Returns:
#             Tuple containing:
#                 - x (Tensor): Scalar features of shape (N, hidden_channels)
#                 - vec (Tensor): Vector features of shape (N, 3, hidden_channels)
#                 - z (Tensor): Atomic numbers (unchanged)
#                 - pos (Tensor): Atomic positions (unchanged)
#                 - batch (Tensor): Batch indices (unchanged)
#         """
#         x = self.embedding(z)

#         edge_index, edge_weight, edge_vec = self.distance(pos, batch, box)
#         # This assert must be here to convince TorchScript that edge_vec is not None
#         # If you remove it TorchScript will complain down below that you cannot use an Optional[Tensor]
#         assert (
#             edge_vec is not None
#         ), "Distance module did not return directional information"

#         edge_attr = self.distance_expansion(edge_weight)
#         mask = edge_index[0] != edge_index[1]
#         norm = torch.norm(edge_vec[mask], dim=1, keepdim=True)
#         eps = 1e-8
#         edge_vec[mask] = edge_vec[mask] / (norm + eps)

#         if self.neighbor_embedding_module is not None:
#             x = self.neighbor_embedding_module(z, x, edge_index, edge_weight, edge_attr)

#         ### test global pos embeddings ###
#         if self.global_pos_emb is not None:
#             #add global positional emb
#             x += self.global_pos_emb(pos)
#         ### test global pos embeddings ###

#         ### CLS TOKEN MODIFICATION ###
#         if self.cls_nodes is not None:
#             #add cls tokens
#             cls_mask, x, edge_index, edge_weight, edge_attr, edge_vec = self.cls_nodes(
#                 batch, pos, x, edge_index, edge_weight, edge_attr, edge_vec
#             )
#         ### CLS TOKEN MODIFICATION ###

#         vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

#         #### MODIFICATIONS FOR SCD ####
#         _num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
#         if cond is None:
#             # c = self.mask_token.expand(_num_graphs, -1)
#             c = self.mask_token.repeat(_num_graphs, 1)
#         else:
#             c = cond
#             if self.training:
#                 drop_mask = torch.rand(c.size(0), device=c.device) < self.p_cond_dropout
#                 drop_mask = drop_mask.float().unsqueeze(1) # (N, 1)
#                 c = c * (1 - drop_mask) + self.mask_token.clamp(min=-4, max=4) * drop_mask # replace with mask token
        
#         if self.cls_nodes is not None:
#             c = torch.cat([c[batch], c], dim=0) # expand to number of nodes + cls nodes
#         else:
#             c = c[batch] # expand to number of nodes 

#         #clip inputs for stability
#         clip_scale = self.clip_value #100.0  #8.0
#         c = c.clamp(min=-clip_scale, max=clip_scale)  # clamp to [-clip_scale, clip_scale]
#         #### MODIFICATIONS FOR SCD ####

#         for attn in self.attention_layers:
#             #clip x and vec for stability
#             x = x.clamp(min=-clip_scale, max=clip_scale)
#             vec = vec.clamp(min=-clip_scale, max=clip_scale)

#             dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec, cond=c)
            
#             #### clip dx and dv for stability ###
#             dx = dx.clamp(min=-clip_scale, max=clip_scale)
#             dvec = dvec.clamp(min=-clip_scale, max=clip_scale)

#             ### CLS TOKEN MODIFICATION ###
#             if self.cls_nodes is not None:
#                 dx, dvec = self.cls_nodes.scale_cls(dx, dvec, batch_idx=batch, cls_mask=cls_mask)
#             ### CLS TOKEN MODIFICATION ###
            
#             x = x + dx
#             vec = vec + dvec

#         x = self.out_norm(x)
#         vec = self.vec_norm(vec)

#         ### CLS TOKEN MODIFICATION ###
#         out_cls = None
#         if self.cls_nodes is not None:
#             # pop cls tokens off
#             x, vec, x_cls, vec_cls = self.cls_nodes.pop(x, vec, cls_mask)
#             out_cls ={
#                 'x': x_cls,
#                 'vec': vec_cls
#             }
#         ### CLS TOKEN MODIFICATION ###

#         return x, vec, z, pos, batch, out_cls

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"hidden_channels={self.hidden_channels}, "
#             f"num_layers={self.num_layers}, "
#             f"num_rbf={self.num_rbf}, "
#             f"rbf_type={self.rbf_type}, "
#             f"trainable_rbf={self.trainable_rbf}, "
#             f"activation={self.activation}, "
#             f"attn_activation={self.attn_activation}, "
#             f"neighbor_embedding={self.neighbor_embedding}, "
#             f"num_heads={self.num_heads}, "
#             f"distance_influence={self.distance_influence}, "
#             f"cutoff_lower={self.cutoff_lower}, "
#             f"cutoff_upper={self.cutoff_upper}), "
#             f"dtype={self.dtype}"
#         ) 