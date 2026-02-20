# """
# Wrapper class for Equivariant Transformer that outputs both invariant and equivariant node embeddings.
# """

# import torch
# from torch import nn, Tensor
# from typing import Optional, Tuple, Dict, List
# import torch.nn.functional as F

# from .output_modules import EquivariantScalar , EquivariantVector
# # from .utils import scatter
# from torch.autograd import grad

# from . import priors

# from .cond_et import ConditionalET
# from .equivariant_transformer import EquivariantTransformer
# from models.modules.conditioning import ProjHead
# from models.modules.clsnodes import CLSHead

# import re
# import warnings

# import yaml



# class BaseModel(nn.Module):
#     """
#     base class for ET and derivative models
#     """
    
#     def __init__(
#         self,
#         emb_dim: int,
#         derivative=False,
#         activation: str = "silu",
#         aggregation: str = "sum",
#         dtype: torch.dtype = torch.float32,
#         prior_model : Optional[nn.Module] = None,
#         mean=None,
#         std=None,
#         pred_cls = False, # use inv cls token for y prediction
#         emb_cls = False, # use inv cls token for embedding
#     ):
#         super().__init__()

#         # self.dtype = dtype
#         # self.emb_dim = emb_dim
#         # self.max_z = max_z
#         # self.cutoff_upper = cutoff_upper
#         # self.aggregation = aggregation
#         self.derivative = derivative
        
#         self.pred_cls = pred_cls
#         self.emb_cls = emb_cls

#         # self.prior_model = prior_model
#         if isinstance(prior_model, priors.BasePrior):
#             prior_model = [prior_model]
#         self.prior_model = (
#             None
#             if prior_model is None
#             else torch.nn.ModuleList(prior_model).to(dtype=dtype)
#         )

#         mean = torch.scalar_tensor(0) if mean is None else mean
#         self.register_buffer("mean", mean)
#         std = torch.scalar_tensor(1) if std is None else std
#         self.register_buffer("std", std)
        
#         self.noise_head = EquivariantVector(
#             hidden_channels=emb_dim,
#             activation="silu",
#             dtype=dtype,
#             reduce_op=aggregation
#         )
#         self.denoise=True # allow user to turn off denoising manually if needed

#         #TODO
#         self.noise_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        
#         if emb_cls:
#             self.embedding_head = CLSHead(emb_dim, emb_dim)
#         else:
#             self.embedding_head = ProjHead(
#                 emb_dim=emb_dim,
#                 agg=aggregation,
#             )

#         if pred_cls:
#             self.scalar_head = CLSHead(emb_dim, 1)

#         else:
#             self.scalar_head = EquivariantScalar(
#                 hidden_channels=emb_dim,
#                 activation=activation,
#                 dtype=dtype,
#             )
        
#         # self.stress_head = None
#         # self.stress_normalizer = None
#         self.stress_head = CLSHead(emb_dim, 3) # output is [G,3,3]
#         self.stress_normalizer = AccumulatedNormalization(accumulator_shape=(3, 3))
    
#     def reset_head(self):
#         # self.noise_head.reset_parameters()
#         # self.embedding_head.reset_parameters()
#         self.scalar_head.reset_parameters()
#         print('--- SCALAR HEAD RESET ---')

#     def reset_embeddings(self):
#         self.rep_model.embedding.reset_parameters()
#         print('--- EMBEDDINGS RESET ---')
    
#     def reset_norms(self, norm_type='DyT'):
#         #norm type must be 'DyT' or 'LN'
#         if isinstance(norm_type, bool):
#             if norm_type:
#                 norm_type = 'DyT' # set default type
#             else:
#                 return # if false, do nothing

#         for attn in self.rep_model.attention_layers:
#             attn.conditional_layernorm.reset_parameters(norm_type=norm_type)
#         print(f'--- CONDITIONAL LAYER NORMS RESET TO {norm_type} ---')

#     def get_layer_groups(self):

#         '''
#         returns a list of tuples, where each tuple contains the name and parameter of a layer group.
#         layer groups are defined by depth in the model, e.g. embedding layers, main layers, head layers.
#         '''
#         ### parse out layer groups so that a layer wise learning rate can be applied
#         self.embedding_layer =[
#             'rep_model.embedding',
#             'rep_model.distance',
#             'rep_model.neighbor',
#             ]
        
#         self.main_layers = [
#             'rep_model.attention_'
#             ]
        
#         self.head_layer = [
#             'rep_model.out',
#             'noise_head',
#             'embedding_head',
#             'scalar_head'
#             ]
        
#         emb_params = []
#         head_params = []
        
#         for name, param in self.named_parameters():
#             if name.startswith(tuple(self.embedding_layer)):
#                 emb_params.append((name, param))
#             elif name.startswith(tuple(self.main_layers)):
#                 continue
#             elif name.startswith(tuple(self.head_layer)):
#                 head_params.append((name, param))
#             else:
#                 raise ValueError(f"Parameter {name} not assigned to any layer group!")

#         layer_groups = [emb_params]
#         for layer in self.rep_model.attention_layers:
#             layer_groups.append([(n,p) for n, p in layer.named_parameters() if p.requires_grad])
#         layer_groups.append(head_params)

#         return layer_groups
    
#     def forward(
#         self,
#         z: Tensor,
#         pos: Tensor,
#         batch: Optional[Tensor] = None,
#         box: Optional[Tensor] = None,
#         q: Optional[Tensor] = None,
#         s: Optional[Tensor] = None,
#         cond: Optional[Tensor] = None,
#         extra_args: Optional[Dict[str, Tensor]] = None,
#     ) -> dict:
#         """

#         Compute the output of the model.

#         This function optionally supports periodic boundary conditions with
#         arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
#         certain requirements:

#         .. code:: python

#            a[1] = a[2] = b[2] = 0
#            a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
#            a[0] >= 2*b[0]
#            a[0] >= 2*c[0]
#            b[1] >= 2*c[1]


#         These requirements correspond to a particular rotation of the system and
#         reduced form of the vectors, as well as the requirement that the cutoff be
#         no larger than half the box width.

        
#         Args:
#             z (Tensor): Atomic numbers of the atoms in the molecule. Shape: (N,).
#             pos (Tensor): Atomic positions in the molecule. Shape: (N, 3).
#             batch (Tensor, optional): Batch indices for the atoms in the molecule. Shape: (N,).
#             box (Tensor, optional): Box vectors. Shape (3, 3).
#             The vectors defining the periodic box.  This must have shape `(3, 3)`,
#             where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
#             If this is omitted, periodic boundary conditions are not applied.
#             q (Tensor, optional): Atomic charges in the molecule. Shape: (N,).
#             s (Tensor, optional): Atomic spins in the molecule. Shape: (N,).
#             extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.

#         return dict: A dictionary containing the following keys:
#             - "noise_pred": Predicted noise from the noise head.
#             - "mol_emb": Molecular embeddings from the embedding head.
#             - "y": Final output after applying the output model.
#             - "force": Predicted forces if `derivative` is True.
            
#         """
#         batch = torch.zeros_like(z) if batch is None else batch
#         if self.derivative:
#             pos.requires_grad_(True)

#         if cond is None:
#             rep_out = self.rep_model(
#                         z=z, pos=pos, batch=batch, box=box, q=q, s=s
#                     )
#         else:
#             rep_out = self.rep_model(
#                     z=z, pos=pos, batch=batch, box=box, q=q, s=s, cond=cond
#                 )
            
#         # x, v, z, pos, batch, CLS = rep_out
#         x, v, z, pos, batch = rep_out[0:5]
#         CLS = None
#         if len(rep_out) > 5:
#             CLS = rep_out[5]
        
#         output_dict = {}
        
#         # predict noise
#         noise_pred = None
#         if self.denoise:
#             noise_pred = self.noise_head.pre_reduce(x, v, z, pos, batch)
#         output_dict["noise_pred"] = noise_pred

#         stress_pred = None
#         if self.stress_head is not None and CLS is not None:
#             stress_pred = self.stress_head(CLS['vec'])
#         output_dict["stress_pred"] = stress_pred

#         #Compute embeddings
#         if self.emb_cls:
#             output_dict['mol_emb'] = self.embedding_head(CLS['x'])
#         else:
#             output_dict['mol_emb'] = self.embedding_head(x, batch)

#         ####### predict y
#         if self.pred_cls:
#             y = self.scalar_head(CLS['x'])
#             if self.std is not None:
#                 y = y * self.std
#             if self.mean is not None:
#                 y = y + self.mean
#         else:
#             #TEMPORARY TESTING
#             # if output_dict['mol_emb'] is not None:
#             #     emb_scale = output_dict['mol_emb'] #.mean(dim=-1, keepdim=True) # [B,1]
#             #     # x = x * emb_scale[batch] # expand to node dim
#             #     x = (x + emb_scale[batch].detach()) #/2 # expand to node dim

#                 # gate x
#                 # x = emb_scale[batch]*F.sigmoid(x)

#             # apply the output network
#             x = self.scalar_head.pre_reduce(x, v, z, pos, batch)

#             #TEMPORARY TESTING
#             # if output_dict['mol_emb'] is not None:
#             #     emb_scale = output_dict['mol_emb'].mean(dim=-1, keepdim=True) # [B,1]
#             #     # x = x * emb_scale[batch] # expand to node dim
#             #     x = x + emb_scale[batch] # expand to node dim

#             # scale by data standard deviation
#             if self.std is not None:
#                 x = x * self.std

#             # apply atom-wise prior model
#             if self.prior_model is not None:
#                 for prior in self.prior_model:
#                     x = prior.pre_reduce(x, z, pos, batch, extra_args)

#             # aggregate atoms
#             x = self.scalar_head.reduce(x, batch)

#             # shift by data mean
#             if self.mean is not None:
#                 x = x + self.mean

#             # apply output model after reduction
#             y = self.scalar_head.post_reduce(x)
#         ########

#         # apply molecular-wise prior model
#         if self.prior_model is not None:
#             for prior in self.prior_model:
#                 y = prior.post_reduce(y, z, pos, batch, box, extra_args)
        
#         output_dict["y"] = y
#         if self.derivative:
#             grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
#             dy = grad(
#                 [y],
#                 [pos],
#                 grad_outputs=grad_outputs,
#                 create_graph=self.training,
#                 retain_graph=self.training,
#             )[0]
#             assert dy is not None, "Autograd returned None for the force prediction."
#             output_dict["dy"] = -dy
           
#         return output_dict

# class ET(BaseModel):
#     def __init__(
#         self,
#         emb_dim = 256,
#         num_layers=8,
#         num_heads=8,
#         num_rbf=64,
#         rbf_type="expnorm",
#         trainable_rbf=True,

#         neighbor_embedding=True,
#         max_num_neighbors=32,
#         distance_influence="both",
#         cutoff_lower=0.0,
#         cutoff_upper=5.0,
#         max_z=118,
#         layernorm_on_vec=False,

#         use_cls_token=False,
#         norm_cls = True,
#         global_pos_emb=False,

#         check_errors=True,
#         box_vecs=None,
#         vector_cutoff=True,

#         dtype=torch.float32,
#         derivative=False,
#         activation: str = "silu",
#         aggregation: str = "sum",
#         mean=None,
#         std=None,
    
#         **kwargs,
        
#     ):
#         super().__init__(
#             emb_dim=emb_dim,
#             derivative=derivative,
#             activation=activation,
#             aggregation=aggregation,
#             dtype=dtype,
#             mean=mean,
#             std=std,
#             **kwargs,
#         )
#         if use_cls_token: print('-----USING CLS TOKENS-----')

#         self.rep_model = EquivariantTransformer(
#             hidden_channels=emb_dim,
#             num_layers=num_layers,
#             num_rbf=num_rbf,
#             rbf_type=rbf_type,
#             trainable_rbf=trainable_rbf,
#             activation=activation,
#             attn_activation=activation,
#             neighbor_embedding=neighbor_embedding,
#             num_heads=num_heads,
#             distance_influence=distance_influence,
#             cutoff_lower=cutoff_lower,
#             cutoff_upper=cutoff_upper,
#             max_z=max_z,
#             max_num_neighbors=max_num_neighbors,
#             check_errors=check_errors,
#             box_vecs=box_vecs,
#             vector_cutoff=vector_cutoff,
#             layernorm_on_vec=layernorm_on_vec,
#             dtype=dtype,

#             use_cls_token=use_cls_token,
#             norm_cls=norm_cls,
#             global_pos_emb=global_pos_emb,  # Use global positional embedding if specified
#             )
        
# class CET(BaseModel):
#     #TODO: add CLS optionality 
#     def __init__(
#         self,
#         emb_dim = 256,
#         num_layers=8,
#         num_heads=8,
#         num_rbf=64,
#         rbf_type="expnorm",
#         trainable_rbf=True,

#         neighbor_embedding=True,
#         max_num_neighbors=32,
#         distance_influence="both",
#         cutoff_lower=0.0,
#         cutoff_upper=5.0,
#         max_z=118,
#         layernorm_on_vec=False,

#         check_errors=True,
#         box_vecs=None,
#         vector_cutoff=True,

#         p_droppath=0.1,
#         p_dropcond=0.2,

#         use_cls_token=False,
#         norm_cls = True,
#         global_pos_emb=False,

#         dtype=torch.float32,
#         derivative=False,
#         activation: str = "silu",
#         aggregation: str = "sum",
#         mean=None,
#         std=None,
#         **kwargs,
#     ):
#         super().__init__(
#             emb_dim=emb_dim,
#             derivative=derivative,
#             activation=activation,
#             aggregation=aggregation,
#             dtype=dtype,
#             mean=mean,
#             std=std,
#             **kwargs,
#         )
#         self.rep_model = ConditionalET(
#             hidden_channels=emb_dim,
#             num_layers=num_layers,
#             num_rbf=num_rbf,
#             rbf_type=rbf_type,
#             trainable_rbf=trainable_rbf,
#             activation=activation,
#             attn_activation=activation,
#             neighbor_embedding=neighbor_embedding,
#             num_heads=num_heads,
#             distance_influence=distance_influence,
#             cutoff_lower=cutoff_lower,
#             cutoff_upper=cutoff_upper,
#             max_z=max_z,
#             max_num_neighbors=max_num_neighbors,
#             check_errors=check_errors,
#             box_vecs=box_vecs,
#             vector_cutoff=vector_cutoff,
#             layernorm_on_vec=layernorm_on_vec,
#             dtype=dtype,
#             p_droppath=p_droppath,
#             p_dropcond=p_dropcond,

#             use_cls_token=use_cls_token,
#             norm_cls=norm_cls,
#             global_pos_emb=global_pos_emb,  
#             )


# #TODO: move this to conditional modules
# class AccumulatedNormalization(nn.Module):
#     """Running normalization of a tensor."""
#     def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
#         super().__init__()

#         self._epsilon = epsilon
#         self._acc_shape = accumulator_shape
#         self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
#         self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
#         self.register_buffer("acc_count", torch.zeros((1,)))
#         self.register_buffer("num_accumulations", torch.zeros((1,)))

#     def reset(self):
#         self.acc_sum = torch.zeros( self._acc_shape )
#         self.acc_squared_sum = torch.zeros(self._acc_shape )
#         self.acc_count = torch.zeros((1,))
#         self.num_accumulations = torch.zeros((1,))


#     def update_statistics(self, batch: torch.Tensor):
#         batch_size = batch.shape[0]
#         self.acc_sum += batch.sum(dim=0)
#         self.acc_squared_sum += batch.pow(2).sum(dim=0)
#         self.acc_count += batch_size
#         self.num_accumulations += 1

#     @property
#     def acc_count_safe(self):
#         return self.acc_count.clamp(min=1)

#     @property
#     def mean(self):
#         return self.acc_sum / self.acc_count_safe

#     @property
#     def std(self):
#         var = (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
#         std = torch.sqrt(var.clamp(min=self._epsilon)).clamp(min=self._epsilon)
#         return std

#     def forward(self, batch: torch.Tensor, update=False):
#         if self.training and update:
#             self.update_statistics(batch)
#         return ((batch - self.mean) / (self.std + 1e-8))
    
#     def inverse(self, batch : torch.Tensor):
#         return batch*(self.std + 1e-8) + self.mean

#     def __repr__(self):
#         # Print out all self variables
#         attrs = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]
#         attr_strs = []
#         for attr in attrs:
#             try:
#                 value = getattr(self, attr)
#                 attr_strs.append(f"{attr}: {value}")
#             except Exception as e:
#                 attr_strs.append(f"{attr}: <error retrieving value: {e}>")
#         return f"{self.__class__.__name__}(\n  " + "\n  ".join(attr_strs) + "\n)"





# # def create_model(args, prior_model=None, mean=None, std=None):
# #     model_config_file = args.get("model_config", None)

# #     #load config yaml and parse into a dictionary
# #     if model_config_file is not None:
# #         with open(model_config_file, "r") as f:
# #             model_config = yaml.safe_load(f)
# #         args.update(model_config)
# #     else:
# #         #raise error
# #         raise ValueError(f"Model config file not found: {model_config_file}")

# #     model_config['mean'] = mean
# #     model_config['std'] = std
# #     model_config['prior_model'] = prior_model
# #     #pop off the model architecture key: 'model'
# #     model_arc = model_config.pop("model", None)
# #     assert model_arc is not None, "Model architecture not specified. please include a 'model' input"
# #     model_class = globals().get(model_arc, None)
# #     assert model_class is not None, f"Model class \"{model_arc}\" not found. Please check the model architecture name."
# #     model = model_class(**model_config)

# #     # print('################')
# #     # # print(model)
# #     # print('---------------')
# #     # print(model_config)

# #     # exit()
    
# #     # model_class = getattr()
# #     # getattr(output_modules, output_prefix + args["output_model"])(
# #     #     args["embedding_dimension"], args["activation"]
# #     # print('##################')
# #     # print(model)
# #     # # print(model_config)
# #     # print('##################')
# #     # exit()

# #     # shared_args = dict(
# #     #     hidden_channels=args["embedding_dimension"],
# #     #     num_layers=args["num_layers"],
# #     #     num_rbf=args["num_rbf"],
# #     #     rbf_type=args["rbf_type"],
# #     #     trainable_rbf=args["trainable_rbf"],
# #     #     activation=args["activation"],
# #     #     neighbor_embedding=args["neighbor_embedding"],
# #     #     cutoff_lower=args["cutoff_lower"],
# #     #     cutoff_upper=args["cutoff_upper"],
# #     #     max_z=args["max_z"],
# #     #     max_num_neighbors=args["max_num_neighbors"],
# #     #     droppath = args['model_droppath'],
# #     #     dropcond = args['model_dropcond'],
# #     # )

# #     # model = CET(
# #     #     emb_dim = args["embedding_dimension"],
# #     #     num_layers = args["num_layers"],
# #     #     num_heads = args["num_heads"],
# #     #     derivative = args['derivative'],
# #     #     max_z = args["max_z"],
# #     #     cutoff_upper = args["cutoff_upper"],
# #     #     activation = args["activation"],
# #     #     aggregation = args["aggregation"],
# #     #     dtype = args["dtype"],
# #     #     prior_model = prior_model,
# #     #     mean=mean,
# #     #     std=std,
# #     #     p_droppath=args['model_droppath'],
# #     #     p_dropcond=args['model_dropcond'],
# #     # )

# #     return model


# # def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
# #     ckpt = torch.load(filepath, map_location="cpu")
# #     if args is None:
# #         args = ckpt["hyper_parameters"]

# #     for key, value in kwargs.items():
# #         if not key in args:
# #             warnings.warn(f'Unknown hyperparameter: {key}={value}')
# #         args[key] = value

# #     model = create_model(args)

# #     state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
# #     loading_return = model.load_state_dict(state_dict, strict=False)
    
# #     if len(loading_return.unexpected_keys) > 0:
# #         # Should only happen if not applying denoising during fine-tuning.
# #         print(f"WARNING: Unexpected keys in state_dict:\n{loading_return.unexpected_keys}")
# #         # print(loading_return.unexpected_keys)
# #         # assert all(("output_model_noise" in k or "pos_normalizer" in k) for k in loading_return.unexpected_keys)
# #     assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

# #     if mean:
# #         model.mean = mean
# #     if std:
# #         model.std = std

# #         print('--- MODEL SUCESSFULLY LOADED ---')
# #     return model.to(device)