"""
Wrapper class for Equivariant Transformer that outputs both invariant and equivariant node embeddings.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Dict, List
import torch.nn.functional as F

from .output_modules import EquivariantScalar , EquivariantVector
# from .utils import scatter
from torch.autograd import grad

from . import priors

# from .cond_et import ConditionalET
from .scet import ConditionalET
from .equivariant_transformer import EquivariantTransformer
from models.modules.conditioning import ProjHead2
# from models.modules.clsnodes import CLSHead

from .frad.torchmd_frad_scd import CondFrad
from .frad.torchmd_frad import FradOriginal

# import re
# import warnings
# import yaml

class BaseModel(nn.Module):
    """
    base class for ET and derivative models
    """
    
    def __init__(
        self,
        emb_dim: int,
        derivative=False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg : str = "sum",
        head_agg : str = "sum",
        dtype: torch.dtype = torch.float32,
        prior_model : Optional[nn.Module] = None,
        mean=None,
        std=None,
        # dy_mean=None,
        # dy_std=None,
        add_head_to_pred = False, # add embedding head prediction to final y prediction
        
    ):
        super().__init__()

        self.derivative = derivative
        self.add_head_to_pred = add_head_to_pred # sum head embeddings and predictions
        self.emb_dim = emb_dim

        if isinstance(prior_model, priors.BasePrior):
            prior_model = [prior_model]
        self.prior_model = (
            None
            if prior_model is None
            else torch.nn.ModuleList(prior_model).to(dtype=dtype)
        )

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        # dy_mean = torch.scalar_tensor(0) if dy_mean is None else dy_mean
        # self.register_buffer("dy_mean", dy_mean)
        # dy_std = torch.scalar_tensor(1) if dy_std is None else dy_std
        # self.register_buffer("dy_std", dy_std)

        self.noise_head = EquivariantVector(
            hidden_channels=emb_dim,
            activation="silu",
            dtype=dtype,
            reduce_op=aggregation
        )
        self.denoise=True # allow user to turn off denoising manually if needed

        #TODO
        self.noise_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.embedding_head = ProjHead2(
                emb_dim=emb_dim,
                agg=emb_agg,
            )
        
        self.scalar_head = EquivariantScalar(
            hidden_channels=emb_dim,
            activation=activation,
            dtype=dtype,
            reduce_op=head_agg,
        )
    
    def pretrain(self):
        #freeze embedings 
        self.rep_model.freeze_embeddings(True) # dont freeze embeddings for now

        #freeze parameters in scalar_head
        for param in self.scalar_head.parameters():
            param.requires_grad = False
    
    def finetune(self):
        #unfreeze embedings
        self.rep_model.freeze_embeddings(False)

        #unfreeze parameters in scalar_head
        for param in self.scalar_head.parameters():
            param.requires_grad = True
    
    def reset_head(self):
        # self.noise_head.reset_parameters()
        # self.embedding_head.reset_parameters()
        self.scalar_head.reset_parameters()
        print('--- SCALAR HEAD RESET ---')

    def reset_embeddings(self):
        self.rep_model.embedding.reset_parameters()
        print('--- EMBEDDINGS RESET ---')
    
    def reset_norms(self, norm_type='LN'):
        #norm type must be 'DyT' or 'LN'
        if isinstance(norm_type, bool):
            if norm_type:
                norm_type = 'LN' # set default type
            else:
                return # if false, do nothing

        for attn in self.rep_model.attention_layers:
            attn.conditional_ln.reset_parameters(norm_type=norm_type)
        
        for norm in self.rep_model.x_post_norm:
            if norm is not None:
                if isinstance(norm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                    norm.reset_parameters()

        for norm in self.rep_model.v_post_norm:
            if norm is not None:
                if isinstance(norm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                    norm.reset_parameters()

        print(f'--- CONDITIONAL LAYER NORMS RESET TO {norm_type} ---')

    def get_parameter_groups(self, weight_decay = 0.01, weight_decay_on_head=False):

        no_wd_flags = ['bias', 'norm','alpha']
        no_wd_modules = ['mask_token',
                        # 'scalar_head',  #TODO: test is removing weight decay from head helps or hurts
                        'prior_model', 
                        '.embedding.',
                        'neighbor_embedding_module',
                        '_norm', 
                        '.norm.', 
                        'alpha', ]
        
        if weight_decay_on_head is False:
            no_wd_modules.append('scalar_head')
        
        no_wd_params = []
        no_wd_names = []
        allow_wd_params = []
        allow_wd_names = []

        for name, param in self.named_parameters():
            if any(name.lower().endswith(flag) for flag in no_wd_flags):
                no_wd_params.append(param)
                no_wd_names.append(name)
            elif any(module in name.lower() for module in no_wd_modules):
                no_wd_params.append(param)
                no_wd_names.append(name)
            else:
                allow_wd_params.append(param)
                allow_wd_names.append(name)

        param_groups = [
            {'params' : allow_wd_params, 'weight_decay': weight_decay, 'name': f'allow_wd'},
            {'params' : no_wd_params, 'weight_decay': 0.00, 'name': f'no_wd'},
        ]

        return param_groups

    def get_layer_groups(self):

        '''
        returns a list of tuples, where each tuple contains the name and parameter of a layer group.
        layer groups are defined by depth in the model, e.g. embedding layers, main layers, head layers.
        '''
        ### parse out layer groups so that a layer wise learning rate can be applied
        self.embedding_layer =[
            'rep_model.embedding',
            'rep_model.distance',
            'rep_model.neighbor',
            ]
        
        self.main_layers = [
            'rep_model.attention_'
            ]
        
        self.head_layer = [
            'rep_model.out',
            'noise_head',
            'embedding_head',
            'scalar_head'
            ]
        
        emb_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if name.startswith(tuple(self.embedding_layer)):
                emb_params.append((name, param))
            elif name.startswith(tuple(self.main_layers)):
                continue
            elif name.startswith(tuple(self.head_layer)):
                head_params.append((name, param))
            else:
                raise ValueError(f"Parameter {name} not assigned to any layer group!")

        layer_groups = [emb_params]
        for layer in self.rep_model.attention_layers:
            layer_groups.append([(n,p) for n, p in layer.named_parameters() if p.requires_grad])
        layer_groups.append(head_params)

        return layer_groups
    
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
        **kwargs
    ) -> dict:
        """

        Compute the output of the model.

        This function optionally supports periodic boundary conditions with
        arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
        certain requirements:

        .. code:: python

           a[1] = a[2] = b[2] = 0
           a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
           a[0] >= 2*b[0]
           a[0] >= 2*c[0]
           b[1] >= 2*c[1]


        These requirements correspond to a particular rotation of the system and
        reduced form of the vectors, as well as the requirement that the cutoff be
        no larger than half the box width.

        
        Args:
            z (Tensor): Atomic numbers of the atoms in the molecule. Shape: (N,).
            pos (Tensor): Atomic positions in the molecule. Shape: (N, 3).
            batch (Tensor, optional): Batch indices for the atoms in the molecule. Shape: (N,).
            box (Tensor, optional): Box vectors. Shape (3, 3).
            The vectors defining the periodic box.  This must have shape `(3, 3)`,
            where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
            If this is omitted, periodic boundary conditions are not applied.
            q (Tensor, optional): Atomic charges in the molecule. Shape: (N,).
            s (Tensor, optional): Atomic spins in the molecule. Shape: (N,).
            extra_args (Dict[str, Tensor], optional): Extra arguments to pass to the prior model.

        return dict: A dictionary containing the following keys:
            - "noise_pred": Predicted noise from the noise head.
            - "mol_emb": Molecular embeddings from the embedding head.
            - "y": Final output after applying the output model.
            - "force": Predicted forces if `derivative` is True.
            
        """
        batch = torch.zeros_like(z) if batch is None else batch
        if self.derivative:
            pos.requires_grad_(True)

        input_args = {'z': z, 'pos': pos, 'batch': batch}
        if box is not None:
            input_args['box'] = box
        if q is not None:
            input_args['q'] = q
        if s is not None:
            input_args['s'] = s
        if cond is not None:
            input_args['cond'] = cond
        
        if kwargs.get('graph_batch', None) is not None:
            input_args['graph_batch'] = kwargs.get('graph_batch', None)

        rep_out = self.rep_model(**input_args)
        
            
        x, v, z, pos, batch = rep_out
        
        output_dict = {}
        
        # predict noise
        noise_pred = None
        if self.denoise:
            noise_pred = self.noise_head.pre_reduce(x, v, z, pos, batch)
        output_dict["noise_pred"] = noise_pred

        #Compute embeddings
        if self.add_head_to_pred:
             output_dict['mol_emb'] = self.embedding_head(x.detach(), batch)
        else:
            output_dict['mol_emb'] = self.embedding_head(x, batch)

        # output_dict['mol_emb'] = self.embedding_head(x, batch)

        ####### predict y

        #TEMPORARY TESTING
        # if output_dict['mol_emb'] is not None:
        #     emb_scale = output_dict['mol_emb'] #.mean(dim=-1, keepdim=True) # [B,1]
        #     # x = x * emb_scale[batch] # expand to node dim
        #     x = (x + emb_scale[batch].detach()) #/2 # expand to node dim

            # gate x
            # x = emb_scale[batch]*F.sigmoid(x)

        if kwargs.get("return_atom_embs", False):
            output_dict["atom_embs"] = x

        # apply the output network
        x = self.scalar_head.pre_reduce(x, v, z, pos, batch)

        if kwargs.get("atom_mask", None) is not None:
            #if an atom mask is provided, apply it to the atomwise outputs
            atom_mask = kwargs.get("atom_mask") # [N,]
            x = x * atom_mask.unsqueeze(-1) # [N, F] * [N, 1]

            ###DEBUG
            # print(f'--- APPLYING ATOM MASK IN BASE MODEL FORWARD ---')
            # exit()
 
        #check if kwargs contains return_atom_outputs
        if kwargs.get("return_atom_outputs", False):
            output_dict["atom_outputs"] = x

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply atom-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                x = prior.pre_reduce(x, z, pos, batch, extra_args)

        # aggregate atoms
        x = self.scalar_head.reduce(x, batch)

        # shift by data mean
        if self.mean is not None:
            x = x + self.mean

        # apply output model after reduction
        y = self.scalar_head.post_reduce(x)

        #######
        if self.add_head_to_pred:
            mol_pred = output_dict['mol_emb'].mean(-1) # [B,]

            if self.std is not None:
                mol_pred = mol_pred * self.std

            if self.mean is not None:
                mol_pred = mol_pred + self.mean
            y = y + mol_pred
        ########

        # apply molecular-wise prior model
        if self.prior_model is not None:
            for prior in self.prior_model:
                y = prior.post_reduce(y, z, pos, batch, box, extra_args)
        
        output_dict["y"] = y
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            dy = grad(
                [y],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            assert dy is not None, "Autograd returned None for the force prediction."
            output_dict["dy"] = -dy
           
        return output_dict

class ET(BaseModel):
    def __init__(
        self,
        emb_dim = 256,
        num_layers=8,
        num_heads=8,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=True,

        neighbor_embedding=True,
        max_num_neighbors=32,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=118,
        layernorm_on_vec=False,

        use_cls_token=False,
        norm_cls = True,
        global_pos_emb=False,

        check_errors=True,
        box_vecs=None,
        vector_cutoff=True,

        dtype=torch.float32,
        derivative=False,
        activation: str = "silu",
        aggregation: str = "sum",
        mean=None,
        std=None,

        **kwargs,
        
    ):
        super().__init__(
            emb_dim=emb_dim,
            derivative=derivative,
            activation=activation,
            aggregation=aggregation,
            dtype=dtype,
            mean=mean,
            std=std,
            **kwargs,
        )
        if use_cls_token: print('-----USING CLS TOKENS-----')

        self.rep_model = EquivariantTransformer(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            check_errors=check_errors,
            box_vecs=box_vecs,
            vector_cutoff=vector_cutoff,
            layernorm_on_vec=layernorm_on_vec,
            dtype=dtype,

            # use_cls_token=use_cls_token,
            # norm_cls=norm_cls,
            # global_pos_emb=global_pos_emb,  # Use global positional embedding if specified
            )
        
class CET(BaseModel):
    def __init__(
        self,
        emb_dim = 256,
        num_layers=8,
        num_heads=8,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=True,

        neighbor_embedding=True,
        max_num_neighbors=32,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=118,
        layernorm_on_vec=False,

        check_errors=True,
        box_vecs=None,
        vector_cutoff=True,

        p_droppath=0.1,
        p_dropcond=0.2,
        # post_norm=True,
        inv_post_norm=False,
        vec_post_norm=False,
        vec_prenorm=True,

        dtype=torch.float32,
        derivative=False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg : str = "sum",
        mean=None,
        std=None,
        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            derivative=derivative,
            activation=activation,
            aggregation=aggregation,
            emb_agg=emb_agg,
            dtype=dtype,
            mean=mean,
            std=std,
            **kwargs,
        )

        self.rep_model = ConditionalET(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            check_errors=check_errors,
            box_vecs=box_vecs,
            vector_cutoff=vector_cutoff,
            layernorm_on_vec=layernorm_on_vec,
            dtype=dtype,
            p_droppath=p_droppath,
            p_dropcond=p_dropcond,
            # post_norm=post_norm,
            inv_post_norm=inv_post_norm,
            vec_post_norm=vec_post_norm,
            vec_prenorm=vec_prenorm,
            )
        
        # if self.derivative:
        #     print('--- CET MODEL USING FORCE PREDICTION ---')
        #     self.rep_model.ignore_conditioning(True) # completely by pass conditioning layer 


class CFrad(BaseModel):
    #TODO: add CLS optionality 
    def __init__(
        self,
        emb_dim = 256,
        num_layers=8,
        num_heads=8,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=True,

        neighbor_embedding=True,
        max_num_neighbors=32,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=118,
        layernorm_on_vec=False,

        ### Frad specific ###
        seperate_noise=False,
        num_spherical=3, 
        num_radial=6, 
        envelope_exponent=5,
        int_emb_size=64,
        basis_emb_size_dist=8, 
        basis_emb_size_angle=8, 
        basis_emb_size_torsion=8,
        num_before_skip=1,
        num_after_skip=2,
        ### Frad specific ###

        p_droppath=0.1,
        p_dropcond=0.2,
        inv_post_norm=False,
        vec_prenorm=True,
        legacy=False, # to use legacy graph generation

        dtype=torch.float32,
        derivative=False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg : str = "sum",
        mean=None,
        std=None,



        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            derivative=derivative,
            activation=activation,
            aggregation=aggregation,
            emb_agg=emb_agg,
            dtype=dtype,
            mean=mean,
            std=std,
            **kwargs,
        )

        '''new inputs:

        seperate_noise=False,
        num_spherical=3, 
        num_radial=6, 
        envelope_exponent=5,
        int_emb_size=64,
        basis_emb_size_dist=8, 
        basis_emb_size_angle=8, 
        basis_emb_size_torsion=8,
        num_before_skip=1,
        num_after_skip=2,
        
        '''

        vec_norm_ = 'whitened' if layernorm_on_vec else None
        md17_flag = self.derivative
        sep_noise = False

        self.rep_model = CondFrad(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            # check_errors=check_errors,
            # box_vecs=box_vecs,
            # vector_cutoff=vector_cutoff,
            layernorm_on_vec=vec_norm_,
            md17=md17_flag,

            #### SCD MODIFICATIONS ####
            p_droppath=p_droppath,
            p_dropcond=p_dropcond,
            inv_post_norm=inv_post_norm,
            vec_prenorm=vec_prenorm,
            legacy=legacy,
            #### SCD MODIFICATIONS ####

            ### Frad specific ###
            seperate_noise=sep_noise,
            num_spherical=num_spherical,
            num_radial=num_radial,
            envelope_exponent=envelope_exponent,
            int_emb_size=int_emb_size,
            basis_emb_size_dist=basis_emb_size_dist,
            basis_emb_size_angle=basis_emb_size_angle,
            basis_emb_size_torsion=basis_emb_size_torsion,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            ### Frad specific ###

            )
        
class ETFrad(BaseModel):
    def __init__(
        self,
        emb_dim = 256,
        num_layers=8,
        num_heads=8,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=True,

        neighbor_embedding=True,
        max_num_neighbors=32,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=118,
        layernorm_on_vec=False,

        ### Frad specific ###
        seperate_noise=False,
        num_spherical=3, 
        num_radial=6, 
        envelope_exponent=5,
        int_emb_size=64,
        basis_emb_size_dist=8, 
        basis_emb_size_angle=8, 
        basis_emb_size_torsion=8,
        num_before_skip=1,
        num_after_skip=2,
        ### Frad specific ###

        p_droppath=0.1,
        p_dropcond=0.2,
        inv_post_norm=False,
        vec_prenorm=True,

        dtype=torch.float32,
        derivative=False,
        activation: str = "silu",
        aggregation: str = "sum",
        emb_agg : str = "sum",
        mean=None,
        std=None,



        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            derivative=derivative,
            activation=activation,
            aggregation=aggregation,
            emb_agg=emb_agg,
            dtype=dtype,
            mean=mean,
            std=std,
            **kwargs,
        )

        vec_norm_ = 'whitened' if layernorm_on_vec else None
        md17_flag = self.derivative
        sep_noise = False

        self.rep_model = FradOriginal(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=neighbor_embedding,
            num_heads=num_heads,
            distance_influence=distance_influence,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_num_neighbors=max_num_neighbors,
            layernorm_on_vec=vec_norm_,
            md17=md17_flag,

            ### Frad specific ###
            seperate_noise=sep_noise,
            num_spherical=num_spherical,
            num_radial=num_radial,
            envelope_exponent=envelope_exponent,
            int_emb_size=int_emb_size,
            basis_emb_size_dist=basis_emb_size_dist,
            basis_emb_size_angle=basis_emb_size_angle,
            basis_emb_size_torsion=basis_emb_size_torsion,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            ### Frad specific ###

            )


#TODO: move this to conditional modules
class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super().__init__()

        self._epsilon = epsilon
        self._acc_shape = accumulator_shape
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def reset(self):
        self.acc_sum = torch.zeros( self._acc_shape )
        self.acc_squared_sum = torch.zeros(self._acc_shape )
        self.acc_count = torch.zeros((1,))
        self.num_accumulations = torch.zeros((1,))


    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        var = (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        std = torch.sqrt(var.clamp(min=self._epsilon)).clamp(min=self._epsilon)
        return std

    def forward(self, batch: torch.Tensor, update=False):
        if self.training and update:
            self.update_statistics(batch)
        return ((batch - self.mean) / (self.std + 1e-8))
    
    def inverse(self, batch : torch.Tensor):
        return batch*(self.std + 1e-8) + self.mean

    def __repr__(self):
        # Print out all self variables
        attrs = [attr for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr))]
        attr_strs = []
        for attr in attrs:
            try:
                value = getattr(self, attr)
                attr_strs.append(f"{attr}: {value}")
            except Exception as e:
                attr_strs.append(f"{attr}: <error retrieving value: {e}>")
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(attr_strs) + "\n)"

