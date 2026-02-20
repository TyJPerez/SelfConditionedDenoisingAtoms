"""
Output modules for the Equivariant Transformer.
Extracted from torchmd-net for standalone use.
"""

from abc import abstractmethod, ABCMeta
from typing import Optional
import torch
from torch import nn
from .utils import (
    act_class_mapping,
    # GatedEquivariantBlock,
    scatter,
    # MLP,
)
from .utils import atomic_masses
from .extensions import is_current_stream_capturing
import warnings
from warnings import warn


class MLP(nn.Module):
    r"""A simple multi-layer perceptron with a given number of layers and hidden channels.

    The simplest MLP has no hidden layers and is composed of two linear layers with a non-linear activation function in between:

    .. math::

        \text{MLP}(x) = \text{Linear}_o(\text{act}(\text{Linear}_i(x)))

    Where :math:`\text{Linear}_i` has input size :math:`\text{in_channels}` and output size :math:`\text{hidden_channels}` and :math:`\text{Linear}_o` has input size :math:`\text{hidden_channels}` and output size :math:`\text{out_channels}`.


    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        hidden_channels (int): Number of hidden features.
        activation (str): Activation function to use.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to 0.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float32.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        activation,
        num_hidden_layers=0,
        dtype=torch.float32,
    ):
        super(MLP, self).__init__()
        act_class = act_class_mapping[activation]
        self.act = act_class()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(in_channels, hidden_channels, dtype=dtype))
        self.layers.append(self.act)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels, dtype=dtype))
            self.layers.append(self.act)
        self.layers.append(nn.Linear(hidden_channels, out_channels, dtype=dtype))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.layers(x)
        return x

class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
        dtype=torch.float,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.vec2_proj = nn.Linear(
            hidden_channels, out_channels, bias=False, dtype=dtype
        )

        act_class = act_class_mapping[activation]
        self.update_net = MLP(
            in_channels=hidden_channels * 2,
            out_channels=out_channels * 2,
            hidden_channels=intermediate_channels,
            activation=activation,
            num_hidden_layers=0,
            dtype=dtype,
        )
        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        self.update_net.reset_parameters()

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        # detach zero-entries to avoid NaN gradients during force loss backpropagation
        vec1 = torch.zeros(
            vec1_buffer.size(0),
            vec1_buffer.size(2),
            device=vec1_buffer.device,
            dtype=vec1_buffer.dtype,
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        if not mask.all():
            warnings.warn(
                (
                    f"Skipping gradients for {(~mask).sum()} atoms due to vector features being zero. "
                    "This is likely due to atoms being outside the cutoff radius of any other atom. "
                    "These atoms will not interact with any other atom unless you change the cutoff."
                )
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2) # NOTE: this is equivariant restriction on expresivity

        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1) # [B, 4, d]

        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1) # [B, 4, d]
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v

class OutputHead(nn.Module):
    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        output_dim : int,
        aggregate : bool =False,
        agg_op="sum",
        activation="silu",
        dtype=torch.float,
        ):
        super().__init__()
        self.aggregate = aggregate
        self.agg_op = agg_op
        self.dim_size = 0
        # self.act = act_class_mapping[activation]()

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    input_dim,
                    hidden_dim,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype,
                ),
                GatedEquivariantBlock(
                    hidden_dim, 
                    output_dim, 
                    activation=activation,
                    scalar_activation=False, 
                    dtype=dtype
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()
    
    def reduce(self, x, batch):
        is_capturing = x.is_cuda and is_current_stream_capturing()
        if not x.is_cuda or not is_capturing:
            self.dim_size = int(batch.max().item() + 1)
        if is_capturing:
            assert (
                self.dim_size > 0
            ), "Warming up is needed before capturing the model into a CUDA graph"
            warn(
                "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                    self.dim_size
                )
            )
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.agg_op)
    
    def forward(self, x, v, batch=None):
        for layer in self.output_network:
            x, v = layer(x, v)
        
        #make sure all parameters have a gradient
        x = x + v.sum() * 0
        v = v + x.unsqueeze(1) * 0

        if self.aggregate:
            assert batch is not None, "Batch is required for aggregation"
            x = self.reduce(x, batch)
            v = self.reduce(v, batch)

        return x, v


class OutputModel(nn.Module, metaclass=ABCMeta):
    """Base class for output models.

    Derive this class to make custom output models.
    As an example, have a look at the :py:mod:`torchmdnet.output_modules.Scalar` output model.
    """

    def __init__(self, allow_prior_model, reduce_op):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def reduce(self, x, batch):
        is_capturing = x.is_cuda and is_current_stream_capturing()
        if not x.is_cuda or not is_capturing:
            self.dim_size = int(batch.max().item() + 1)
        if is_capturing:
            assert (
                self.dim_size > 0
            ), "Warming up is needed before capturing the model into a CUDA graph"
            warn(
                "CUDA graph capture will lock the batch to the current number of samples ({}). Changing this will result in a crash".format(
                    self.dim_size
                )
            )
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x

class Scalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        **kwargs,
    ):
        super(Scalar, self).__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        self.output_network = MLP(
            in_channels=hidden_channels,
            out_channels=1,
            hidden_channels=hidden_channels // 2,
            activation=activation,
            num_hidden_layers=kwargs.get("num_layers", 0),
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.output_network.reset_parameters()

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)

class EquivariantScalar(OutputModel):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        allow_prior_model=True,
        reduce_op="sum",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__(
            allow_prior_model=allow_prior_model, reduce_op=reduce_op
        )
        if kwargs.get("num_layers", 0) > 0:
            warn("num_layers is not used in EquivariantScalar")
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 1, activation=activation, dtype=dtype
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0


class EquivariantVector(EquivariantScalar):
    def __init__(
        self,
        hidden_channels,
        activation="silu",
        reduce_op="sum",
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op="sum",
            dtype=dtype,
            **kwargs,
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze() + x.sum() * 0





####################
