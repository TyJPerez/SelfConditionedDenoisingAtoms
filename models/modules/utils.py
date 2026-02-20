import torch
from torch import nn, Tensor
from typing import Optional 

def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    ### Taken from TorchMD-NET utils
    """Broadcasts src to the shape of other along the given dimension."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    ### Taken from TorchMD-NET utils
    """
    Has the signature of torch_scatter.scatter, but uses torch.scatter_reduce instead."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    operation_dict = {
        "add": "sum",
        "sum": "sum",
        "mul": "prod",
        "mean": "mean",
        "min": "amin",
        "max": "amax",
    }
    reduce_op = operation_dict[reduce]
    # take into account the dimensionality of src
    index = _broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    res = out.scatter_reduce(dim, index, src, reduce_op)
    return res