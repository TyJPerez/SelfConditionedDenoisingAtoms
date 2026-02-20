"""
Modified from fairchem core graph compute utils

"""

from __future__ import annotations

import torch

# from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc, radius_graph_pbc_v2
from models.ET_models.graph_utils.radius_graph_pbc import (
    radius_graph_pbc,
    radius_graph_pbc_v2
)

def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


# TODO: compiling internal graph gen is not supported right now
@torch.compiler.disable()
def generate_graph(
    data: dict,  # this is still a torch geometric batch object currently, turn this into a dict
    cutoff: float,
    max_neighbors: int,
    enforce_max_neighbors_strictly: bool,
    radius_pbc_version: int,
    pbc: torch.Tensor,
    self_loops: bool = False,
) -> dict:
    """Generate a graph representation from atomic structure data.

    Args:
        data (dict): A dictionary containing a batch of molecular structures.
            It should have the following keys:
                - 'pos' (torch.Tensor): Positions of the atoms.
                - 'cell' (torch.Tensor): Cell vectors of the molecular structures.
                - 'natoms' (torch.Tensor): Number of atoms in each molecular structure.
        cutoff (float): The maximum distance between atoms to consider them as neighbors.
        max_neighbors (int): The maximum number of neighbors to consider for each atom.
        enforce_max_neighbors_strictly (bool): Whether to strictly enforce the maximum number of neighbors.
        radius_pbc_version: the version of radius_pbc impl
        pbc (list[bool]): The periodic boundary conditions in 3 dimensions, defaults to [True,True,True] for 3D pbc

        self_loops (bool, optional): Whether to include self-loops in the graph. Defaults to False.
    Returns:
        dict: A dictionary containing the generated graph with the following keys:
            - 'edge_index' (torch.Tensor): Indices of the edges in the graph.
            - 'edge_distance' (torch.Tensor): Distances between the atoms connected by the edges.
            - 'edge_distance_vec' (torch.Tensor): Vectors representing the distances between the atoms connected by the edges.
            - 'cell_offsets' (torch.Tensor): Offsets of the cell vectors for each edge.
            - 'offset_distances' (torch.Tensor): Distances between the atoms connected by the edges, including the cell offsets.
            - 'neighbors' (torch.Tensor): Number of neighbors for each atom.
    
    NOTES:
        atoms outside of cell are not considered for neighbor search - cell must enclose all atoms
    
    """

    if radius_pbc_version == 1:
        radius_graph_pbc_fn = radius_graph_pbc
    elif radius_pbc_version == 2:
        radius_graph_pbc_fn = radius_graph_pbc_v2
    else:
        raise ValueError(f"Invalid radius_pbc version {radius_pbc_version}")

    (
        edge_index_per_system,
        cell_offsets_per_system,
        neighbors_per_system,
    ) = list(
        zip(
            *[
                radius_graph_pbc_fn(
                    data[idx],  # loop over the batches?
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                    pbc=pbc[idx],
                )
                for idx in range(len(data))
            ]
        )
    )

    # atom indexs in the edge_index need to be offset
    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
    atom_index_offset[0] = 0
    edge_index = torch.hstack(
        [
            edge_index_per_system[idx] + atom_index_offset[idx]
            for idx in range(len(data))
        ]
    )
    cell_offsets = torch.vstack(cell_offsets_per_system)
    neighbors = torch.hstack(neighbors_per_system)

    out = get_pbc_distances(
        data.pos,
        edge_index,
        data.cell,
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )

    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offset_distances = out["offsets"]
    distance_vec = out["distance_vec"]

    # Add self-loops if requested
    if self_loops:
        num_atoms = data.pos.size(0)
        self_loop_index = torch.arange(num_atoms, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        
        # Concatenate self-loops to existing edges
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        
        # Add zero distances and vectors for self-loops
        zero_distances = torch.zeros(num_atoms, device=edge_dist.device)
        zero_vectors = torch.zeros(num_atoms, 3, device=distance_vec.device)
        zero_offsets = torch.zeros(num_atoms, 3, device=cell_offset_distances.device)
        
        edge_dist = torch.cat([edge_dist, zero_distances])
        distance_vec = torch.cat([distance_vec, zero_vectors])
        cell_offset_distances = torch.cat([cell_offset_distances, zero_offsets])


    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": cell_offset_distances,
        "neighbors": neighbors,
    }



from torch_geometric.data import Data
class GraphGenerator():
    '''
    Generates periodic/non-periodic graph representations from atomic structures using specified neighbor search methods.
    
    '''
    neighbor_method_options = ['grid', 'brute']
    required_keys = ['pos', 'cell', 'natoms', 'pbc']
    def __init__(
        self,
        cutoff: float,
        max_neighbors: int,
        self_loops: bool = False,
        neighbor_method: str = 'brute', # or 'brute_force'
        enforce_max_neighbors_strictly: bool = False,
        alt_cell_key: str = "box", # alternative key to use if "cell" is not present
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.alt_cell_key = alt_cell_key
        
        if neighbor_method == 'grid':
            # use spacial grid-based neighbor search - faster for large systems
            # use for systems with > ~1000 atoms
            self.radius_pbc_version = 2 
        elif neighbor_method == 'brute':
            # basic neighbor search - faster for small systems
            # use for systems with < ~1000 atoms
            self.radius_pbc_version = 1
        else:
            raise ValueError(f"Invalid neighbor method {neighbor_method}, must be one of {self.neighbor_method_options}")

        self.self_loops = self_loops

    def __call__(self, batch) -> dict:

        #check if batch is a standard dict or torch geometric batch object
        if isinstance(batch, Data):
            pass
        elif isinstance(batch, dict):
            batch = Data.from_dict(batch)
        else:
            raise ValueError(f"Batch must be a torch geometric Data object or a dict, got {type(batch)}")
        
        #assert that batch has required keys
        if not hasattr(batch, 'cell'):
            assert hasattr(batch, self.alt_cell_key), f"Batch is missing both 'cell' and alternative cell key '{self.alt_cell_key}' for graph generation"
            # batch.cell = getattr(batch, self.alt_cell_key)
            setattr(batch, 'cell', getattr(batch, self.alt_cell_key))

        if not hasattr(batch, 'natoms'):
            # batch.natoms = torch.bincount(batch.batch)
            setattr(batch, 'natoms', torch.bincount(batch.batch))
        # if not hasattr(batch, 'pbc'):
        #     #default to all pbc false
        #     batch.pbc = torch.zeros((batch.natoms.size(0), 3), dtype=torch.bool, device=batch.pos.device)

        for key in self.required_keys:
            if not hasattr(batch, key):
                raise ValueError(f"Batch is missing required key '{key}' for graph generation")

        out = generate_graph(
            batch,
            pbc = batch.pbc,
            cutoff = self.cutoff,
            max_neighbors = self.max_neighbors,
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly,
            radius_pbc_version = self.radius_pbc_version,
            self_loops = self.self_loops,
        )
        
        return out