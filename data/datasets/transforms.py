import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from StructureCloud.utils.augment import repeat_unit_cell
import os
try: 
    from StructureCloud.Datasets.utils import hf_cache_location
except ImportError: 
    #get huggingface cache location
    def hf_cache_location(subdir='datasets'):
        try:
            from huggingface_hub import constants
            hf_cache_dir = constants.HF_HOME
        except ImportError:
            default_cache = os.path.expanduser("~/.cache/huggingface")
            hf_cache_dir = default_cache

        hf_cache_dir = os.path.join(hf_cache_dir, subdir) # add subdir

        assert os.path.exists(hf_cache_dir), f"HuggingFace cache directory does not exist: {hf_cache_dir}"
        # print(f"HuggingFace cache directory: {hf_cache_dir}")
        return hf_cache_dir


class AddStandardKeys(BaseTransform):
    r"""Add standard keys (pbc, natoms, cell) to molecular/crystal data.
    
    This transform ensures that the data object has the required attributes
    for molecular dynamics and crystal structure analysis:
    - pbc: periodic boundary conditions (defaults to False)
    - natoms: number of atoms in the structure
    - cell: unit cell matrix (either from existing attribute or computed from bounding box)
    
    Args:
        cell_buffer (float, optional): Buffer distance to add around the structure
            when creating a bounding box cell. (default: :obj:`3.0`)
        alt_cell_key (str, optional): Alternative attribute name to look for
            existing cell information. (default: :obj:`'box'`)
        force_pbc (bool, optional): If True, force PBC to be enabled.
            If False, defaults to disabled. (default: :obj:`False`)
    """
    
    def __init__(self, cell_buffer=0.5, alt_cell_key='box', 
                #  force_pbc=False
                remove_edge_index=True,
                 ):
        self.cell_buffer = cell_buffer
        self.alt_cell_key = alt_cell_key
        self.remove_edge_index = remove_edge_index
        # self.force_pbc = force_pbc
    
    def __call__(self, data, depth=0):

        if isinstance(data, tuple) and depth==0:
            #recursivly apply to each data object in the tuple
            out = tuple([self.__call__(d, depth=depth+1) for d in data])
            return out

        # Add PBC attribute
        if not hasattr(data, 'pbc'):
            # assume non-periodic
            data.pbc = torch.zeros((1, 3), dtype=torch.bool, device=data.pos.device)
            # if self.force_pbc:
            #     data.pbc = torch.ones((1, 3), dtype=torch.bool, device=data.pos.device)
            # else:
            #     data.pbc = torch.zeros((1, 3), dtype=torch.bool, device=data.pos.device)
        else:
            # ensure correct shape
            if data.pbc.dim() == 1:
                data.pbc = data.pbc.reshape(1, 3)

        # Add natoms attribute
        if not hasattr(data, 'natoms'):
            data.natoms = torch.tensor([data.pos.size(0)], dtype=torch.long, device=data.pos.device)
        
        # Add cell attribute
        if not hasattr(data, 'cell'):
            if hasattr(data, self.alt_cell_key):
                data.cell = getattr(data, self.alt_cell_key)
            else:
                # Create bounding box cell
                min_coords = data.pos.min(dim=0).values
                max_coords = data.pos.max(dim=0).values
                lengths = (max_coords - min_coords) + self.cell_buffer
                data.cell = torch.diag(lengths).unsqueeze(0).to(data.pos.device)
                # data.cell = torch.eye(3).reshape(1,3,3)
        
        #remove edge_index if present
        if hasattr(data, 'edge_index') and self.remove_edge_index:
            del data.edge_index
        
        return data
    
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'cell_buffer={self.cell_buffer}, '
                f'alt_cell_key={self.alt_cell_key}, '
                f'force_pbc={self.force_pbc})')


class RandomCellRepeats(BaseTransform):
    r"""Randomly repeat unit cells along periodic axes to create larger supercells.
    
    This transform replicates unit cells only along crystallographic axes that are
    marked as periodic in data.pbc. Non-periodic axes are never replicated.
    
    Args:
        p_rep (float, optional): Probability of replicating along each periodic axis
            per iteration. (default: :obj:`0.2`)
        rep_iters (int, optional): Number of iterations to attempt random
            replications. (default: :obj:`2`)
        min_atoms (int, optional): Minimum number of atoms required. If the
            initial structure has fewer atoms, forced replication occurs along
            periodic axes only. (default: :obj:`0`)
        min_system_length (float, optional): Minimum length requirement for
            each cell vector along periodic axes. (default: :obj:`0.0`)
        max_reps_per_axis (int, optional): Maximum replications allowed
            per axis to prevent excessive memory usage. (default: :obj:`4`)
        cell_key (str, optional): Attribute name containing the unit cell
            matrix. (default: :obj:`'cell'`)
    """
    
    def __init__(
        self,
        p_rep=0.2,
        rep_iters=1,
        min_atoms=0,
        min_system_length=0.0,
        max_reps_per_axis=3,
        cell_key='cell',
        track_repeats=False, # add attribute data.cell_reps
    ):
        self.p_rep = p_rep
        self.rep_iters = rep_iters
        self.min_atoms = min_atoms
        self.min_system_length = min_system_length
        self.max_reps_per_axis = max_reps_per_axis
        self.cell_key = cell_key
        self.track_repeats = track_repeats
    
    def __call__(self, data, depth=0):

        if isinstance(data, tuple) and depth==0:
            #recursivly apply to each data object in the tuple
            out = tuple([self.__call__(d, depth=depth+1) for d in data])
            return out
         
        # Ensure we have required attributes
        if not hasattr(data, self.cell_key):
            raise ValueError(f"Data object missing required attribute '{self.cell_key}'")
        if not hasattr(data, 'pbc'):
            raise ValueError("Data object missing required attribute 'pbc'")
        
        # Get periodic boundary conditions
        pbc = data.pbc.squeeze()  # Shape: [3]
        periodic_axes = pbc.cpu().numpy().astype(bool)

        if self.track_repeats:
            data.cell_reps = torch.ones(3, dtype=torch.long, device=data.pos.device)
        
        # If no periodic axes, return data unchanged
        if not any(periodic_axes):
            return data
        
        cell = getattr(data, self.cell_key)
       
        
        # Determine required replications based on constraints (periodic axes only)
        required_reps = self._compute_required_reps(data, cell, periodic_axes)
        
        # Apply random replications (periodic axes only)
        final_reps = self._apply_random_reps(required_reps, periodic_axes)
        
        # Only replicate if needed
        if final_reps != [1, 1, 1]:
            data = self._replicate_cell(data, final_reps)
        
        return data
    
    def _compute_required_reps(self, data, cell, periodic_axes):
        """Compute minimum required replications based on constraints for periodic axes only."""
        reps = [1, 1, 1]
        cell_matrix = cell.squeeze()
        
        # Check minimum system length constraint (periodic axes only)
        for dim in range(3):
            if not periodic_axes[dim]:
                continue  # Skip non-periodic axes
                
            cell_vector = cell_matrix[dim]
            length = torch.norm(cell_vector).item()
            
            if length < self.min_system_length:
                required_reps = int(np.ceil(self.min_system_length / length))
                reps[dim] = min(required_reps, self.max_reps_per_axis)
        
        # Check minimum atoms constraint (distribute among periodic axes only)
        num_atoms = data.pos.size(0)
        if num_atoms < self.min_atoms:
            num_periodic_axes = sum(periodic_axes)
            if num_periodic_axes > 0:
                # Calculate current atom count after length-based replications
                current_atom_count = num_atoms * np.prod(reps)
                
                # Determine how many additional repetitions we need total
                if current_atom_count < self.min_atoms:
                    total_expansion_needed = np.ceil(self.min_atoms / current_atom_count)
                    additional_reps_needed = int(total_expansion_needed - 1)
                    
                    # Randomly distribute additional repetitions among periodic axes
                    periodic_axis_indices = [i for i in range(3) if periodic_axes[i]]
                    
                    for _ in range(additional_reps_needed):
                        # Choose a random periodic axis that hasn't hit the max
                        available_axes = [
                            axis for axis in periodic_axis_indices 
                            if reps[axis] < self.max_reps_per_axis
                        ]
                        
                        if available_axes:
                            chosen_axis = np.random.choice(available_axes)
                            reps[chosen_axis] += 1
                        else:
                            # All axes at max, break
                            break
        
        return reps
    
    def _apply_random_reps(self, initial_reps, periodic_axes):
        """Apply random replications on top of required ones (periodic axes only)."""
        reps = initial_reps.copy()
        
        # Get list of periodic axes for random selection
        periodic_axis_indices = [i for i in range(3) if periodic_axes[i]]
        
        if not periodic_axis_indices:
            return reps  # No periodic axes to replicate
        
        for _ in range(self.rep_iters):
            if torch.rand(1).item() < self.p_rep:
                # Choose random periodic axis to replicate
                axis_idx = torch.randint(0, len(periodic_axis_indices), (1,)).item()
                axis = periodic_axis_indices[axis_idx]
                
                # Only replicate if under the maximum
                if reps[axis] < self.max_reps_per_axis:
                    reps[axis] += 1
        
        return reps
    
    def _replicate_cell(self, data, reps):
        """Replicate the unit cell with given repetitions."""
        # Store replication info
        if self.track_repeats:
            data.cell_reps = torch.tensor(reps, dtype=torch.long, device=data.pos.device)
        
        # Perform replication
        new_pos, new_z, new_cell = repeat_unit_cell(
            data.pos, 
            data.z, 
            getattr(data, self.cell_key), 
            reps=tuple(reps), 
            fractional_pos=False
        )
        
        # Update data
        data.pos = new_pos
        data.z = new_z
        setattr(data, self.cell_key, new_cell)
        
        # Update natoms if it exists
        if hasattr(data, 'natoms'):
            data.natoms = torch.tensor([new_pos.size(0)], dtype=torch.long, device=new_pos.device)
        
        return data
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'p_rep={self.p_rep}, '
            f'rep_iters={self.rep_iters}, '
            f'min_atoms={self.min_atoms}, '
            f'min_system_length={self.min_system_length}, '
            f'max_reps_per_axis={self.max_reps_per_axis})'
        )


from models.ET_models.graph_utils.compute import generate_graph
from torch_geometric.data import Batch as GraphBatch

class CreateGraph(BaseTransform):
    r"""
    Generate graph edges for periodic or non-periodic systems
   
    """
    
    neighbor_method_options = ['grid', 'brute']
    required_keys = ['pos', 'cell', 'natoms', 'pbc']

    #other attributes available to output into data object
    attr_avail = ['cell_offsets', 'offset_distances', 'neighbors']

    def __init__(
        self,
        cutoff: float,
        max_neighbors: int,
        self_loops: bool = False,
        neighbor_method: str = 'brute', # or 'brute_force'
        enforce_max_neighbors_strictly: bool = False,
        alt_cell_key: str = "cell", # alternative key to use if "cell" is not present
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

    def __call__(self, data):
        if isinstance(data, list):
            #throw not implemented error
            raise NotImplementedError("Batch processing not implemented")

            batch = GraphBatch.from_data_list(data)

            out = generate_graph(
                batch,
                pbc=batch.pbc,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                self_loops=self.self_loops,
            )
            # out_list = out.to_data_list()
            batch.edge_index = out['edge_index']
            batch.edge_distance = out['edge_distance']
            batch.edge_distance_vec = out['edge_distance_vec']

            #update the batch _slice_dict to include new attributes
            edge_slice = torch.tensor([0, out['edge_index'].size(1)/2, out['edge_index'].size(1)], dtype=torch.long)
            batch._slice_dict['edge_index'] = edge_slice
            batch._slice_dict['edge_distance'] = edge_slice
            batch._slice_dict['edge_distance_vec'] = edge_slice

            #update batch _inc_dict to include new attributes
            edge_inc = torch.tensor([0, out['edge_index'].size(1)/2], dtype=torch.long)
            batch._inc_dict['edge_index'] = edge_inc
            batch._inc_dict['edge_distance'] = edge_inc
            batch._inc_dict['edge_distance_vec'] = edge_inc

            # batch._store['edge_index'] = out['edge_index']
            # batch._store['edge_distance'] = out['edge_distance'] 
            # batch._store['edge_distance_vec'] = out['edge_distance_vec']
            # print(batch.keys())
            data = batch.to_data_list()
            # print(data[0].keys())
            
            return data
        else:
            batch = GraphBatch.from_data_list([data])

            out = generate_graph(
                batch,
                pbc=batch.pbc,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
                radius_pbc_version=self.radius_pbc_version,
                self_loops=self.self_loops,
            )

            data.edge_index = out['edge_index']
            data.edge_distance = out['edge_distance']
            data.edge_distance_vec = out['edge_distance_vec']

            return data
    

class SCD_noise(BaseTransform):
    ''' 
    '''
    def __init__(
        self,
        reg_noise  : float = 0.005,
        corrupt_noise : float = 0.04,
        simple_noise : bool = False,
        cutoff: float = 5.0,
        max_neighbors: int = 32,
        self_loops: bool = True,
        neighbor_method: str = 'brute', # or 'brute_force'
        enforce_max_neighbors_strictly: bool = False,
        alt_cell_key: str = "cell", # alternative key to use if "cell" is not present
        ):

        self.reg_noise = reg_noise
        self.corrupt_noise = corrupt_noise
        self.simple_noise = simple_noise

        self.add_graph = CreateGraph(
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            self_loops=self_loops,
            neighbor_method=neighbor_method,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            alt_cell_key=alt_cell_key,
            )
    
    def __call__(self, data, depth=0):

        if self.simple_noise:
            # add noise and return a single target
            # used for either COORD or regularizing noise on periodic systems
            
            if isinstance(data, tuple) and depth==0:
                #recursivly apply to each data object in the tuple
                out = tuple([self.__call__(d, depth=depth+1) for d in data])
                return out

            if self.corrupt_noise > 0.0:
                target_noise = torch.randn_like(data.pos) * self.corrupt_noise
                data.pos += target_noise
                data.noise = target_noise
                
            data = self.add_graph(data)
            return data
        else:
            if isinstance(data, tuple):
                #assume first sample will used as conditioning input
                clean_data, corrupt_data = data
                #this is the case for pocket-conditioned ligand denoising
            else:
                # prep data for scd
                clean_data = data
                corrupt_data = data.clone()

            if self.reg_noise > 0.0:
                # Apply small regularization noise to the clean data
                clean_data.pos += torch.randn_like(clean_data.pos) * self.reg_noise
                
            # Apply noise to the noisy data
            target_noise = torch.randn_like(corrupt_data.pos) * self.corrupt_noise
            corrupt_data.pos += target_noise

            # Add graph structure to both clean and noisy data
            clean_data = self.add_graph(clean_data)
            corrupt_data = self.add_graph(corrupt_data)

            # clean_data, corrupt_data = self.add_graph([clean_data, corrupt_data])
            corrupt_data.noise = target_noise

            return clean_data, corrupt_data

class Compose(BaseTransform):
    ''' Composes multiple transforms together
    
    NOTE: use this compose for SCD_noise. 
    torch_geometric's compose automatically handles tuples of data objects, 
    and inadvertently duplicates objects going through SCD_noise

    '''
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data