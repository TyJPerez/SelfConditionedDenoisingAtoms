
import os
import tarfile
import urllib.request
from glob import glob
from pathlib import Path
import bisect
import numpy as np
from typing import Union, List
from dataclasses import dataclass
from typing import Optional


import ase.db
import torch
from torch_geometric.data import Dataset, Data

from tqdm import tqdm

#### A simple class to store retrieval information about OMol25 splits ####
@dataclass
class SplitInfo:
    """Data class to store information about OMol25 dataset splits"""
    name: str
    size: int  # Number of samples
    storage: str  # Storage size (e.g., '456GB')
    url: str  # Download URL
    
    def __repr__(self):
        return f"SplitInfo(name='{self.name}', size={self.size:,}, storage='{self.storage}')"

class OMol25Splits:
    """
    Container for OMol25 dataset split information.
    Allows querying split information by name.
    
    Usage:
        splits = OMol25Splits()
        train_4m = splits('train_4M')
        print(train_4m.url)
        print(train_4m.size)
    """
    
    BASE_URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/"
    
    def __init__(self):
        # Define all available splits
        self.splits = {
            # Training splits
            'train_all': SplitInfo(
                name='train_all',
                size=101_666_280,
                storage='456GB',
                url=f"{self.BASE_URL}train.tar.gz"
            ),
            'train_4M': SplitInfo(
                name='train_4M',
                size=3_986_754,
                storage='19GB',
                url=f"{self.BASE_URL}train_4M.tar.gz"
            ),
            'train_neutral': SplitInfo(
                name='train_neutral',
                size=34_335_828,
                storage='101GB',
                url=f"{self.BASE_URL}train_neutral.tar.gz"
            ),
            
            # Validation splits
            'val': SplitInfo(
                name='val',
                size=2_762_021,
                storage='20GB',
                url=f"{self.BASE_URL}val.tar.gz"
            ),
            'val_neutral': SplitInfo(
                name='val_neutral',
                size=27_697,
                storage='119MB',
                url=f"{self.BASE_URL}val_neutral.tar.gz"
            ),
            
            # Test splits
            'test': SplitInfo(
                name='test',
                size=2_805_046,
                storage='8GB',
                url=f"{self.BASE_URL}test.tar.gz"
            ),
        }
    
    def __call__(self, split_name: str) -> Optional[SplitInfo]:
        """
        Query split information by name.
        
        Args:
            split_name: Name of the split (case-insensitive)
            
        Returns:
            SplitInfo object or None if split not found
        """
        print(f"Querying OMol25 split: '{split_name}'")
        # return self.splits.get(split_name.lower())
        #check if split name is in splits dict
        if not split_name in self.splits:
            print(f"Split '{split_name}' not found. Available splits are: {list(self.splits.keys())}")

        return self.splits.get(split_name)
    
    def get(self, split_name: str) -> Optional[SplitInfo]:
        """Alias for __call__"""
        return self(split_name)
    
    def list_splits(self):
        """List all available splits"""
        print("Available OMol25 splits:\n")
        print(f"{'Split':<20} {'Samples':>15} {'Storage':>10}")
        print("-" * 50)
        
        # Group by category
        categories = {
            'Training': ['train_all', 'train_4M', 'train_neutral'],
            'Validation': ['val', 'val_neutral'],
            'Test': ['test']
        }
        
        for category, split_names in categories.items():
            print(f"\n{category}:")
            for name in split_names:
                split = self.splits[name]
                print(f"  {split.name:<18} {split.size:>15,} {split.storage:>10}")
    
    def __repr__(self):
        return f"OMol25Splits({len(self.splits)} splits available)"



#### A simplified dataset object extracted from fairchem library ####
class SimpleAseDBDataset:
    """
    Simplified ASE Database Dataset for OMol25.
    Connects to ASE databases and provides access to atoms objects.
    
    This is a minimal implementation extracted from fairchem's AseDBDataset,
    containing only the essential functionality needed for OMol25.
    
    Args:
        config (dict): Configuration dictionary with keys:
            - src (str): Path to ASE DB file, folder with DBs, or list of paths
            - connect_args (dict, optional): Keyword arguments for ase.db.connect()
            - select_args (dict, optional): Keyword arguments for ase.db.select() to filter data
            - a2g_args (dict, optional): Arguments for atoms-to-graph conversion:
                - r_edges (bool): Compute edges (default: False)
                - radius (float): Cutoff radius for edges (default: 6.0)
                - max_neigh (int): Maximum neighbors per atom (default: None)
                - r_energy (bool): Read energy from calculator (default: True)
                - r_forces (bool): Read forces from calculator (default: True)
                - r_stress (bool): Read stress from calculator (default: False)
        subset_size (int, optional): If provided, randomly sample this many examples from the dataset
        seed (int): Random seed for subset sampling (default: 42)
    """
    
    def __init__(self, config: dict, 
                 subset_size: int = None, 
                 seed: int = 42):
        self.config = config
        self.subset_size = subset_size
        self.seed = seed
        
        # Parse a2g_args for atoms-to-graph conversion
        a2g_args = config.get("a2g_args", {}) or {}
        self.r_edges = a2g_args.get("r_edges", False)
        self.radius = a2g_args.get("radius", 6.0)
        self.max_neigh = a2g_args.get("max_neigh", None)
        self.r_energy = a2g_args.get("r_energy", True)
        self.r_forces = a2g_args.get("r_forces", True)
        self.r_stress = a2g_args.get("r_stress", False)
        
        self._load_dataset_get_ids(config)
    
    def _load_dataset_get_ids(self, config: dict) -> List[int]:
        """Load database(s) and get list of IDs"""
        # Handle different input types for src
        if isinstance(config["src"], list):
            filepaths = []
            for path in sorted(config["src"]):
                if os.path.isdir(path):
                    filepaths.extend(sorted(glob(f'{path}/*')))
                elif os.path.isfile(path):
                    filepaths.append(path)
                else:
                    filepaths.extend(sorted(glob(path)))
        elif os.path.isfile(config["src"]):
            filepaths = [config["src"]]
        elif os.path.isdir(config["src"]):
            filepaths = sorted(glob(f'{config["src"]}/*'))
        else:
            filepaths = sorted(glob(config["src"]))
        
        # Connect to all databases
        self.dbs = []
        for path in filepaths:
            try:
                self.dbs.append(
                    self.connect_db(
                        path,
                        config.get("connect_args", {}),
                    )
                )
            except (ValueError, Exception):
                # Skip files that aren't valid databases
                pass
        
        if len(self.dbs) == 0:
            raise ValueError(f"No valid ASE databases found at: {config['src']}")
        
        self.select_args = config.get("select_args", {})
        if self.select_args is None:
            self.select_args = {}
        
        # Get IDs from each database
        self.db_ids = []
        for db in self.dbs:
            if hasattr(db, "ids") and self.select_args == {}:
                # Fast path: use pre-existing ids list
                self.db_ids.append(db.ids)
            else:
                # Slow path: query database
                self.db_ids.append([row.id for row in db.select(**self.select_args)])
        
        # Create cumulative length array for indexing
        idlens = [len(ids) for ids in self.db_ids]
        self._idlen_cumulative = [sum(idlens[:i+1]) for i in range(len(idlens))]
        
        self.num_samples = sum(idlens)
        
        # Create subset indices if requested
        if self.subset_size is not None:
            import numpy as np
            rng = np.random.RandomState(self.seed)
            if self.subset_size > self.num_samples:
                raise ValueError(f"Subset size {self.subset_size} is larger than dataset size {self.num_samples}")
            self.subset_indices = rng.choice(self.num_samples, size=self.subset_size, replace=False)
            self.subset_indices = sorted(self.subset_indices)  # Sort for better cache locality
        else:
            self.subset_indices = None
    
    @staticmethod
    def connect_db(address: Union[str, Path], connect_args: dict = None) -> ase.db.core.Database:
        """Connect to an ASE database"""
        if connect_args is None:
            connect_args = {}
        
        # Set readonly=True for LMDB databases for safety
        if "aselmdb" in str(address):
            connect_args = {**connect_args, "readonly": True}
        
        return ase.db.connect(str(address), **connect_args)
    
    def get_atoms(self, idx: int) -> ase.Atoms:
        """
        Get atoms object corresponding to datapoint idx.
        
        Args:
            idx (int): Index in dataset (or subset if subset_size was specified)
            
        Returns:
            atoms: ASE atoms object corresponding to datapoint idx
        """
        # Map subset index to original index if using subset
        if self.subset_indices is not None:
            idx = self.subset_indices[idx]
        
        # Figure out which db this index belongs to
        db_idx = bisect.bisect(self._idlen_cumulative, idx)
        
        # Extract index within that specific database
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._idlen_cumulative[db_idx - 1]
        assert el_idx >= 0
        
        # Get the atoms row and convert to atoms object
        atoms_row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()
        
        # Put additional data back into atoms.info
        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)
        
        # Store system ID if not present
        if "sid" not in atoms.info:
            atoms.info["sid"] = idx
        
        return atoms
    
    def _atoms_to_data(self, atoms: ase.Atoms, sid: str = None) -> dict:
        """
        Convert ASE Atoms object to a data dictionary.
        Adapted from AtomicData.from_ase without fairchem dependencies.
        
        Args:
            atoms: ASE Atoms object
            sid: System ID (optional)
            
        Returns:
            dict with keys: pos, atomic_numbers, cell, pbc, natoms, 
                           edge_index, cell_offsets, nedges, charge, spin, 
                           fixed, tags, energy, forces, stress (if available)
        """
        # Basic structure
        pos = torch.from_numpy(atoms.positions).float()
        atomic_numbers = torch.from_numpy(atoms.numbers).long()
        cell = torch.from_numpy(atoms.cell.array).float().unsqueeze(0)  # [1, 3, 3]
        pbc = torch.from_numpy(atoms.pbc).bool().unsqueeze(0)  # [1, 3]
        natoms = torch.tensor([len(atoms)], dtype=torch.long)
        
        # Charge and spin
        charge = torch.tensor([atoms.info.get("charge", 0)], dtype=torch.long)
        spin = torch.tensor([atoms.info.get("spin_multiplicity", 1)], dtype=torch.long)
        
        # Fixed atoms and tags
        if atoms.constraints:
            fixed = torch.zeros(len(atoms), dtype=torch.long)
            for constraint in atoms.constraints:
                if hasattr(constraint, 'index'):
                    fixed[constraint.index] = 1
        else:
            fixed = torch.zeros(len(atoms), dtype=torch.long)
        
        tags = torch.from_numpy(atoms.get_tags()).long()
        
        # Compute edges if requested
        if self.r_edges:
            edge_index, cell_offsets = self._compute_edges(atoms)
            nedges = torch.tensor([edge_index.shape[1]], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            cell_offsets = torch.zeros((0, 3), dtype=torch.float32)
            nedges = torch.tensor([0], dtype=torch.long)
        
        data_dict = {
            'pos': pos,
            'atomic_numbers': atomic_numbers,
            'cell': cell,
            'pbc': pbc,
            'natoms': natoms,
            'edge_index': edge_index,
            'cell_offsets': cell_offsets,
            'nedges': nedges,
            'charge': charge,
            'spin': spin,
            'fixed': fixed,
            'tags': tags,
            'sid': sid if sid else str(atoms.info.get("sid", "")),
        }
        
        # Extract targets from calculator if available
        if atoms.calc is not None:
            try:
                if self.r_energy:
                    energy = atoms.get_potential_energy()
                    data_dict['energy'] = torch.tensor([energy], dtype=torch.float32)
            except Exception:
                pass
            
            try:
                if self.r_forces:
                    forces = atoms.get_forces()
                    data_dict['forces'] = torch.from_numpy(forces).float()
            except Exception:
                pass
            
            try:
                if self.r_stress:
                    stress = atoms.get_stress(voigt=False)  # 3x3 tensor
                    data_dict['stress'] = torch.from_numpy(stress).float().unsqueeze(0)
            except Exception:
                pass
        
        return data_dict
    
    def _compute_edges(self, atoms: ase.Atoms):
        """
        Compute edge_index and cell_offsets for periodic systems.
        Uses pymatgen for neighbor finding.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            edge_index: [2, num_edges] tensor
            cell_offsets: [num_edges, 3] tensor
        """
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            
            struct = AseAtomsAdaptor.get_structure(atoms)
            
            # Get neighbor list with tolerance to remove self-loops
            _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
                r=self.radius, numerical_tol=1e-8, exclude_self=True
            )
            
            # Limit to max_neigh if specified
            if self.max_neigh is not None:
                _nonmax_idx = []
                for i in range(len(atoms)):
                    idx_i = (_c_index == i).nonzero()[0]
                    # Sort neighbors by distance, keep only max_neigh
                    idx_sorted = np.argsort(n_distance[idx_i])[:self.max_neigh]
                    _nonmax_idx.append(idx_i[idx_sorted])
                _nonmax_idx = np.concatenate(_nonmax_idx)
                
                _c_index = _c_index[_nonmax_idx]
                _n_index = _n_index[_nonmax_idx]
                n_distance = n_distance[_nonmax_idx]
                _offsets = _offsets[_nonmax_idx]
            
            # Stack as [neighbor_index, center_index]
            edge_index = torch.LongTensor(np.vstack((_n_index, _c_index)))
            cell_offsets = torch.FloatTensor(_offsets)
            
            # Remove edges with distance < tolerance (edge cases)
            edge_distances = torch.FloatTensor(n_distance)
            nonzero = torch.where(edge_distances >= 1e-8)[0]
            edge_index = edge_index[:, nonzero]
            cell_offsets = cell_offsets[nonzero]
            
            return edge_index, cell_offsets
            
        except ImportError:
            # Fallback: no edges if pymatgen not available
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3), dtype=torch.float32)
    
    def __len__(self):
        """Return number of samples in dataset"""
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get preprocessed data by index.
        
        Returns:
            dict with preprocessed data if a2g_args provided, otherwise ASE Atoms object
        """
        atoms = self.get_atoms(idx)
        
        # If a2g_args provided, convert to data dict
        if self.config.get("a2g_args") is not None:
            return self._atoms_to_data(atoms)
        else:
            # Return raw atoms object (backward compatible)
            return atoms
    
    def __del__(self):
        """Clean up database connections"""
        for db in self.dbs:
            if hasattr(db, "close"):
                db.close()


#### A simple class to download and sample the OMol25 S2EF datasets ####


class DownloadProgressBar(tqdm):
    """Progress bar for download with urllib"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class OMol25Dataset(Dataset):
    """
    PyTorch Geometric Dataset wrapper for OMol25.
    
    Args:
        root (str): Path to the OMol25 dataset directory (e.g., 'omol25')
        split (str): Dataset split name (e.g., 'train_4M', 'val', 'test')
        transform (callable, optional): Transform to apply to each Data object
        enable_download (bool): If True, download dataset if not found
    
    Returns:
        Data object with keys: pos, z, cell, pbc, natoms, energy, force
        Note: For test split, energy and force will be None (labels not provided)
    """
    
    def __init__(
        self, 
        root: str,
        split: str = 'train_4M',
        transform: Optional[callable] = None,
        enable_download: bool = False,
        subset_size: int = None,
        subset_seed: int = 42,
        ref_energies: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split_name = split
        self.dataset_path = self.root / split
        self.enable_download = enable_download
        self.subset_size = subset_size
        self.subset_seed = subset_seed

        #get split info 
        self.split_info = OMol25Splits()


        # Check that dataset path exists
        if not self.dataset_path.exists():
            if not enable_download:
                #get split storage info
                split_info = self.split_info(self.split_name)
                storage_size = split_info.storage if split_info else "unknown size"

                raise FileNotFoundError(
                    f"Dataset path {self.dataset_path} does not exist. "
                    f"Set enable_download=True to download the dataset."
                    f" Dataset Expected download size: {storage_size}."
                )
            else:
                print(f"Dataset not found at {self.dataset_path}")
                print(f"Downloading {split}...")
                self._download_and_extract()
        
        # Initialize the underlying AseDBDataset using SimpleAseDBDataset
        self.ase_dataset = SimpleAseDBDataset({"src": str(self.dataset_path)}, 
                                              subset_size=subset_size, seed=subset_seed)

        # Test split doesn't have labels
        self.has_labels = (self.split_name != 'test')

        
        self.ref_table = None
        self.ref_energies = ref_energies
        if ref_energies is not None:
            self.ref_table = torch.nn.Embedding(
                len(ref_energies), 1, _freeze=True, _weight=ref_energies.view(-1, 1)
                )

        self.transform = transform
    
    def _download_and_extract(self):
        """Download and extract the dataset"""
        # Get split information
        split_info = self.split_info(self.split_name)
        
        if split_info is None:
            available = list(self.split_info.splits.keys())
            raise ValueError(
                f"Split '{self.split_name}' not found. "
                f"Available splits: {available}"
            )
        
        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        # Download path
        tar_filename = f"{self.split_name}.tar.gz"
        tar_path = self.root / tar_filename
        
        # Download the file
        print(f"Downloading from: {split_info.url}")
        print(f"Size: {split_info.storage}, Samples: {split_info.size:,}")
        print(f"Saving to: {tar_path}")
        
        try:
            with DownloadProgressBar(
                unit='B', 
                unit_scale=True,
                miniters=1, 
                desc=f"Downloading {tar_filename}"
            ) as t:
                urllib.request.urlretrieve(
                    split_info.url, 
                    tar_path, 
                    reporthook=t.update_to
                )
            
            print(f"\nDownload complete!")
            
            # Extract the tar.gz file
            print(f"Extracting {tar_filename}...")
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Get list of members for progress bar
                members = tar.getmembers()
                
                # Extract with progress bar
                with tqdm(
                    total=len(members), 
                    desc=f"Extracting {tar_filename}",
                    unit="files"
                ) as pbar:
                    for member in members:
                        tar.extract(member, path=self.root)
                        pbar.update(1)
            
            print(f"Extraction complete!")
            print(f"Dataset saved to: {self.dataset_path}")
            
            # Clean up tar file
            print(f"Removing {tar_filename}...")
            tar_path.unlink()
            print("Done!")
            
        except Exception as e:
            # Clean up partial downloads
            if tar_path.exists():
                tar_path.unlink()
            if self.dataset_path.exists():
                import shutil
                shutil.rmtree(self.dataset_path)
            raise RuntimeError(f"Failed to download/extract dataset: {e}")
    
    def len(self):
        """Return the number of samples in the dataset"""
        return len(self.ase_dataset)
    
    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        """
        Get a single sample as a PyTorch Geometric Data object.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Data object with the following attributes:
                - pos: Atomic positions [num_atoms, 3]
                - z: Atomic numbers [num_atoms]
                - cell: Unit cell [3, 3]
                - pbc: Periodic boundary conditions [3]
                - natoms: Number of atoms (scalar)
                - energy: Total energy in eV (scalar) or None if not available
                - force: Atomic forces in eV/Å [num_atoms, 3] or None if not available
        """
        # Get atoms object from AseDB
        atoms = self.ase_dataset.get_atoms(idx)
        
        # Extract atomic structure
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        
        # Extract cell and PBC
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32).reshape(1, 3, 3)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool).reshape(1, 3)
        
        # Number of atoms
        natoms = torch.tensor([len(atoms)], dtype=torch.long)
        
        # NOTE: Test split doesn't have labels (no calculator attached)
        energy = None
        force = None
        if self.has_labels:
            # Extract labels (energy and forces) if available
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float32)
            force = torch.tensor(atoms.get_forces(), dtype=torch.float32)

            if self.ref_table is not None:
                # compute and subtract reference energy if provided
                ref_energy = self.ref_table(z).sum()
                energy = energy - ref_energy.squeeze(0)
    
        # Create PyG Data object
        data = Data(
            pos=pos,
            z=z,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            # energy=energy,
            # force=force,
            y = energy,
            dy = force,
        )

        if self.transform is not None:
            data = self.transform(data)

        return data


from torch_geometric.data import DataLoader

#### A dataset wrapper class to incoporate Omol25 into training pipeline ####
class omol25_S2EF(OMol25Dataset):
    def __init__(self, 
                 root: str,
                 dataset_arg=None,  
                 split='train',
                 transform: Optional[callable] = None, 
                 enable_download: bool = False,
                 subset_size: int = None,
                 subset_seed: int = 42,
                 ):
        
        #names of splits
        self.split_select = {'train': 'train_4M',
                             'val': 'val',
                             'test': 'val'} 
        
        self.root = root
        self.split = split
        self.transform = transform
        self.enable_download = enable_download

        #path to store targets after extraction
        self.targets_path = os.path.join(self.root, f'{self.split_select[split]}_targets.pt')

        #check for path to reference energies
        self.ref_eng_path = os.path.join(self.root, self.split_select['train'], f'energy_linref.pt')
        if not os.path.exists(self.ref_eng_path):
            print(f"Reference energies not found at {self.ref_eng_path}. "
                  "Please compute them using compute_omol25_references() function.")
        
        #load reference energies
        self.ref_energy = torch.load(self.ref_eng_path)['element_references']

        super().__init__(root, 
                         split=self.split_select[split], 
                         transform=transform,
                         enable_download=enable_download, 
                         subset_size=subset_size, 
                         subset_seed=subset_seed,
                         ref_energies=self.ref_energy
                         )
    
    # def get_atomref(self):
    #     return self.ref_energy

    def get_subset(self, split, size=None, seed=42):

        if split == 'val':
            size=10000 # manually define validation set as subset of 10k for quick evals
        
        subset = self.__class__(
            root=self.root,
            split=split,
            transform=self.transform,
            enable_download=self.enable_download,
            subset_size=size,
            subset_seed=seed,
            )
        return subset
    
    def get_targets(self, max_samples=None, recompute=False):
        """Return all targets as a tensor"""
        #check if targets have already been extracted
        if os.path.exists(self.targets_path) and not recompute:
            print(f"Loading targets from {self.targets_path}...")
            all_energies = torch.load(self.targets_path)
            return all_energies

        batch_size = 1024
        loader = DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=64)
        print(f"Extracting targets from dataset...")
        all_energies = []

        for batch in tqdm(loader, desc="Extracting targets"):
            all_energies.append(batch.y.squeeze(0))
            #collect norm of forces
            # all_forces_norm.append(data.dy.norm(dim=1))
            # all_force_vals.append(data.dy.flatten())
            if max_samples is not None and len(all_energies) >= max_samples:
                break
        
        all_energies = torch.cat(all_energies, dim=0)
        # all_force_vals = torch.cat(all_force_vals, dim=0)

        #save targets for future use
        torch.save((all_energies), self.targets_path)
            
        return all_energies
    
    def normalize(self):
        if self.split != 'train':
            raise ValueError("Normalization statistics should be computed on the training set only.")
        
        """Normalize targets using dataset statistics"""
        energy_vals = self.get_targets()
        energy_mean = energy_vals.mean()
        energy_std = energy_vals.std()

        #forces are already roughgly normalized,
        # force_mean = forces.mean()
        # force_std = forces.std()
        return energy_mean, energy_std
        

################# functions to compute references for OMol25 dataset #################

def compute_omol25_references(
    dataset_path: str = "omol25/train_4M",
    output_path: str = "omol25/train_4M",
    batch_size: int = 128,
    num_batches: int = 1000,
    num_workers: int = 8,
):
    """
    Compute and save element references for OMol25 dataset.
    
    Args:
        dataset_path: Path to OMol25 training dataset
        output_path: Where to save the reference files
        batch_size: Batch size for fitting
        num_batches: Number of batches to use (None = use all data)
        num_workers: DataLoader workers
    """
    try:
        import fairchem.core
    except ImportError:
        raise ImportError(
            "fairchem library is required to compute element references. "
        )

    from fairchem.core.modules.normalization.element_references import fit_linear_references
    from fairchem.core.datasets import AseDBDataset
    
    # Load training dataset
    print(f"Loading dataset from {dataset_path}...")
    train_dataset = AseDBDataset({"src": dataset_path,
                        "a2g_args": {
                            "r_energy": True,   # read energy from ASE DB
                            "r_forces": True,   # read forces if you want them
                        },
                        }
                        )
    print(f"Dataset size: {len(train_dataset):,} samples")
    
    # Fit linear references
    print(f"\nFitting element references...")
    print(f"  Using {num_batches} batches of size {batch_size}")
    print(f"  Total samples: ~{num_batches * batch_size:,}")
    
    element_refs = fit_linear_references(
        targets=["energy"],  # Add "forces" if needed
        dataset=train_dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        num_workers=num_workers,
        max_num_elements=118,
        driver="least_squares",  # or "ridge" for regularization
    )
    
    # Save references
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for target, references in element_refs.items():
        save_path = output_path / f"{target}_linref.pt"
        torch.save(references.state_dict(), save_path)
        print(f"\n{target} references saved to: {save_path}")
        
        # Print some reference values
        state = references.state_dict()
        print(f"  Shape: {state['element_references'].shape}")
        print(f"  Non-zero elements: {(state['element_references'] != 0).sum().item()}")
        
        # Show first few non-zero references (H, C, N, O typically)
        refs = state['element_references']
        for z in range(1, min(20, len(refs))):
            if refs[z] != 0:
                print(f"    Element {z}: {refs[z].item():.4f} eV")



if __name__ == "__main__":

    training_set_data_path = "tmp/omol25/train_4M"

    #assert that the dataset exists
    if not os.path.exists(training_set_data_path):
        raise FileNotFoundError(
            f"OMol25 training dataset not found at {training_set_data_path}. "
            f"Please download the dataset before running this script."
        )
    

    compute_omol25_references(
        dataset_path=training_set_data_path,
        output_path=training_set_data_path,
        batch_size=128,
        num_batches=10000,  # ~1.28M samples
        num_workers=8,
    )