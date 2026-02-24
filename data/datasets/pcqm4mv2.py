from typing import Optional, Callable

import os
import torch
from torch_geometric.data import InMemoryDataset
from huggingface_hub import hf_hub_download

class PCQ(InMemoryDataset):
    r"""PCQM4Mv2 dataset with 3D coordinates loaded from HuggingFace.
    
    This dataset loads pre-processed PyTorch Geometric data from the 
    HuggingFace repository 'Ty-Perez/PCQ', which contains the processed
    3D molecular structures.
    
    Args:
        root (str): Root directory where the dataset should be stored.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 dataset_arg: Optional[str] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Not needed since we download directly to processed
        return []

    @property
    def processed_file_names(self):
        return ['pcqm4mv2__xyz.pt']

    def download(self):
        """Download pre-processed data from HuggingFace repository."""
        print("Downloading processed dataset from HuggingFace...")
        
        # Download the processed file from HuggingFace
        hf_hub_download(
            repo_id="Ty-Perez/PCQ",
            filename="processed/pcqm4mv2__xyz.pt",
            repo_type="dataset",
            local_dir=self.root,
            local_dir_use_symlinks=False
        )
        
        print(f"Downloaded to {self.processed_paths[0]}")

    def process(self):
        """No processing needed - data is already processed."""
        # The data is already processed and downloaded to the correct location
        # This method is called by InMemoryDataset if processed files don't exist,
        # but since we download directly to processed/, this shouldn't be needed
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'
