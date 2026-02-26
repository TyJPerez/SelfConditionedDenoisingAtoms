import os 
import numpy as np
import torch
import pandas as pd
import json
from torch.utils.data import Dataset

from StructureCloud.Datasets.utils import hf_download_repo

def get_LBA_data_path(task_name: str = None) -> str:
    """Download the LBA repo from Hugging Face and return the local path.

    Requires huggingface_hub. Installs are not performed here; if missing, raise a clear error.
    """
    repo_id = "Ty-Perez/LBA"
    local_path = hf_download_repo(repo_id=repo_id, verbose=False)
    return local_path

class LBADataset_base(Dataset):
    avail_splits = ['train', 'val', 'test', 'all', 'train-val']
    avail_tasks = ['seq-id-30', 'seq-id-60']
    avail_outputs = ['protein', 'pocket', 'ligand', 'neglog_aff', 'smiles']
    # neglog_aff is the negative log binding affinity label

    ''' 
    Load LBA dataset for a given task and split 
    
    Expected directory structure:
    dataset_dir/
        data/                # directory containing all .pt files
            pdb-id-0_protein.pt     # {pdb-id-0} = actual pdb id name
            pdb-id-0_pocket.pt
            pdb-id-0_ligrand.pt
            pdb-id-1_protein.pt     # three files per sample
            pdb-id-1_pocket.pt
            pdb-id-1_ligrand.pt
            ...
        summary.json         # summary file containing metadata and train/val/test splits
            rows: 
                pdb_id,ligand_smiles,neglog_aff,protein_filename,pocket_filename,ligand_filename
        seq-id-30.json          # data splits for seq-id-30 task
            {"train": [pdb_id_1, pdb_id_2, ...],
             "val": [pdb_id_3, pdb_id_4, ...],
             "test": [pdb_id_5, pdb_id_6, ...]}
        seq-id-60.json          # data splits for seq-id-60 task
            same structure as seq-id-30.json

    args:
        task_name (str): name of the task to load. must be in avail_tasks
        split (str): data split to load. must be in avail_splits
        dataset_dir (str): default = None. path to dataset directory. if None, will attempt to download from huggingface    
        outputs (list of str): default = ['protein', 'pocket', 'ligand']. 
            orderd list of outputs to load. can be any combination of ['protein', 'pocket', 'ligand'] in any order.
            'protein' files are large, so only load if needed.
            'ligand' and 'pocket' files contain binding label data
    '''

    def __init__(self,
                split,
                task_name = 'seq-id-30',
                dataset_dir=None,
                output = ['pocket', 'ligand'], 
                ):
        assert split in self.avail_splits, f"split {split} not in available splits: {self.avail_splits}"
        assert task_name in self.avail_tasks, f"task_name {task_name} not in available tasks: {self.avail_tasks}"
        
        #set outputs to load
        self.set_outputs(output)
        
        if dataset_dir is None:
            dataset_dir = get_LBA_data_path()
        
        self.dataset_dir = dataset_dir
        self._data_path = os.path.join(self.dataset_dir, 'data') # directory containing all .pt files
        self._summary_path = os.path.join(self.dataset_dir, 'summary.csv') # summary file path
        
        assert os.path.exists(self._data_path), f"dataset_dir {dataset_dir} does not exist"
        assert os.path.exists(self._summary_path), f"summary file {self._summary_path} does not exist"

        data_files = [f for f in os.listdir(self._data_path) if f.endswith('.pt')]
        assert len(data_files) > 0, f"No data files found in {dataset_dir}"

        #load summary csv file
        summary = pd.read_csv(self._summary_path)
        #convert to dict
        self.summary = summary.to_dict(orient='list')

        #collect summary data
        self.task = task_name
        self.split = split

        #collect some columns for easier access
        self.all_pdb_ids = self.summary['pdb_id']
        self.all_targets = self.summary['neglog_aff']
        self.all_smiles = self.summary['ligand_smiles']

        assert len(self.all_pdb_ids)*3 == len(data_files), f"Number of data files {len(data_files)} does not match number of pdb ids {len(self.all_pdb_ids)}"

        self.data = self.get_split(task_name, split) # collect list of pdb ids for the given split

    def get_split(self, task, split):
       # return pdb ids for a given task and split
        if split == 'all':
            return self.all_pdb_ids
        
        #collect split information 
        split_path = os.path.join(self.dataset_dir, f"{task}.json")
        assert os.path.exists(split_path), f"split file {split_path} does not exist"
        
        with open(split_path, 'r') as f:
            split_dict = json.load(f)
        
        if split == 'train-val':
            #combine train and val splits
            train_ids = split_dict['train']
            val_ids = split_dict['val']
            return train_ids + val_ids

        return split_dict[split]
    
    def set_split(self, task, split):
        self.data = self.get_split(task, split)
    
    def __len__(self):
        return len(self.data)
    
    def _load(self, filename):
        return torch.load(os.path.join(self._data_path, filename), weights_only=True)
    def load_protein(self, pdb_id):        
        return self._load(f"{pdb_id}_protein.pt")
    def load_pocket(self, pdb_id):        
        return self._load(f"{pdb_id}_pocket.pt")
    def load_ligand(self, pdb_id):        
        return self._load(f"{pdb_id}_ligand.pt")
    
    def set_outputs(self, outputs):
        for out in outputs:
            assert out in self.avail_outputs, f"output {out} not in available outputs: {self.avail_outputs}"
        self.outputs = outputs

    def __getitem__(self, idx):
        sample_name = self.data[idx]
        outputs = ()
        for out in self.outputs:
            if out == 'protein':
                data = self.load_protein(sample_name)
            elif out == 'pocket':
                data = self.load_pocket(sample_name)
            elif out == 'ligand':
                data = self.load_ligand(sample_name)
            elif out == 'neglog_aff':
                list_idx = self.all_pdb_ids.index(sample_name)
                data = self.all_targets[list_idx]
            elif out == 'smiles':
                list_idx = self.all_pdb_ids.index(sample_name)
                data = self.all_smiles[list_idx]
            else:
                raise ValueError(f"Unknown output type: {out}")
            outputs += (data,)

        #unpack single output
        if len(outputs) == 1:
            return outputs[0]
        #otherwise return tuple
        return outputs

from torch_geometric.data import Data

class LBABenchmark(LBADataset_base):
    ''' 
    LBA Benchmark Dataset class.

    loads data in predefined format for benchmarking.
    
    '''
    def __init__(self,
                root = None, # place holder for compatibility
                dataset_arg = None, # fold index 
                transform = None,
                split = 'train-val',
                task_name = 'seq-id-30',

                #experimental options
                use_residue_z = False,
                z_res_offset = 90,
                lig_only = False,
                ):
        
        if dataset_arg is not None:
            task_name = dataset_arg

        super().__init__(split=split,
                         task_name=task_name,
                         dataset_dir=None,
                         output=['pocket', 'ligand'],)
        
        self.transform = transform
        
        #experimental options
        self.use_residue_z = use_residue_z
        self.lig_only = lig_only

        self.AA_TO_INT = {
        # Standard 20
        'ALA': 0 + z_res_offset,   
        'ARG': 1 + z_res_offset,   
        'ASN': 2 + z_res_offset,   
        'ASP': 3 + z_res_offset,   
        'CYS': 4 + z_res_offset,
        'GLN': 5 + z_res_offset,   
        'GLU': 6 + z_res_offset,   
        'GLY': 7 + z_res_offset,   
        'HIS': 8 + z_res_offset,   
        'ILE': 9 + z_res_offset,
        'LEU': 10 + z_res_offset,  
        'LYS': 11 + z_res_offset,  
        'MET': 12 + z_res_offset,  
        'PHE': 13 + z_res_offset,  
        'PRO': 14 + z_res_offset,
        'SER': 15 + z_res_offset,  
        'THR': 16 + z_res_offset,  
        'TRP': 17 + z_res_offset,  
        'TYR': 18 + z_res_offset,  
        'VAL': 19 + z_res_offset,
        
        # Non-standard / modified amino acids
        'SEC': 20 + z_res_offset,  # Selenocysteine
        'PYL': 21 + z_res_offset,  # Pyrrolysine
        'ASX': 22 + z_res_offset,  # Asparagine or Aspartate (ambiguous)
        'GLX': 23 + z_res_offset,  # Glutamine or Glutamate (ambiguous)
        'XAA': 24 + z_res_offset,  # Unknown/any amino acid
        'UNK': 25 + z_res_offset,  # Unknown
        }
    
    def __getitem__(self, idx):
        pocket, ligand = super().__getitem__(idx)

        pocket_pos = pocket['pos']
        ligand_pos = ligand['pos']

        pocket_z = pocket['z']
        if self.use_residue_z:
            res_name = pocket['res_name']  # list of AA residue names
            pocket_z = torch.tensor([self.AA_TO_INT.get(res, zi) for res, zi in zip(res_name, pocket_z)], dtype=torch.long)

        
        if self.lig_only:
            ligand_mask = torch.ones_like(ligand_pos[:, 0])
            all_pos = ligand_pos
            all_z = ligand['z']
        else:
            ligand_mask = torch.cat([torch.zeros_like(pocket_pos[:, 0]), torch.ones_like(ligand_pos[:, 0])], dim=0)
            all_pos = torch.cat([pocket_pos, ligand_pos], dim=0)
            all_z = torch.cat([pocket_z, ligand['z']], dim=0)
        
        y = ligand['neglog_aff']

        Data_obj = Data(
            pos=all_pos,
            z=all_z,
            # pocket_mask=pocket_mask,
            ligand_mask=ligand_mask,
            y=y,
        )

        if self.transform is not None:
            Data_obj = self.transform(Data_obj)

        return Data_obj
    
    def get_subset(self, split):

        #return a new copy of the dataset with the new split
        subset = self.__class__(
            split=split,
            task_name=self.task,
            transform=self.transform,
        )
        return subset
    
    def get_targets(self):
        target_vals = []
        for sample_id in self.data:
            list_idx = self.all_pdb_ids.index(sample_id)
            y = self.all_targets[list_idx]
            target_vals.append(y)
        target_vals = torch.tensor(target_vals, dtype=torch.float)
        return target_vals

    def normalize(self):
        target_vals = self.get_targets()
        mean = target_vals.mean()
        std = target_vals.std()
        return mean, std

class lba_dataset(LBABenchmark):
    def __init__(self,
                 root, # place holder for compatibility
                 dataset_arg, # fold index 
                 **kwargs):
        kwargs['task_name'] = dataset_arg
        kwargs['split'] = 'train-val'
        super().__init__( **kwargs)
