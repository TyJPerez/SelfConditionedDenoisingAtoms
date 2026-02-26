import os
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json

from pymatgen.core import Structure

#TODO: write replacement class for when pymatgen is not available

from StructureCloud.Datasets.utils import hf_download_file
   

def get_matbench_data_path(task_name: str) -> str:
    ''' download and return path to matbench data for a given task name '''
    path = hf_download_file(
        repo_id = 'Ty-Perez/matbench_properties',
        filename = f'{task_name}.tar.gz',
    )
    return str(path)

class MBDataset_base(Dataset):
    avail_splits = ['train', 'test']
    avail_tasks = ['matbench_dielectric', 
                   'matbench_expt_gap', 
                   'matbench_expt_is_metal', 
                   'matbench_glass', 
                   'matbench_jdft2d', 
                   'matbench_log_gvrh', 
                   'matbench_log_kvrh', 
                   'matbench_mp_e_form', 
                   'matbench_mp_gap', 
                   'matbench_mp_is_metal', 
                   'matbench_perovskites', 
                   'matbench_phonons', 
                   'matbench_steels']
    def __init__(self,
                task_name,
                fold_idx=0,
                split='train', #train of test
                dataset_dir=None, 
                ):
        assert task_name in self.avail_tasks, f"task_name {task_name} not in available tasks: {self.avail_tasks}"
        if dataset_dir is None:
            dataset_dir = get_matbench_data_path(task_name)
        
        self.dataset_dir = dataset_dir
        assert os.path.exists(dataset_dir), f"dataset_dir {dataset_dir} does not exist"
        data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
        summary_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
        assert len(data_files) > 0, f"No data files found in {dataset_dir}"
        assert len(summary_files) == 1, f"Expected one .json file in {dataset_dir}, found {len(summary_files)}. remove all json files except summary.json"
        
        #load summary file
        summary_file = os.path.join(dataset_dir, summary_files[0])
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        #collect metadata
        self.mb_task = summary['task_name']
        self.mb_target_name = summary['target_key']
        self.metadata = summary['metadata']
        self.all_sample_ids = summary['all_ids']
        self.targets_dict = summary['targets']

        assert len(self.all_sample_ids) == len(data_files), f"Number of data files {len(data_files)} does not match number of sample ids {len(self.all_sample_ids)}"

        #parse out training folds
        fold_idx = str(int(fold_idx))
        self.all_folds = summary['training_folds']
        avail_folds = list(self.all_folds.keys())
        assert fold_idx in avail_folds, f"fold_idx {fold_idx} not in available folds: {avail_folds}"
        assert split in self.avail_splits, f"split {split} not in available splits: {self.avail_splits}"
        
        self.split = split
        self.fold_idx = fold_idx
        self.data = self.all_folds[fold_idx][split]

        #check if pymagen library is available
        self.has_pymatgen = False
        try:
            import pymatgen
            self.has_pymatgen = True
        except ImportError:
            self.has_pymatgen = False
            #print a warning
            print("Warning: pymatgen library not found. Structure objects will be returned as dictionaries.")

    def __len__(self):
        return len(self.data)

    def info(self):
        '''return basic info about dataset using metadata'''
        out = ''
        for key, value in self.metadata.items():
            if key == 'bibtex_refs':
                # pass
                continue
            out_str += f"{key}: {value}\n"
        return out
    
    def __repr__(self):
        #return basic info about dataset using metadata
        out_str = f'Name: {self.mb_task}\n'
        out_str += f"Fold {self.fold_idx}: {self.split}\n"
        out_str += f"size: {len(self.data)} / {len(self.all_sample_ids)}\n"
        return out_str

    def load_sample(self, name):
        path = os.path.join(self.dataset_dir, f"{name}.pt")
        sample_dict = torch.load(path, weights_only=True)
        target = sample_dict['target']
        structure = sample_dict['structure']
        if self.has_pymatgen:
            structure = Structure.from_dict(sample_dict['structure'])

        return structure, target

    def __getitem__(self, idx):
        sample_name = self.data[idx]
        structure, target = self.load_sample(sample_name)
        return structure, target

class MatbenchDataset(MBDataset_base):
    # Note: this requires pymatgen to work properly
    def __init__(self, transform=None, **kwargs):
        
        super().__init__(**kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        structure, target = super().__getitem__(idx)

        data = Data()
        data.pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        data.z = torch.tensor(structure.atomic_numbers, dtype=torch.long)
        data.cell = torch.tensor(structure.lattice.matrix, dtype=torch.float).reshape(1,3,3)
        data.pbc = torch.tensor(structure.lattice.pbc, dtype=torch.bool).reshape(1,3)
        data.y = torch.tensor([target], dtype=torch.float)

        if self.transform:
            data = self.transform(data)

        return data
       
class mbench_gap(MatbenchDataset):
    def __init__(self,
                 root = None, # place holder for compatibility
                 dataset_arg=0, # fold index 
                 **kwargs):
        #FIXME: tempory hack until uploaded to hf
        # manually input path to dataset directory
        kwargs['task_name'] = 'matbench_mp_gap'
        kwargs['fold_idx'] = dataset_arg
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def get_subset(self, split):
        assert split in ['train', 'val', 'test'], f"split {split} not in ['train', 'val', 'test']"
        if split == 'val':
            split = 'test'  #map val to test for matbench datasets
        kwargs = self.kwargs.copy()
        kwargs['split'] = split
        subset = self.__class__(**kwargs)
        
        return subset
    
    def get_targets(self):
        target_vals = []
        for sample_id in self.data:
            y = self.targets_dict[sample_id]
            target_vals.append(y)
        target_vals = torch.tensor(target_vals, dtype=torch.float)
        return target_vals

    def normalize(self):
        # return mean and std of targets from this split
        target_vals = self.get_targets()
        target_vals = torch.tensor(target_vals, dtype=torch.float)
        mean = target_vals.mean()
        var = target_vals.var()
        std = target_vals.std()
        return mean, std