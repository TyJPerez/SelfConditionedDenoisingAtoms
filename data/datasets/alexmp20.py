import torch
from typing import Iterable
from torch_geometric.data import Data
from torch.utils.data import Dataset
from datasets import concatenate_datasets, interleave_datasets
from datasets import load_dataset
import os
from tqdm import tqdm

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
        assert os.path.exists(hf_cache_dir), f"HuggingFace cache directory does not exist: {hf_cache_dir}"    

        hf_cache_dir = os.path.join(hf_cache_dir, subdir) # add subdir

        #check if subdir exists, if not create it
        if not os.path.exists(hf_cache_dir):
            os.makedirs(hf_cache_dir)

        # assert os.path.exists(hf_cache_dir), f"HuggingFace cache directory does not exist: {hf_cache_dir}"
        # print(f"HuggingFace cache directory: {hf_cache_dir}")
        return hf_cache_dir


class AlexMP20_dataset(Dataset):
    ''' 
    simple loader for graph batching 
    
    args:
        split : str
            dataset split to use, one of ['train', 'val', 'test', 'all']
        label : str
            target label to use, one of available labels in all_labels
        transform : callable, optional
            a function/transform that takes in a Data object and returns a transformed version. The data is transformed before every access.
        preprocess (bool) : False
            whether to preprocess and save the dataset for faster loading in the future.
            increases loading speed by 5-10x
    
    '''

    # available labels and their keys in the dataset
    all_labels = {
    'space_group' : 'space_group',
    'chemical_system' : 'chemical_system',
    'energy_above_hull' : 'energy_above_hull',
    'band_gap' : 'dft_band_gap',
    'bulk_modulus' : 'dft_bulk_modulus',
    'mag_density' : 'dft_mag_density',
    'hhi_score' : 'hhi_score',
    'ml_bulk_modulus' : 'ml_bulk_modulus',
    'id' : 'ids',
    }

    #available splits
    splits = ['train', 'val', 'test', 'all']

    # name key used for saving preprocessed dataset
    dataset_name = 'alexmp20'
    hf_path = "Ty-Perez/AMP20"

    def __init__(self, 
                 split= 'train', 
                 label = 'band_gap',
                 transform=None,
                 preprocess = False, # increases loading speed by 5-10x
                 ):
        super().__init__()
        
        assert split in self.splits, f"Split {split} not recognized. Available splits: {self.splits}"
        if label is not None:
            assert label in self.all_labels.keys(), f"Label {label} not recognized. Available labels: {self.all_labels.keys()}"
        self.preprocess = preprocess
        self.split = split

        dataset = self.load_dataset(self.dataset_name, split)

        self.dataset = dataset
        self.label = label
        self.get_label = lambda x : x[self.all_labels[self.label]]
        self.transform = transform
    
    def load_hf_ds(self, split):
        if split == 'all':
            #load and join all three datasets
            Alexmp20_ds = load_dataset(self.hf_path)
            dataset = concatenate_datasets([
                    Alexmp20_ds['train'], 
                    Alexmp20_ds['val'], 
                    Alexmp20_ds['test']
                ])
        else:
            dataset = load_dataset(self.hf_path, split=split)

        return dataset

    def load_dataset(self, dataset_name, split):
        hf_cache_dir = hf_cache_location(subdir='datasets') # select datasets subdir
        preprocess_dataset_path = os.path.join(hf_cache_dir, f'{dataset_name}_{split}.pt')
        
        if not self.preprocess:
            dataset = self.load_hf_ds(split)
            return dataset
        
        if os.path.exists(preprocess_dataset_path):
            print(f"Loading preprocessed dataset from {preprocess_dataset_path}")
            dataset = torch.load(preprocess_dataset_path, map_location='cpu', weights_only=False)
            self.preprocess = True
            return dataset
        # elif self.preprocess:
        else:
            dataset = self.preprocess_and_save(
                self.load_hf_ds(split),
                preprocess_dataset_path)

            #verify loading
            print(f"Verifying dataset loading from {preprocess_dataset_path}")
            dataset = torch.load(preprocess_dataset_path, map_location='cpu', weights_only=False)
            return dataset
        
        # else:
        #     dataset = self.load_hf_ds(split)
        #     return dataset
        
    def force_preprocess(self):
        hf_cache_dir = hf_cache_location(subdir='datasets') # select datasets subdir
        preprocess_dataset_path = os.path.join(hf_cache_dir, f'{self.dataset_name}_{self.split}.pt')
        self.preprocess_and_save(
            self.load_hf_ds(self.split),
            preprocess_dataset_path)
        
        print(f"Verifying dataset loading from {preprocess_dataset_path}")
        dataset = torch.load(preprocess_dataset_path, map_location='cpu', weights_only=False)
        self.dataset = dataset
        self.preprocess = True

    def preprocess_and_save(self, dataset : Iterable, save_path : str):
        new_data_list = []
        print(f"Preprocessing dataset and saving to {save_path}...")
        for idx in tqdm(range(len(dataset))):
            raw_data = dataset[idx]
            processed_data = self._process_sample(raw_data)
            new_data_list.append(processed_data)
        print("Saving preprocessed dataset...")
        torch.save(new_data_list, save_path)
        print(f"Preprocessed dataset saved to {save_path}")
        return new_data_list

    def _process_sample(self, raw_data):
        ####format property values

        #NOTE: datasets loaded from huggingface are subject to changes.
        # if you get an error based on keys not being found, print the keys of raw_data and update the formatting code below accordingly.

        # #print all keys in raw data
        # for key in raw_data.keys():
        #     print(f"{key} : {raw_data[key]}")
        # exit()

        #basic inputs
        raw_data['pos'] = torch.tensor(raw_data['pos'])
        # raw_data['pos'] = torch.tensor(raw_data['positions'])
        raw_data['cell'] = torch.tensor(raw_data['cell'])
        raw_data['atomic_numbers'] = torch.tensor(raw_data['atomic_numbers'])
        raw_data['pbc'] = torch.tensor([bool(x) for x in raw_data['pbc']])
        
        #numerical labels
        num_labels = ['energy_above_hull', 'dft_band_gap',
                      'dft_bulk_modulus', 'dft_mag_density', 
                      'hhi_score', 'ml_bulk_modulus']
        for key in num_labels:
            raw_data[key] = torch.tensor(raw_data[key], dtype=torch.float)

        #text labels
        str_labels = ['space_group', 'chemical_system', 'ids']
        for key in str_labels:
            raw_data[key] = str(raw_data[key])
        
        raw_data = Data(**raw_data) # convert to torch geometric data object
        
        return raw_data

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        idx = int(idx)

        raw_data = self.dataset[idx]

        if not self.preprocess:
            raw_data = self._process_sample(raw_data)
        
        #take only necessary fields of raw_data
        out_data = Data()
        out_data.pos = raw_data['pos']
        out_data.cell = raw_data['cell'].reshape(1,3,3)
        out_data.z = raw_data['atomic_numbers']
        out_data.pbc = raw_data['pbc'].reshape(1,3)
        
        if self.label is not None:
            out_data.y = self.get_label(raw_data)
            out_data.id = raw_data['ids']

        if self.transform is not None:
            out_data = self.transform(out_data)

        return out_data
    


from data.datasets.transforms import AddStandardKeys, RandomCellRepeats
from torch_geometric.transforms import Compose

class alexmp20_dataset(AlexMP20_dataset):
    def __init__(self, 
                 root: str = None, # placeholder for compatibility
                 dataset_arg=None, # placeholder for compatibility
                 transform=None,
                 split='all',
                 preprocess=False,
                #  p_rep=0.1, 
                #  rep_iters=1, 
                #  min_atoms=4, 
                 **kwargs):
        
        #check if dataset arg is not None and a viable option for split
        if dataset_arg is not None:
            if dataset_arg not in self.splits:
                pass
            else:
                split = dataset_arg
        
        super().__init__(
            split=split,
            preprocess=preprocess,
            transform=transform,
            label=None,
            **kwargs)
        
        # transforms = [
        #     AddStandardKeys(),
        #     RandomCellRepeats(
        #         p_rep=p_rep,
        #         rep_iters=rep_iters,
        #         min_atoms=min_atoms,
        #         max_reps_per_axis=2,
        #     )]
        #check if kwargs has a transform, if not set a default one
        # if 'transform' in kwargs:
        #     if kwargs['transform'] is not None:
        #         transforms.append(kwargs['transform'])

        # kwargs['transform'] = Compose(transforms)

        # super().__init__(**kwargs)