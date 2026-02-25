import numpy as np
import torch
from torch_geometric.data import Data

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from tqdm import tqdm
import os
import pickle

import pandas as pd

try:
    from StructureCloud.Datasets.utils import hf_cache_location
except ImportError:
    from .transforms import hf_cache_location

from torch.utils.data import Dataset


### TODO: make preprocessing multiprocessing compatible
import multiprocessing
from time import time
from multiprocessing import Pool


def num_fmt(x):
    return torch.tensor(x, dtype=torch.float32)
def z_fmt(x):
    return torch.tensor(x, dtype=torch.long)
def get_file_name(idx):
    return f'mol_{idx}.pt'


class GEOM_dataset(Dataset):

    #Default preprocessing dump to huggingface cache location
    default_root = os.path.join(hf_cache_location(subdir='datasets'), 'geom')

    '''
    
    args:
        root (str): root directory to save/load dataset files (if preprocessed)
        dataset_arg (str): placeholder variable for compatibility. If provided, will be used as subsample_name.
        transform (callable, optional): A function/transform that takes in
            a Data object and returns a transformed version. The data object will be
            transformed before every access. (default: :obj:`None`)

        subsample_name (str = 'GEOM10'): which GEOM subsample to use, options are 'GEOM1', 'GEOM10', 'GEOM32', etc.
        split (str = 'drugs'): which data split to use, options are 'drugs', 'qm9', 'molnet'
        sample (bool = True):
            if true the length of the dataset will be the number of unique molecules, and a random conformer will be sampled on each access.
            if false the length of the dataset will be the total number of conformers, and all conformers will be indexed individually,
            in this case, accessing dataset[i] may return a conformer from any molecule in the dataset.
        preprocess (bool = True): whether to preprocess the dataset, or load the preprocessed data already done. 
            This reduces load time and provides a conformer count summary, but requires additional disk space.
        

    '''

    def __init__(self,
                 root : str = None, #'tmp/geom',
                 dataset_arg : str = None, # placeholder for compatibility
                 transform : callable = None,
                 subsample_name : str ='GEOM10', 
                 split : str ='drugs',
                 sample : bool = True,
                 preprocess : bool = True,
                 allow_multiprocessing : bool = True,
                 ):
        
        if dataset_arg is not None:
            subsample_name = dataset_arg
        
        if root is None:
            root = self.default_root

        #define numerical formatting functions
        # self.num_fmt = lambda x: torch.tensor(x, dtype=torch.float32)
        # self.z_fmt = lambda x: torch.tensor(x, dtype=torch.long)
        # #define naming convention for saving files after preprocessing
        # self.get_file_name = lambda idx: f'mol_{idx}.pt'
        
        self.subsample_name = 'GEOM1' if subsample_name is None else subsample_name
        self.dataset_name = subsample_name
        self.split = split
        self.root = root
        self.transform = transform

        self.data_root = os.path.join(root, self.dataset_name, self.split)
        self.summary_path = os.path.join(self.data_root, 'summary.csv')
        self.use_preprocessed = preprocess
        
        if self.use_preprocessed:
            #check if summary.csv exists
            if not os.path.exists(self.summary_path):
                print(f"Summary file not found at {self.summary_path}")
                print(f"Preprocessing dataset...")
                if allow_multiprocessing:
                    self._preprocess_multiprocess()
                else:
                    self.preprocess()
                assert os.path.exists(self.summary_path), "Preprocessing failed to create summary.csv"
            #load summary.csv to check if preprocessing is needed
            self.data_summary = pd.read_csv(self.summary_path)
            self.conf_count = self.data_summary['n_conformers'].tolist()
            self.data = [os.path.join(self.data_root, fn) for fn in self.data_summary['file_name'].tolist()]
        else:
            self.data_summary = None
            self.data = self.get_hf_dataset()
            self.conf_count = []

        self.mol_index = np.arange(len(self.data))
        self.conf_index = np.arange(len(self.data))
        self.sample = sample
        
        if not sample:
            assert self.data_summary is not None, "To use sample=False, preprocessed data with conformer counts is required."
            mol_index = []
            conf_index = []
            for i, n_conf in enumerate(self.conf_count):
                mol_index += [i] * n_conf
                conf_index += list(range(n_conf))
            
            self.conf_index = conf_index    
            self.mol_index = mol_index

            print(f"Total number of conformers in dataset: {len(self.conf_index)}")
    
    def _process_sample(self, sample):
        # sample = self.dataset[idx]
        pos = num_fmt(sample['position'])
        n_conformers = pos.shape[0]

        #z and smiles are saved as individual entries b/c they are the same across conformers
        z = z_fmt(sample['node_class'])[0]
        labels = sample['labels'][0]
        smiles = labels['SMILES']
        data = Data(pos=pos, z=z, smiles=smiles, n_conformers=n_conformers)
        return data
    
    def get_hf_dataset(self):
        # return iterable huggingface dataset

        huggingface_kwargs = {
            'split': self.split,
        }
        try:
            ds = load_dataset(
                f'StructureCloud/{self.dataset_name}',
                **huggingface_kwargs
                )
        except Exception as e:
            #if the repo does not have properly assigned splits, pull the folder with the split name instead
            split = self.splithuggingface_kwargs.pop('split', 'train')
            huggingface_kwargs['data_files'] = f"{split}/*.jsonl.gz"
            ds = load_dataset(
                f'StructureCloud/{self.dataset_name}',
                **huggingface_kwargs
                )
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            ds = ds[splits[0]]

        return ds

    def _mp_sample(self, idx):
        data = self._process_sample(self.data[idx])
        n_conformers = data.n_conformers
        n_atoms = data.z.shape[0]
        f_name = get_file_name(idx)

        #save data object to file
        p_data = os.path.join(self.data_root, f_name)
        torch.save(data, p_data)

        #create summary row
        row = [idx, f_name, data.smiles, n_conformers, n_atoms]

        return row

    def _preprocess_multiprocess(self):
        #TODO: test this function
        self.data = self.get_hf_dataset() 
        summary_rows = []
        summary_header = ['idx','file_name','smiles','n_conformers','n_atoms']

        os.makedirs(os.path.dirname(self.data_root), exist_ok=True)

        available_cpus = multiprocessing.cpu_count()
        num_workers = min(len(self.data), available_cpus // 2)
        num_files = len(self.data)
        print(f"Available CPUs for preprocessing: {available_cpus}")
        print(f"Using {num_workers} worker processes for preprocessing {num_files} files.")

        process_start_timer = time()

        with Pool(num_workers) as p:
            # summary_rows = p.map(self._mp_sample, range(len(self.data)))
            summary_rows = list(tqdm(
                p.imap(self._mp_sample, range(len(self.data))),
                total=len(self.data),
                desc="Multi-Processing molecules"
            ))

        #sort summary rows by idx
        summary_rows = sorted(summary_rows, key=lambda x: x[0])
        process_end_timer = time()
        print(f"Multiprocessing preprocessing completed in {process_end_timer - process_start_timer:.2f} seconds.")

        self.save_summary(summary_rows, summary_header)

    def preprocess(self):

        self.data = self.get_hf_dataset() 
        summary_rows = []
        summary_header = ['idx','file_name','smiles','n_conformers','n_atoms']

        os.makedirs(os.path.dirname(self.data_root), exist_ok=True)
        
        for idx, sample in enumerate(tqdm(self.data)):

            data = self._process_sample(sample)
            n_conformers = data.n_conformers
            n_atoms = data.z.shape[0]
            f_name = get_file_name(idx)

            row = [idx,f_name, data.smiles, n_conformers, n_atoms]
            summary_rows.append(row)

            #save data object to file
            p_data = os.path.join(self.data_root, f_name)
            torch.save(data, p_data)


        # save out summary csv
        self.save_summary(summary_rows, summary_header)
        # with open(self.summary_path, 'w') as f:
        #     f.write(','.join(summary_header) + '\n')
        #     for row in summary_rows:
        #         f.write(','.join(map(str, row)) + '\n')
        # print(f"Summary saved to {self.summary_path}")

    def save_summary(self, rows, header):
        # save out summary csv
        with open(self.summary_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(map(str, row)) + '\n')
        print(f"Summary saved to {self.summary_path}")

    def conformer_count(self):
        if len(self.conf_count) == 0:
            print('WARNING: conformer counts not available, must preprocess first. returning number of molecules instead.')
            return self.__len__()
        return np.sum(self.conf_count)

    def __len__(self):
        return len(self.mol_index)

    def __getitem__(self, idx): 

        if self.use_preprocessed:
            if self.sample:
                n_conf = self.conf_count[idx]
                conf_idx = np.random.randint(0, n_conf)
            else:
                conf_idx = self.conf_index[idx]
                idx = self.mol_index[idx]  
            conformers = torch.load(self.data[idx], weights_only=False)

        else:
            conformers = self._process_sample(self.data[idx])
            n_conf = conformers.n_conformers
            if self.sample:
                conf_idx = np.random.randint(0, n_conf)
            else:
                raise NotImplementedError("Indexing individual conformers not implemented for non-preprocessed dataset.")

        # conformers = self.data[idx]
        pos = conformers.pos[conf_idx]
        z = conformers.z
        smiles = conformers.smiles #[rand_idx]
        data = Data(pos=pos, z=z, smiles=smiles)

        if self.transform is not None:
            data = self.transform(data)

        return data
    
if __name__ == '__main__':
    #preprocessing
    test_dataset = GEOM_dataset(
        dataset_arg='GEOM1',
        preprocess=True,
        allow_multiprocessing=True
    )

    # test_dataset._preprocess_multiprocess()

    print(f"Number of molecules in dataset: {len(test_dataset.data)}")
    print(f"Number of conformers in dataset: {test_dataset.conformer_count()}")
    print('---- sample ----')
    rand_idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[rand_idx]
    print(sample)
  