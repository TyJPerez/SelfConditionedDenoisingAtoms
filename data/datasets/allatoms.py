from StructureCloud.Datasets import MultiDataset
from torch.utils.data import Dataset

from data.datasets.alexmp20 import AlexMP20_dataset
from data.datasets.geom import GEOM_dataset
from data.datasets.pcqm4mv2 import PCQM4MV2_XYZ
from data.datasets.sair import SampledSAIRDataset

# from data.datasets.transforms import AddStandardKeys, RandomCellRepeats
# from torch_geometric.transforms import Compose

'''
NOTE: The all atoms dataset requires the StructureCloud library, which is not yet released.
but StructureCloud will be coming soon!
'''

class AddTag():
    'used to tag origin of data samples in multi-dataset scenarios'
    def __init__(self, tag):
        self.tag = tag
    def __call__(self, data, depth=0):
        if isinstance(data, tuple) and depth==0:
            #recursivly apply to each data object in the tuple
            out = tuple([self.__call__(d, depth=depth+1) for d in data])
            return out

        data.tag = self.tag
        return data
    
def load_all_datasets():

    '''
      Edit this function to add/remove datasets from the combined dataset.
      Each dataset will be tagged with a 'tag' attribute for identification. 
      Make sure to also add the corresponding import statements at the top of the file.
    '''

    dataset_list = []
    
    amp20 = AlexMP20_dataset(
        split="all",
        label=None,
        transform=AddTag("alexmp20")
        )
    dataset_list.append(amp20)
    print(f'amp20 loaded. Size: {len(amp20)}')

    pcq = PCQM4MV2_XYZ(
        root='tmp/pcq',
        dataset_arg=None,
        transform=AddTag("pcq")
        )
    dataset_list.append(pcq)
    print(f'pcq loaded. Size: {len(pcq)}')

    sair = SampledSAIRDataset(
        root='tmp/sair_data/',
        transform=AddTag("sair"),
        num_neighbors=30,
        p_pocket=0.5,
        random_shift=5.0
        )
    dataset_list.append(sair)
    print(f'sair loaded. Size: {len(sair)}')

    geom10 = GEOM_dataset(
        root=None,
        subsample_name='GEOM10', 
        split='drugs',
        sample=True,
        preprocess= True,
        transform=AddTag("geom10")
        )
    dataset_list.append(geom10)
    print(f'geom10 loaded. Size: {len(geom10)}, conformers: {geom10.conformer_count()}')

    return dataset_list

class AllAtomsDataset(Dataset):
    ''' 
    A simple wrapper to combine multiple datasets into one.

    args:
        root (str): placeholder for compatibility - does nothing
        dataset_arg (str): placeholder for compatibility - does nothing

        dataset_list (list of Dataset): list of datasets to combine
        data_keys (list of str): list of data keys to keep in the final data object
        transform: a transform to apply to each data object after loading

    returns:
        a data object sampled from the combined datasets
    
    '''

    def __init__(self,
                 root=None, # placeholder for compatibility
                 dataset_arg=None, # placeholder for compatibility
                 transform=None,
                 data_keys = ['pos', 'z', 'cell', 'pbc', 'natoms', 'tag', 
                              'edge_index', 'edge_distance', 'edge_distance_vec'], 
                 
                 ):
        dataset_list = load_all_datasets()
        self.data_keys = data_keys
        self.dataset = MultiDataset(dataset_list)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def conformer_count(self, only_conformer_datasets=False):
        count = 0
        for ds in self.dataset.datasets:
            if hasattr(ds, 'conformer_count'):
                count += ds.conformer_count()
            else:
                if not only_conformer_datasets:
                    count += len(ds)
        return count

    def __getitem__(self, idx):
        idx = int(idx)
        data = self.dataset[idx]
       
        #filter data keys for batching compatibility
        for k in data.keys():
            if k not in self.data_keys:
                del data[k]

        if self.transform:
            data = self.transform(data)

        return data

