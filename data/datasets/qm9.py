from pathlib import Path
import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric

from tqdm import tqdm
import os

class QM9(QM9_geometric):

    target_dict_custom ={
    0 :'mu', # use special head, no standardization 
    5 : 'R2', # use special head, no standardization 

    # use standardization (norm to unit gaussian) and no prior model
    1 :'alpha', 
    2 :'homo',
    3 :'lumo',
    4 :'gap',
    6 : 'zpve', 
    11 : 'cv',

    #use atomref with no standardization (no shift or scale)
    7 : 'u0', 
    8 : 'u298', 
    9 : 'h298', 
    10 : 'g298', 

    12 : 'u0_atom', #FIXME
    13 : 'u298_atom', #FIXME
    14 : 'h298_atom', #FIXME
    15 : 'g298_atom', #FIXME
    
}
    
    def __init__(self, root, transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            # f'properties are {", ".join(qm9_target_dict.values())}.'
            f'properties are {", ".join(QM9.target_dict_custom.values())}.'
        )

        self.label = dataset_arg
        # label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        label2idx = dict(zip(QM9.target_dict_custom.values(), QM9.target_dict_custom.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super().__init__(root, transform=transform)

        self.use_legacy_atomref = True


    def get_atomref(self, max_z=118):
        if not self.use_legacy_atomref:
            print(f' ========Using NEW atomref for QM9 target: {self.label} =======')
            #check if recomputed references exist
            ref_path = os.path.join(self.root, f'{self.label}_ref.pt')
            if os.path.exists(ref_path):
                ref_data = torch.load(ref_path)
                atomref = ref_data['element_references']

                #clip to max_z
                atomref = atomref[:max_z]

                return atomref
            else:
                print(f'Atom references for {self.label} not found at {ref_path}. Using default QM9 references.')
                assert False, 'You must compute atom references using fairchem tools. this can be done by running this file as a script. see data/datasets/qm9.py'
                return None
        else:
            #### OLD ATOMREF
            #FIXME: remove this section after testing
            ''' 
            NOTE: we recompute atomrefs using fairchem tools in the 
            function below becasue its unclear how torchgeometric has computed the default values provided.
            '''
            atomref = self.atomref(self.label_idx)
            if atomref is None:
                return None
            if atomref.size(0) != max_z:
                tmp = torch.zeros(max_z).unsqueeze(1)
                idx = min(max_z, atomref.size(0))
                tmp[:idx] = atomref[:idx]
                return tmp
            return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()

    def process(self):
        super(QM9, self).process()



# class QM9_ref(QM9):
#     def __init__(self, root, dataset_arg=None, transform=None):
#         super().__init__(root=root, transform=None, dataset_arg=dataset_arg )

#         self.tfm = transform

#     def __getitem__(self, idx):

#         data = super().__getitem__(idx)

#         if self.tfm is not None:
#             data = self.tfm(data)

#         return data

