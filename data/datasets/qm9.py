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

    # use standardization and no prior model
    1 :'alpha', # long run (500k), noise_weight=0.2
    2 :'homo',
    3 :'lumo',
    4 :'gap',
    6 : 'zpve', # long run (500k), noise_weight=0.2
    11 : 'cv',

    #use atomref with no standardization
    7 : 'u0', # long run (500k) noise_weight=0.2
    8 : 'u298', # long run (500k) noise_weight=0.2
    9 : 'h298', # long run (500k) noise_weight=0.2
    10 : 'g298', # long run (500k) noise_weight=0.2

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




class QM9_ref(QM9):
    def __init__(self, root, dataset_arg=None, transform=None):
        super().__init__(root=root, transform=None, dataset_arg=dataset_arg )

        self.tfm = transform

    def __getitem__(self, idx):

        data = super().__getitem__(idx)

        if self.tfm is not None:
            data = self.tfm(data)

        return data

def compute_qm9_ref_offset(root):
    '''
    Compute element reference energies for QM9 dataset targets using fairchem library.
    Saves the reference energies to disk for later use in training.
    '''

    try:
        import fairchem.core
    except ImportError:
        raise ImportError(
            "fairchem library is required to compute element references. "
        )

    from fairchem.core.modules.normalization.element_references import fit_linear_references
    from fairchem.core.datasets import AseDBDataset

    ### for each qm9 target, compute a reference energy for each of the 5 elements used [H, C, N, O, F]
    # create a dict to hold the reference energies
    # 
    
    #compute reference energy values for only these targets
    all_targets = ['u0', 'u298', 'h298', 'g298'] #, 'alpha', 'homo', 'lumo', 'gap', 'zpve', 'cv']
    batch_size = 128
    num_batches = 100 #2000 #1020
    num_workers = 8

    output_dir = Path(root)

    print(f'Output directory: {output_dir}')
    
    #assert the path exists
    assert output_dir.exists()

    def format_y(data):
        data.y = data.y.float().squeeze()
        data.atomic_numbers = data.z
        return data

    for var in all_targets:
        print(f'Processing {var}')
        qm9_ds = QM9_ref(root, dataset_arg=var, transform=format_y)

        # print('qm9 loaded')
        # sample = qm9_ds[0]
        # print(sample)
        # exit()

        element_refs = fit_linear_references(
        targets=["y"],  # Add "forces" if needed
        dataset=qm9_ds,
        batch_size=batch_size,
        num_batches=num_batches,
        num_workers=num_workers,
        max_num_elements=118,
        driver="least_squares",  # or "ridge" for regularization
        )

        for target, references in element_refs.items():
            save_path = output_dir / f"{var}_ref.pt"
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

    # parent_directory = "/global/homes/t/tyjperez/Projects/SelfConditionedDenoising/"
    # dataset_path = os.path.join(parent_directory, "tmp/qm9")
    
    dataset_path = 'tmp/qm9'
    #assert that the dataset exists
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist. Please download the QM9 dataset first."
    
    compute_qm9_ref_offset(
        root = dataset_path
    )