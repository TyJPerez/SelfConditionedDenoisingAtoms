from pathlib import Path
import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric

from tqdm import tqdm
import os

'''
NOTE: the correct QM9 dataset should have exactly 130831 molecules. 
if your version of torch_geometric has more than this, or throws and error you may have to 
edit this loader to correct for this.

this repo was originall written using torch geometric 2.6.1
other version may or may not work as expected.

'''

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

        #verify that dataset is correct length
        if len(self) != 130831:
            print(f"Warning: QM9 dataset has incorrect length {len(self)}!! Expected 130831. Please verify your version of torch_geometric and the QM9 dataset. You may need to edit this loader to correct for this.")

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

    @property
    def raw_file_names(self):
        return ['qm9_v3.pt']

    def download(self):
        from torch_geometric.data import download_url, extract_zip
        import os
        path = download_url(self.processed_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    # def process(self):
    #     super(QM9, self).process()

    def process(self) -> None:

        import sys
        from torch_geometric.utils import one_hot, scatter
        from torch_geometric.data import Data

        HAR2EV = 27.211386246
        KCALMOL2EV = 0.04336414

        conversion = torch.tensor([
            1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
            1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
        ])

        # try:
        #     import rdkit
        #     from rdkit import Chem, RDLogger
        #     from rdkit.Chem.rdchem import BondType as BT
        #     from rdkit.Chem.rdchem import HybridizationType
        #     RDLogger.DisableLog('rdApp.*')

        # except ImportError:
        #     rdkit = None

        # if rdkit is None:
        if True:
            # print(("Using a pre-processed version of the dataset. Please "
            #        "install 'rdkit' to alternatively process the raw data."),
            #       file=sys.stderr)

            data_list = torch.load(self.raw_paths[0], weights_only=False)  # raw_paths[0] = qm9_v3.pt
            data_list = [Data(**data_dict) for data_dict in tqdm(data_list, desc="Loading pre-processed data")]

            if self.pre_filter is not None:
                data_list = [d for d in tqdm(data_list, desc="Filtering data") if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in tqdm(data_list, desc="Transforming data")]

            self.save(data_list, self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            if mol is None:
                print(f"Warning: RDKit failed to parse molecule {i}. Skipping.")
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            name = mol.GetProp('_Name')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


