#imoirt dataset form tirch 
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from torch_geometric.data import Data
import torch


def _select_by_residue_indices(data, res_indices):
    '''Create a sub-dataset by selecting only the specified residues by their indices'''

    #select atoms in the closest residues
    atom_res_idx = data['atom_res_idx']

    #create a mask where valuse in atom_res_idx are in min_indices
    neighbor_atom_mask = torch.where(torch.isin(atom_res_idx, res_indices), True, False)
    nerighbor_res_mask = torch.tensor([i in res_indices for i in range(data['res_pos'].shape[0])])

    select_data = {
        'pos' : data['pos'][neighbor_atom_mask],
        'z' : data['z'][neighbor_atom_mask],
        'atom_chain_id' : [id for i, id in enumerate(data['atom_chain_id']) if neighbor_atom_mask[i]],
        'atom_res_idx' : data['atom_res_idx'][neighbor_atom_mask],
        'atom_is_ligand' : data['atom_is_ligand'][neighbor_atom_mask],
        'res_name' : [name for i,name in enumerate(data['res_name']) if nerighbor_res_mask[i]],
        'res_num' : data['res_num'][nerighbor_res_mask],
        'res_pos' : data['res_pos'][nerighbor_res_mask],
        'res_chain_id' : [id for i, id in enumerate(data['res_chain_id']) if nerighbor_res_mask[i]],
        'res_is_ligand' : data['res_is_ligand'][nerighbor_res_mask],
    }
    return select_data

def _filter_by_distance(data, center_pos, num_neighbors, shift=None, require_ligand=False):
    
    all_res = data['res_pos']

    # is_lig = data['res_is_ligand']
    # lig_com = data['res_pos'][is_lig]

    if shift is not None:
        center_pos = center_pos + shift.reshape(1,3)
    
    num_residues = all_res.shape[0]
    k_neighbors = min(num_neighbors + 1, num_residues)

    ## NOTE: distance calc will include ligand distance to self to select its own atoms as well
    dists = torch.cdist(all_res, center_pos)
    min_dists, min_indices = torch.topk(dists.squeeze(), k=k_neighbors, largest=False)
    
    if require_ligand:
        #ensure that at least one ligand residue is included
        is_lig = data['res_is_ligand']
        if not is_lig[min_indices].any():
            #add the ligand residue index to min_indices
            lig_indices = torch.where(is_lig)[0]
            min_indices = torch.cat([min_indices[:-1], lig_indices[:1]], dim=0)
    
    return _select_by_residue_indices(data, min_indices)

def _sample_residue_neighbors(data, res_id, num_neighbors=30, shift=None):
    ''' 
    Extract specified residue and neighboring residues
    returns full residues 
    '''
    target_res_pos = data['res_pos'][res_id].unsqueeze(0)  # [1, 3]

    return _filter_by_distance(data, target_res_pos, num_neighbors, shift)

    # all_res = data['res_pos']
    # if shift is not None:
    #     target_res_pos = target_res_pos + shift.reshape(1,3)

    # dists = torch.cdist(all_res, target_res_pos)
    # min_dists, min_indices = torch.topk(dists.squeeze(), k=(num_neighbors + 1), largest=False)
    # return _select_by_residue_indices(data, min_indices)

def _sample_ligand_neighbors(data, num_neighbors=30, shift=None, require_ligand=False):
    ''' 
    Extract ligand and neighboring residues
    returns full residues 
    '''
    is_lig = data['res_is_ligand']
    lig_com = data['res_pos'][is_lig]
    all_res = data['res_pos']

    return _filter_by_distance(data, lig_com, num_neighbors, shift, require_ligand=require_ligand)

    # if shift is not None:
    #     lig_com = lig_com + shift.reshape(1,3)

    # ## NOTE: distance calc will include ligand distance to self to select its own atoms as well
    # dists = torch.cdist(all_res, lig_com)
    # min_dists, min_indices = torch.topk(dists.squeeze(), k=(num_neighbors + 1), largest=False)
    # return _select_by_residue_indices(data, min_indices)


class SAIRDataset(Dataset):
    def __init__(self, 
                 root,
                 source_dir ='processed',
                 ):
        #TODO: temp fix
        root = '/global/homes/t/tyjperez/Projects/SelfConditionedDenoising/tmp/sair_data/'

        self.source_data = os.path.join(root, source_dir)

        assert os.path.exists(self.source_data), f"Source data directory does not exist: {self.source_data}"
        src_files = [f for f in tqdm(os.listdir(self.source_data)) if f.endswith('.pt')]
        # self.data_files = src_files
        self.dataset = src_files

        self.labels_path = os.path.join(root, 'sair.parquet')

    def load_data(self, idx):
        path = os.path.join(self.source_data, self.dataset[idx])
        data = torch.load(path, weights_only=True)

        #data keys
        # 'pos' : pos, # all positions of atoms [N, 3]
        # 'z' : z,     # atomic numbers of atoms [N, ]
        # 'atom_chain_id' : atom_chain_id, # chain IDs of atoms [N, ]
        # 'atom_res_idx' : atom_res_idx, # indexing of which residue each atom belongs to [N, ]
        # 'atom_is_ligand' : atom_is_ligand, # ligand atom indices, binary mask [N, ]
        # 'res_name' : res_name, # residue names [R, str]
        # 'res_num' : res_num, # residue types as numbers [R, ]
        # 'res_pos' : res_pos, # residue centroid positions [R, 3]
        # 'res_chain_id' : res_chain_id, # chain IDs of residues [R, ]
        # 'res_is_ligand' : res_is_ligand # ligand residue indices, binary mask [R, ]

        #load labels from parquet at specified index if needed
        #TODO

        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)

class SampledSAIRDataset(SAIRDataset):
    def __init__(self, 
                 root, 
                 dataset_arg=None, # placeholder for compatibility
                 num_neighbors=30,
                 p_pocket = 0.5, # probability of sampling ligand neighborhood vs random residue neighborhood
                #  p_drop_ligand = 0.1, # probability of dropping ligand atoms from neighborhood
                 random_shift = 0.0, 
                 transform=None,
                 ):
        super().__init__(root=root)
        self.num_neighbors = num_neighbors
        self.p_pocket = p_pocket
        self.num_neighbors = num_neighbors
        self.random_shift = random_shift
        self.transform = transform
    
    def conformer_count(self):
        #assume each protein has 5 conformers
        return len(self)*5

    def __getitem__(self, idx):
        idx = int(idx)
        conformers = super().__getitem__(idx)
        conf_keys = list(conformers.keys())
        #choose a random key
        rand_conf = conf_keys[torch.randint(len(conf_keys), (1,)).item()]
        sample = conformers[rand_conf]

        ### Subsample local neighborhoods
        rand_shift = None
        if self.random_shift > 0.0:
            rand_shift = torch.randn(3) * self.random_shift
        #sample uniform random to choose ligand or residue neighborhood
        if torch.rand(1).item() < self.p_pocket:
            sample = _sample_ligand_neighbors(sample, num_neighbors=self.num_neighbors, shift=rand_shift)
        else:
            #choose a random residue index to center the neighborhood
            res_id = torch.randint(sample['res_pos'].shape[0], (1,)).item()
            sample = _sample_residue_neighbors(sample, res_id, num_neighbors=self.num_neighbors, shift=rand_shift)

        # out = Data(**sample)
        out = Data()
        out.pos = sample['pos']
        out.z = sample['z']
        out.natoms = torch.tensor([sample['pos'].shape[0]], dtype=torch.long)
        # torch.tensor(sample['pos'].shape[0])

        if self.transform is not None:
            out = self.transform(out)

        return out

# root = '/global/homes/t/tyjperez/Projects/SelfConditionedDenoising/tmp/sair_data/'

# test_ds = SampledSAIRDataset(root)
# print(f"Dataset length: {len(test_ds)}")


class SAIRPocket(SAIRDataset):
    ''' used for pocket conditional denoising '''
    def __init__(self, 
                 root, 
                 dataset_arg=None, # placeholder for compatibility
                 num_neighbors=30,
                 random_shift = 2.0, 
                 transform=None,
                 pocket_first = True, # determine order of output data (pocket, ligand) vs (ligand, pocket)
                 ):
        super().__init__(root=root)
        self.num_neighbors = num_neighbors
        self.num_neighbors = num_neighbors
        self.random_shift = random_shift
        self.transform = transform
        self.pocket_first = pocket_first
        self.require_ligand = True # ensure ligand is always included in neighborhood

        if dataset_arg is not None:
            if 'ligand' in dataset_arg:
                self.pocket_first = False
                print("--- SAIRPocket: setting ligand_first output order. ---")
                
    
    def conformer_count(self):
        #assume each protein has 5 conformers
        return len(self)*5

    def select_sample(self, idx, conf_id=None):
        out = self.__getitem__(idx, conf_id)
        return out
    
    def __getitem__(self, idx, conf_id=None):
        idx = int(idx)
        conformers = super().__getitem__(idx)
        conf_keys = list(conformers.keys())
        if conf_id is None:
            #choose a random key
            rand_conf = conf_keys[torch.randint(len(conf_keys), (1,)).item()]
            sample = conformers[rand_conf]
        else:
            sample = conformers[conf_id]

        ### Subsample local neighborhoods
        rand_shift = None
        if self.random_shift > 0.0:
            rand_shift = torch.randn(3) * self.random_shift

        #sample the pocket neighborhood
        sample = _sample_ligand_neighbors(sample, 
                                          num_neighbors=self.num_neighbors, 
                                          shift=rand_shift, 
                                          require_ligand=self.require_ligand)

        #split the ligand and pocket
        # is_lig = sample['res_is_ligand']
        ligand_data = Data()
        ligand_data.pos = sample['pos'][sample['atom_is_ligand']]
        ligand_data.z = sample['z'][sample['atom_is_ligand']]
        ligand_data.natoms = torch.tensor([ligand_data.pos.shape[0]], dtype=torch.long)

        pocket_data = Data()
        pocket_data.pos = sample['pos'][~sample['atom_is_ligand']]
        pocket_data.z = sample['z'][~sample['atom_is_ligand']]
        pocket_data.natoms = torch.tensor([pocket_data.pos.shape[0]], dtype=torch.long)
        
        if self.pocket_first:
            out = (pocket_data, ligand_data)
        else:
            out = (ligand_data, pocket_data)

        if self.transform is not None:
            out = self.transform(out)

        return out