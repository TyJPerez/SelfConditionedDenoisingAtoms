from torch_geometric.datasets import MD17


#available molecules in MD17 dataset
# aspirin
# ethanol
# malonaldehyde
# "naphtalene" # NOTE: this is an actualy type in the torchgeometirc library, and thus is misspelled here
# "salicylic acid" # NOTE: there is an actual space in this key
# toluene
# uracil
# paracetamol


class xMD17(MD17):
    #include corrected URL for revised MD17 dataset
    revised_url = ('https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1')
    # def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
    def __init__(self, root, dataset_arg=None, transform=None, pre_transform=None, pre_filter=None):
        self.revised_url = ('https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1')
        default_root = 'tmp/md17'
        default_target = 'revised benzene'
        if root is None:
            root = default_root
        if dataset_arg is None:
            dataset_arg = default_target
        super().__init__(root, name=dataset_arg, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # Add forces to data object
        data.dy = data.force
        data.y = data.energy

        #remove force and energy attributes to avoid confusion
        data.force = None
        data.energy = None

        return data
