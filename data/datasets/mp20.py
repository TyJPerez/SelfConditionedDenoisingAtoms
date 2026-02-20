from StructureCloud.Datasets import MP20_dataset
from data.datasets.transforms import AddStandardKeys, RandomCellRepeats
from torch_geometric.transforms import Compose

import torch
import numpy as np


class mp20_dataset(MP20_dataset):
    def __init__(self, p_rep=0.1, rep_iters=1, min_atoms=4, **kwargs):
        
        transforms = [
            AddStandardKeys(),
            RandomCellRepeats(
                p_rep=p_rep,
                rep_iters=rep_iters,
                min_atoms=min_atoms,
                max_reps_per_axis=2,
            )]
        #check if kwargs has a transform, if not set a default one
        if 'transform' in kwargs:
            transforms.append(kwargs['transform'])

        kwargs['transform'] = Compose(transforms)

        super().__init__(**kwargs)