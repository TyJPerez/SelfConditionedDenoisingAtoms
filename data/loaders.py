from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_warn

from .utils import make_splits, MissingEnergyException
from . import datasets

from torch_scatter import scatter

from .mol_aug import torsion_transform, center, random_rotate, invert

import numpy as np
import math

from data.datasets.transforms import AddStandardKeys, RandomCellRepeats, SCD_noise, CreateGraph, Compose

class DataModule(LightningDataModule):
    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self.params = hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

    def setup(self, stage):
        if self.dataset is None:
               
                def molecule_transform(data):
                    # not aplicable for materials and periodic systems
                    invert_prob = self.params["p_invert"]
                    angle_std = self.params["torsion_angle_std"]
                    max_torsions = self.params["max_bonds_rotated"]
                    rand_rotate = self.params["random_rotate"]
                    center_pos = self.params.get("center", True)

                    #apply random torsion 
                    if angle_std > 0 and max_torsions > 0:
                        data = torsion_transform(data,
                                                angle_std=angle_std,
                                                max_torsions=max_torsions
                                                 )
                        #this adds new keys: 'tor_edges', 'tor_angles'

                    #invert the coordinates with probability 0.5
                    if invert_prob > 0:
                        data = invert(data, prob=invert_prob)
                    
                    #center the coordinates
                    if center_pos:
                        data = center(data)

                    #apply random rotation
                    if rand_rotate:
                        rot_angle= 2*math.pi if rand_rotate else 0.0
                        data = random_rotate(data, max_angle=rot_angle)

                    return data
                
                # Loader-side graph/noise path: required for periodic graphs and the fallback
                # path when the TorchMD compiled graph kernel is unavailable.
                if self.params['noise_in_loader']: #required for periodic systems
                    p_rep = self.params["p_cell_repeat"]
                    rep_iters = self.params["cell_repeat_iters"]
                    min_atoms = self.params["rep_min_atoms"] # cell is repeated if num atoms < min_atoms
                    
                    corrupt_noise = self.params["noise_scale"]
                    graph_cutoff = self.params["graph_cutoff"]
                    neighbor_method = self.params["neighbor_method"]
                    max_neighbors = self.params["max_neighbors"]
                    # neighbor_method = self.params.get("neighbor_method", "brute")
                    # neighbor_method = self.params.get("neighbor_method", "grid")
                    # max_neighbors = self.params.get("max_neighbors", 32)

                    #base transform
                    base_transform = [
                            AddStandardKeys(),
                            RandomCellRepeats(
                                p_rep=p_rep,
                                rep_iters=rep_iters,
                                min_atoms=min_atoms,
                                max_reps_per_axis=2,
                                )]
                    
                    # self_cond + simple_noise=False returns paired views used by self-conditioning.
                    if self.params['self_cond']: # SCD loss - return two samples
                        # assert self.params['pretraining'], "SCD 'noise in loader' is only implimented for pretraining phase"
                        if self.params['pretraining']:
                            # reg_noise_scale = 0.005
                            base_transform.append(
                            SCD_noise(
                                simple_noise=False,
                                reg_noise = 0.005, #self.params["reg_noise"],
                                corrupt_noise = corrupt_noise,
                                cutoff = graph_cutoff,
                                max_neighbors = max_neighbors,
                                neighbor_method = neighbor_method,
                                ),
                            )
                            transform = Compose(base_transform)
                            # Pretraining validation mirrors train corruption to track denoising behavior.
                            val_transform = transform # validation will also noise the data
                        else:
                            # Finetuning keeps train corruption but evaluates on clean graphs.
                            train_transform = base_transform.copy()
                            train_transform.append(
                            SCD_noise(
                                simple_noise=False,
                                reg_noise = 0.00001, #self.params["reg_noise"],
                                corrupt_noise = corrupt_noise,
                                cutoff = graph_cutoff,
                                max_neighbors = max_neighbors,
                                neighbor_method = neighbor_method,
                                ),
                            )
                            transform = Compose(train_transform)
                            val_transform = Compose(base_transform) # validation uses only structural transforms


                        
                    else: # otherwise standard loss, with optional noise predictions for coord or regularization
                        base_transform.append(
                            SCD_noise(
                                simple_noise=True, #return only one batch
                                reg_noise = 0.000, #self.params["reg_noise"],
                                corrupt_noise = corrupt_noise,
                                cutoff = graph_cutoff,
                                max_neighbors = max_neighbors,
                                neighbor_method = neighbor_method,
                                ),
                            )
                        transform = Compose(base_transform)
                        # Validation keeps deterministic graph creation (no corruption, no random repeats).
                        val_transform = [
                            AddStandardKeys(), 
                            SCD_noise( #this only adds graph without noise
                                simple_noise=True, #return only one batch
                                reg_noise = 0.00,
                                corrupt_noise = 0.0,
                                cutoff = graph_cutoff,
                                max_neighbors = max_neighbors,
                                neighbor_method = neighbor_method,
                                ),
                        ]
                        val_transform = Compose(val_transform) 
                    
                else:
                    # transform = molecule_transform
                    # val_transform = None

                    transform = Compose([AddStandardKeys(),molecule_transform])
                    val_transform = Compose([AddStandardKeys()])


                dataset_factory = lambda t: getattr(datasets, self.params["dataset"])(self.params["dataset_root"], dataset_arg=self.params["dataset_arg"], transform=t)

                # Train split can be noisy/augmented, while val/test should stay comparable.
                # Instantiate two dataset views so splits share indices but use different transforms.
                # Noisy version of dataset
                self.dataset_maybe_noisy = dataset_factory(transform)
                # Clean version of dataset
                self.dataset = dataset_factory(val_transform)
        
        if self.params['predefined_splits']:
            #load predefined splits from dataset if available
            self.train_dataset = self.dataset_maybe_noisy.get_subset('train')
            self.val_dataset = self.dataset.get_subset('val')
            self.test_dataset = self.dataset.get_subset('test')

        else:
            self.idx_train, self.idx_val, self.idx_test = make_splits(
                len(self.dataset),
                self.params["train_size"],
                self.params["val_size"],
                self.params["test_size"],
                self.params["seed"],
                join(self.params["log_dir"], "splits.npz"),
                # self.params["splits"],
                None, # do not provide order to dataset
            )
            print(
                f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
            )

            self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)

        #get parameter but default to false if not present
        if self.params.get("standardize", False):
            self._standardize()

        # if self.params["standardize"]:
        #     self._standardize()


    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]

        # By design, test is run periodically during fit when test_interval is reached.
        if (
            len(self.test_dataset) > 0
            and self.trainer.current_epoch % self.params["test_interval"] == 0
            and self.trainer.global_step > 1
        ):  
            loaders.append(self._get_dataloader(self.test_dataset, "test"))

        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        # Check if reload_dataloaders_every_n_epochs exists (newer versions) or reload_dataloaders_every_epoch (older versions)

        reload_attr = None
        if hasattr(self.trainer, 'reload_dataloaders_every_n_epochs'):
            reload_attr = self.trainer.reload_dataloaders_every_n_epochs
        elif hasattr(self.trainer, 'reload_dataloaders_every_epoch'):
            reload_attr = self.trainer.reload_dataloaders_every_epoch
        
        store_dataloader = (
            store_dataloader and not reload_attr
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.params["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.params["inference_batch_size"]
            shuffle = False

        n_workers = self.params.get("num_workers", 1)
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers, #self.params["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        # Check if the train_dataset has a normalize method
        if hasattr(self.train_dataset, "normalize"):
            # use predefined mean and std from dataset if available
            print("Using predefined mean and std from dataset")
            self._mean, self._std = self.train_dataset.normalize()
            return

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.params["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
