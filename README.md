# Self-Conditioned Denoising for Atomistic Representation Learning

This is the official implementation for the paper:
**Self-Conditioned Denoising for Atomistic Representation Learning** *(arxiv link coming soon)*

Self-conditioned denoising (SCD) is a self-supervised pretraining method for atomistic data that is domain agnostic (small molecules, periodic materials, and/or protiens/biomolecules), and benefits from pretraining with both ground state and higher energy structures. This repo provides a simple implimentation of SCD using TorchMD-Net as the backbone and access to pretrained models through huggingface.
 
Pretrained checkpoints are available on HuggingFace for two domains:

| Model | HuggingFace Repo | Pretraining Dataset | Domain |
|---|---|---|---|
| `ct-scd-pcq` | [Ty-Perez/ct-scd-pcq](https://huggingface.co/Ty-Perez/ct-scd-pcq) | PCQM4MV2 | Molecules |
| `ct-scd-amp` | [Ty-Perez/ct-scd-amp](https://huggingface.co/Ty-Perez/ct-scd-amp) | Alex-MP-20 | Materials |


---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. (Recommended) Build the optimized graph creation kernel**

TorchMD-Net provides an optimized CUDA kernel for graph generation that speeds up on-the-fly graph creation for non-periodic systems. To use it, compile it with:

```bash
cd models/ET_models && python setup.py build_ext --inplace
```

A general graph creation method that also supports periodic materials is available without compilation. See `examples.ipynb` for details on using either method.

---

## Loading and Using a Pretrained Model

See `examples.ipynb` for instructions on:
- Loading a pretrained model from HuggingFace
- Instantiating an untrained model from a config
- Running a forward pass with and without the TorchMD-Net kernel

**Model outputs**

The model forward pass returns a dictionary with the following keys:

| Key | Description |
|---|---|
| `"y"` | Scalar property prediction |
| `"noise_pred"` | Predicted per atom coordinate noise |
| `"mol_emb"` | Output from graph level embedding projection head |
| `"atom_embs"` | (Optional) Output from atom level embeddings (requires return_atom_embs=True)|

---

## Pretraining

Pretrain a model using SCD on the PCQM4MV2 dataset:
```bash
python train.py --conf configs/pretrain_pcq.yaml
```

Pretrain a model using SCD on the Alex-MP-20 dataset:
```bash
python train.py --conf configs/pretrain_amp20.yaml
```

> `configs/pretrain_multidata.yaml` (multi-domain pretraining) requires access to the StructureCloud library, which is not yet publicly released.

---

## Finetuning

Load a pretrained checkpoint from HuggingFace:
```bash
python train.py --conf configs/finetune_qm9.yaml --load-hf ct-scd-pcq --job-id scd-pcq_qm9-homo
```

Load a pretrained checkpoint from a local path:
```bash
python train.py --conf configs/finetune_qm9.yaml --load-model 'experiments/{NAME}/{checkpoint}.ckpt' --job-id pretrained_qm9-homo
```

> The default finetuning target for QM9 is `homo`. This can be changed in `configs/finetune_qm9.yaml`.

> `configs/finetune_matbench.yaml` requires access to StructureCloud, which is not yet publicly released.

---

## Supported Datasets

| Dataset | Domain | Available Config |
|---|---|---|
| PCQM4MV2 | Molecules | `pretrain_pcq.yaml` |
| GEOM | Drug-like conformers | *Requires Structure Cloud* |
| Alex-MP-20 | Inorganic materials | `pretrain_amp20.yaml` |
| OMol25 | 4M subset | use 'OMOL25' in `pretrain_pcq.yaml` |
| QM9 | Small molecules | `finetune_qm9.yaml` |
| Matbench | Inorganic Materials | `finetune_matbench.yaml` |
| LBA | Ligand binding affinity | `data/datasets/lba.py` |

NOTE: Some datasets require our unleaeased helper library Structure Cloud. Coming soon! 

---

## Citation

If you use this work, please cite:

```bibtex
@article{perez2025scd,
  title   = {Self-Conditioned Denoising for Atomistic Representation Learning},
  author  = {Perez, Tynan and G´omez-Bombarelli, Rafael},
  year    = {2025},
  note    = {Preprint, arxiv link coming soon}
}
```

---

## About

This repository was originally forked from the [pre-training-via-denoising](https://github.com/shehzaidi/pre-training-via-denoising) repo.
