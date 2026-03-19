# Self-Conditioned Denoising for Atomistic Representation Learning

This is the official implementation for the paper:
[**Self-Conditioned Denoising for Atomistic Representation Learning**](https://arxiv.org/pdf/2603.17196)

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

if you do not want to use the compiled TorchMD-Net Kernel, set noise_in_loader=True in the config file for you select run.

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
python train.py --conf configs/pretrain_pcq.yaml # use faster graph creation
python train.py --conf configs/pretrain_pcq.yaml --noise_in_loader True # if torchnetmd kernal is not compiled
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
Note, if you did not compile the torchnet graph kernel us must set `--noise_in_loader True`

> The default finetuning target for QM9 is `homo`. This can be changed in `configs/finetune_qm9.yaml`.

> `configs/finetune_matbench.yaml` requires access to StructureCloud, which is not yet publicly released.

---

## Supported Datasets

| Dataset | Domain | Available Config |
|---|---|---|
| PCQM4MV2 | Molecules | `pretrain_pcq.yaml` |
| GEOM | Drug-like conformers | *Requires Structure Cloud* |
| Alex-MP-20 | Inorganic materials | `pretrain_amp20.yaml` |
| OMol25 | Organic Molecules | use 'OMOL25' in `pretrain_pcq.yaml` |
| QM9 | Small molecules | `finetune_qm9.yaml` |
| Matbench | Inorganic Materials | `finetune_matbench.yaml` |
| LBA | Proteins | `data/datasets/lba.py` |

NOTE: Some datasets require our unleaeased helper library Structure Cloud. Coming soon! 

---

## Batch Clipping
Graph batch memory can vary a lot across steps. Batch clipping keeps each batch under a node budget to reduce OOM risk while maintaining good GPU utilization.

In this repo, clipping is handled automatically by `BatchClipper` during training. If a batch is too large, extra samples are moved to an internal cache and re-used in later smaller batches.

Set these config parameters to control clipping behavior:

- `max_nodes_per_batch`: **main clipping limit** (max total nodes allowed per batch). If unset, clipping is disabled.
- `batch_clipper_cache_size`: max number of clipped samples to keep in cache for refill.
- `allow_test_clipping`: if `True`, clipping is also applied on validation/test loaders. For strict eval comparability, set this to `False`.

Example:

```yaml
max_nodes_per_batch: 4000
batch_clipper_cache_size: 1000
allow_test_clipping: false
```

You can set these directly in your run config (for example, `configs/pretrain_*.yaml` or `configs/finetune_*.yaml`) or override from CLI:

```bash
--max-nodes-per-batch 4000 --batch-clipper-cache-size 1000 --allow-test-clipping False
```
---

## Graph creation options

| Mode | `noise_in_loader` | `allow_periodic` | Notes |
|---|---|---|---|
| TorchMD-Net compiled kernel (fastest) | `False` | `False` | Preferred for non-periodic molecules when extension is compiled. |
| Loader-side graph creation | `True` | `False` or `True` | Works without compiled extension and is required for periodic workflows. |

Note: `noise_in_loader=True` does not necessarily apply positional noise if noise scale is set to zero.

### Which mode should I use?

| Task | Recommended settings | Why |
|---|---|---|
| Molecule pretraining/fine-tuning (non-periodic) with compiled extension | `noise_in_loader=False`, `allow_periodic=False` | Uses fast TorchMD-Net graph construction. |
| Molecule training without compiled extension | `noise_in_loader=True`, `allow_periodic=False` | Keeps training functional without CUDA extension build. |
| Periodic/materials training | `noise_in_loader=True`, `allow_periodic=True` | Periodic edges are built in loader path. |

### Self-conditioning batch contract

When `self_cond=True` and loader-side transforms are active:

- The data pipeline may return paired graph views per sample (instead of a single graph).
- Training uses tuple-aware clipping internally (`TupleBatchClipper`) in this mode.
- If you add custom datasets/transforms, keep this paired-view contract intact for SCD runs.

---

## Evaluation Semantics

During `fit`, validation and test are not fully separate in this codebase:

- Every epoch, the validation loader runs.
- Additionally, when `current_epoch % test_interval == 0` (and training has started), a test loader is appended and evaluated in the same validation phase.

This means reported metrics may include periodic test-set evaluation during training rather than test-only-at-end behavior.

### Recommended settings for clean comparisons

- Set `allow_test_clipping: false` for strict evaluation comparability.
- Keep `test_interval` explicit in configs so readers know how often test metrics are sampled during fit.
- For final reporting, run a dedicated test pass after training and treat that as the headline number.

---

## Citation

If you use this work, please cite:

```bibtex
@article{perez2026,
  author = {Tynan Perez and Rafael Gomez-Bombarelli},
  title = {Self-Conditioned Denoising for Atomistic Representation Learning},
  year = {2026}
  url={https://arxiv.org/abs/2603.17196}
}
```

---

## About

This repository was originally forked from the [pre-training-via-denoising](https://github.com/shehzaidi/pre-training-via-denoising) repo.
