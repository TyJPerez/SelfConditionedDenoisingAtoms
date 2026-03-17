# Self-Conditioned Denoising for Atomistic Representation Learning
This is the official implementation for the paper:
[Self-Conditioned Denoising for Atomistic Representation Learning](https://insert-the-link-to-the-paper-here.com)

This repo provides simple access to several pretrained models pretrained by self-conditioned denoising as described in our publication linked above.

## Setup 
1. Create and set up your environment 
```bash
pip install -r requirements.txt
```

2. RECOMMENDED (Optional): build graph creation kernal
TorchMD-Net provides an optimized kernal for graph generation that can speed up on the fly graph creation for non-periodic systems. To use this kernal, you must compile it using the following script.

```bash
cd models/ET_models && python setup.py build_ext --inplace
```

An alternative more general graph creation method that is applicable to periodic materials is also provided. See `examples.ipynb' for additional details on how to use either graph creation method.

## Loading and using a pretrained model
See examples in `examples.ipynb`

## Pretraining 
Pretrain a model using scd on the PCQ dataset
```bash
python train.py --conf configs/pretrain_pcq.yaml
```

Pretrain a model using scd on the Alex-MP-20 dataset
```bash
python train.py --conf configs/pretrain_amp20.yaml
```

## Finetuning

Load a pretrained checkpoint from our huggingface
```bash
python train.py --conf configs/finetune_qm9.yaml --load-hf ct-scd-pcq --job-id scd-pcq_qm9-homo
```

Load a pretrained checkpoint from a local repo
```bash
python train.py --conf configs/finetune_qm9.yaml --load-model 'experiments/{NAME}/{checkpoint}.ckpt' --job-id pretrained_qm9-homo
```

## about
this is repo was originally forked from the 'pre-training-via-denoising' repo here [https://github.com/shehzaidi/pre-training-via-denoising]


## Citation

