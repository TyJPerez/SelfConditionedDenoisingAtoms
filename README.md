# SelfConditionedDenoising
Self-Conditioned Denoising for molecular pretraining

## Setup 
torch-md net derivative models include a optimized kernal for graph generation. Thus must be compiled before use

```bash
cd models/ET_models && python setup.py build_ext --inplace
```

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

## about this repo
this is repo was originally modified from the 'pre-training-via-denoising' repo here [https://github.com/shehzaidi/pre-training-via-denoising]
