
##debug run command
python train.py --conf configs/debug_0.yaml --job-id refactor_debug

python train.py --conf configs/pretrain_pcq.yaml --job-id debug_ppcq

python train.py --conf configs/pretrain_amp20.yaml --job-id debug_pamp20


python train.py --conf configs/finetune_qm9.yaml --job-id refactor_qm9_test

### all pretrained models - finished
cet-pcq

file name               descriptoin
----
scd-s-n004_pcq              # cet-pcq
scd-n004_pcq300k            # cet-pcq300k
scd-frad-n004_pcq_b32n4     # cget on full pcq

----
scd-n004_geom_b48n4         # geom10 pretrained
scd-n004_geom01_b48n4       # geom1 pretrained

----
cet-amp20_b142n1_0          # cet on AMP20
cget-amp20_b32n2_0          # cget on AMP20

----
cet-sair_b12n2_0            # cet on Sair dataset

cet-sairpocket_b12n2_0      # cet on Sair dataset, denoising of ligand, given pocket embedding

cet-sairligand_b12n2_0      # cet on Sair dataset, denoising of pocket, given ligand embedding


### in progress

cet-all_b56n4_2         # combind dataset




------------

## to run 



## nov 21
# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_ligmask_debug -r true 
# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_ligmask_debug-sum --set-head-agg sum -r true

# python train.py --conf configs/cet-lba_grimm.yaml --job-id lba_protmask_mean2 -r true

## nov 20
# python train.py --conf configs/cget-mbgap_grimm.yaml --load-hf cget-amp20 --job-id cget-amp20_b16_hm_lr1e4 --lr 1e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_b16_hm_nonstd --standardize False

# python train.py --conf configs/cet-lba_grimm.yaml --job-id et_b5_emay01
# python train.py --conf configs/cet-lba_grimm.yaml --job-id et_b5_wd1 --weight-decay 0.1



### 

## nov 19
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr1e4 --lr 1e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr2e4_nz005 --noise-scale 0.005 --denoising-weight 0.1
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4_plt-dbug -r true --load-model experiments/etmbgap_grm_b16_hm_lr2e4/last.ckpt

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4 -r true -ep 54 --no-wandb-resume

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_debug_dp -r true --no-wandb-resume

# python train.py --conf configs/_debug.yaml --job-id plot_debug_x --no-wandb-resume

## nov 18
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr1e4_nz-aug --noise-scale 0.005 --denoising-weight 0.1 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr2e4 --lr 2e-4 
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr5e5 --lr 5e-5 
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_lr5e4 --lr 5e-4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf cet-amp20 --job-id cet-amp20_b16_hm_lr2e4
# python train.py --conf configs/cget-mbgap_grimm.yaml --job-id cget_b16_hm_lr2e4
# python train.py --conf configs/cet-mbgap_grimm.yaml --load-hf scd-s-n004_pcq --job-id cet-pcq_b16_hm_lr2e4

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id cet_b16_hm_lr2e4_cut8 --graph_cutoff 8.0


# python train.py --conf configs/cet-lba_grimm.yaml --job-id etlba_debug_0

# Nov 16
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4_wd1e5
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4_prep03 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_headmean_lr1e4
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16nz0dp001_hm_lr1e4_prep03 --p_cell_repeat 0.3 --cell_repeat_iters 2
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_nz005_lr1e4 --noise-scale 0.005 --denoising-weight 0.1


# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_hm_dp01 --final-droppath 0.1

# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_debug_head --load-hf scd-s-n004_pcq

# Nov 16
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_sum
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_cutoff8 --graph_cutoff 8.0
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e3
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id etmbgap_grm_b16_nz0_dp001_mean_lr1e4

##
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep03
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep0 --p_cell_repeat 0.0
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep0_nz0 --p_cell_repeat 0.0 --noise-scale 0.000001
# python train.py --conf configs/cet-mbgap_grimm.yaml --job-id et_mbgap_baseline_grm_b24_prep03_beta109 --beta1 0.9

# python train.py --conf configs/scd-all_pretrain.yaml --job-id debug_scdall
# python train.py --conf configs/debug_qm9.yaml --job-id debug_graphgen

# python train.py --conf configs/frad_pretrain.yaml --job-id frad_debug
# python train.py --conf configs/scd-s-n004_pretrain.yaml --job-id debug_pt --batch-size 32

# python train.py --conf configs/scd-s-n004_pretrain.yaml --job-id debug_ctr_b64 --batch-size 64 -r True --train-size 5000
# python train.py --conf configs/scd-n004-ctr_pretrain.yaml --job-id debug_ctr2_b64 --batch-size 64 -r True --train-size 5000


# python train.py --conf configs/scd-qm9-ft-base.yaml --job-id debug_ctr_ft_test --pretrained-model experiments/debug_ctr_b64/last.ckpt

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id load_test_debug --dataset-arg aspirin


### running oct 16
# python train.py --conf configs/scdL-md17.yaml --load-hf scd-s-n004_pcq --job-id cet-pcq_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scdL-md17.yaml --load-hf CET-GEOM10_0 --job-id cet-geom10_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scdL-md17.yaml --load-hf CET-GEOM01_0 --job-id cet-geom1_b8_aspirin_1k_dnz --dataset-arg aspirin

# python train.py --conf configs/cget-md17.yaml --load-hf CGET-pcq_0 --job-id cget-pcq_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/cget-qm9-ft-base.yaml --load-hf CGET-pcq_0 --job-id cget-debug --dataset-arg homo
# python train.py --conf configs/cget-md17.yaml --load-hf CGET-pcq_0 --job-id cget-debug-md17 --dataset-arg aspirin

### running oct 15
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_1k_dnz --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_benzene_1k_dnz --dataset-arg benzene
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_ethanol_1k_dnz --dataset-arg ethanol
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_malonaldehyde_1k_dnz --dataset-arg malonaldehyde

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_naphthalene_1k_dnz --dataset-arg naphtalene 
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_salicylic_acid_1k_dnz --dataset-arg "salicylic acid"

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_toluene_1k_dnz --dataset-arg "toluene"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_uracil_1k_dnz --dataset-arg "uracil"

### running oct 14
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_atom_emb --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_mol_emb --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_0 --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_nonorm --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aps_fwd_dy_normx1k --dataset-arg aspirin

### running oct 13

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id debug_match_ema --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_1y2dy_1 --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id match_ema_dy --dataset-arg aspirin

#### Running oct 10
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_lrg_cutoff_10A --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_eng_cond --dataset-arg aspirin

##### running oct 9
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id debug_aspirin_scd --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_scd_dnw03 --dataset-arg aspirin --denoising-weight 0.3
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id aspirin_scd_nocond --dataset-arg aspirin

# python train.py --conf configs/scd-md17-SL.yaml --job-id aspirin_scd_nocond_fresh --dataset-arg aspirin

##### running oct 8
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_wdy1wy01  --dataset-arg aspirin --energy-weight 0.1 --force-weight 1.0
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_wd0 --dataset-arg aspirin --weight-decay 0.0
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_lr001 --dataset-arg aspirin --lr 0.001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_nullC_lr001_RoP --dataset-arg aspirin --lr 0.001 --lr-schedule 'reduce_on_plateau' --num-steps 200000
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSm-n04_pcq --job-id scdSmn04_b8_aspirin --dataset-arg aspirin


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin_s2 --dataset-arg aspirin --seed 2

### running oct 7
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_LAref --prior-model LearnableAtomref

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_aspirin --dataset-arg aspirin
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_ethanol --dataset-arg ethanol
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_malonaldehyde --dataset-arg malonaldehyde
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_naphthalene --dataset-arg naphtalene # NOTE type in torchgeometric libn
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_salicylic_acid --dataset-arg "salicylic acid"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_toluene --dataset-arg "toluene"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_uracil --dataset-arg "uracil"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_paracetamol --dataset-arg "paracetamol"
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdSn004_b8_azobenzene --dataset-arg "azobenzene"


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs8 --batch-size 8

# paracetamol
# azobenzene

##### RUNNING #### - oct 4
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_long-lr2 --dataset-arg alpha
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16 --batch-size 16
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n1e4 --batch-size 16 --noise-scale 0.0001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_dp01 --batch-size 16 --final-droppath 0.1

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4 --batch-size 16 --noise-scale 0.0005 # best 
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n1e3 --batch-size 16 --noise-scale 0.001
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4_dp01 --batch-size 16 --noise-scale 0.0005 --final-droppath 0.1
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b16_n5e4_dn --batch-size 16 --noise-scale 0.0005 --denoising-weight 0.01

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_1k
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_RlrOnP
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_emay01 --ema-alpha-y 0.1
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b8_emay02 --ema-alpha-y 0.2

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs8 --batch-size 8
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_muval_bs32 --batch-size 32 --test-interval 50


# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_nonorms_lr3dp001 --dataset-arg alpha
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_nonoise --dataset-arg alpha
# python train.py --conf configs/scd-qm9-ft-long.yaml --load-hf scd-s-n004_pcq --job-id scd-alpha_LAref --dataset-arg alpha


# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs512-cos

# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs8-cos

# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs16-add_embhead

# python train.py --conf configs/et-rmd17-SL.yaml --job-id et-debug
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-baseline
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001
# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001-dbwu
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_benz_debug


# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995 --beta1 0.995

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995_bs32 --beta1 0.995 --batch-size 32

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_beta995_n005 --beta1 0.995 --noise-scale 0.005
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_b29 --beta1 0.995 --beta2 0.9
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_nonorms --beta1 0.995

# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_nonorms_lr5e4 --beta1 0.995 --lr 0.0005
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_vecnorms_lr4e4 --beta1 0.995 --lr 0.0004
# python train.py --conf configs/scd-md17-SL.yaml --load-hf scdSs_n004_pcq_400k --job-id scdS_benz_b1995_vecnorms_lr1e4_1 --beta1 0.995 --lr 0.0001 --inference-batch-size 64

##### RUNNING ####
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u0_b64 --dataset-arg u0
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u0_b32 --dataset-arg u0
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_u296_b128 --dataset-arg u298
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_h296_b128 --dataset-arg h298
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128 --dataset-arg g298

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128_b1995 --dataset-arg g298




# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_b128_b1995_bs256 --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_bswp --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b9999 --dataset-arg g298 --batch-size 256 --inference-batch-size 256

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b98 --dataset-arg g298 --batch-size 256 --inference-batch-size 256
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id scd-sn004_g296_bs256_b98_1 --dataset-arg g298 --batch-size 256 --inference-batch-size 256

# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id test_g296_bs256_b995_yema001 --dataset-arg g298 --batch-size 256 --inference-batch-size 256 --ema-alpha-y 0.01
# python train.py --conf configs/scd-qm9-ft-energy.yaml --load-hf scd-s-n004_pcq --job-id test_g296_bs256_b995_yema01 --dataset-arg g298 --batch-size 256 --inference-batch-size 256 --ema-alpha-y 0.1




# python train.py --conf configs/scd-md17-SL.yaml --job-id scdsm-benz-bs512

# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b8
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b32
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b128
# python train.py --conf configs/et-md17-SL.yaml --job-id etsm-benz-b512

# python train.py --conf configs/scd-md17-SL.yaml --job-id scd-benz-wd001-dbwu-b32

# python train.py --conf configs/et-QM9-FT.yaml --job-id debug-coord-u0 --pretrained-model experiments/coord-baseline/last.ckpt
# python train.py --conf configs/et-QM9-FT-energy.yaml --job-id debug-coord-u0 --pretrained-model experiments/coord-baseline/last.ckpt

# python train.py --conf configs/et-QM9-SL.yaml --job-id et-SL-u0


#### load pretrained hf model 
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0-test

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0_atom_test

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cet-ft-u0Aref_embpregate

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id cetft-u0Aref_yema08_dnw08_lr5e5

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_momentum
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_yema-modema
# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0atom-std_no-ema

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_newLN

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLN-H-E_lowlr_n0001

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLN-H-E_lowlr_n0

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLNHE_lr1e4n005_rDyT

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id u0Aref_rsetLNHE_lr1e4n005_rLN

# python train.py --conf configs/nersc_qm9-ft-long.yaml --load-hf cet-pcq_2 --job-id alpha_rsetLNHE_DyT

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id h298_wd03
# python train.py --conf configs/nersc_qm9-ft-long.yaml --load-hf cet-pcq_2 --job-id zpve_cet

# python train.py --conf configs/nersc_qm9-ft-base.yaml --load-hf cet-pcq_2 --job-id ehomo_rsetemb

# python train.py --conf configs/nersc_qm9-ft-base.yaml --load-hf cet-pcq_2 --job-id ehomo_asis

# python train.py --conf configs/nersc_qm9-ft-energy.yaml --load-hf cet-pcq_2 --job-id g298_nw05_b64



##### SL TEST
# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_newscd_test

# python train.py --conf configs/et-QM9-SL.yaml --job-id u0_et-baseline

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_newscd_sig-gate

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-x

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-dpth0

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-jpth01

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0_scd_detach-invpnorm-jpth01-vecnorm

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0-scd_wd001_dpth-warmup 

# python train.py --conf configs/scd-qm9-sl.yaml --job-id u0-scd_vecDyT_dpth-warmup

# python train.py --conf configs/scd-qm9-sl.yaml --job-id ckpt_debug_test


# python train.py --conf configs/scd-qm9-sl.yaml --job-id momup_on