import numpy as np
import torch
import torch.nn.functional as F
from StructureCloud.Datasets import StructureCloudDataset as scd
from StructureCloud.Datasets import TGDataset as tgd
from StructureCloud.Datasets import TGDataset as tgd
from torch_geometric.loader import DataLoader as GDataLoader

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from models.ET_models.cond_et import CondEquivMultiHeadAttention
from models.ET_models.utils import act_class_mapping, OptimizedDistance, ExpNormalSmearing

from data.loaders import DataModule
from models.model_helper2 import create_model

from model_eval.clock_model import (
    analyze_model_performance, 
    print_performance_summary,
    compare_models,
    print_detailed_memory_breakdown
)

def print_model_param_counts(model, model_name):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_string = f'{model_name} has {num_params/1e6:.2f} million parameters'
    print(print_string)
    return print_string

def print_rep_param_count(model, model_name):
    return print_model_param_counts(model.rep_model, model_name + "_backbone")

def main():

    ### LOAD DATA ###

    et_config_file = 'configs/model_configs/torchnet_basic.yaml'
    frad_config_file = 'configs/model_configs/frad_basic.yaml'
    scd_config_file = 'configs/model_configs/scd.yaml'

    # args = {'model_config' : model_config_file,}
    frad_model = create_model({'model_config' : frad_config_file,})
    scd_model = create_model({'model_config' : scd_config_file,})
    et_model = create_model({'model_config' : et_config_file,})

    log = []
    log.append(print_model_param_counts(frad_model, "FRAD"))
    log.append(print_rep_param_count(frad_model, "FRAD"))
    log.append(print_model_param_counts(scd_model, "SCD"))
    log.append(print_rep_param_count(scd_model, "SCD"))
    log.append(print_model_param_counts(et_model, "ET"))
    log.append(print_rep_param_count(et_model, "ET"))
    
    et_results = analyze_model_performance(et_model, 
                                       test_loader, 
                                       device='cuda:0', 
                                       num_batches=32,  # Analyze 32 batches
                                       warmup_batches=5  # Skip first 5 batches for timing
                                       )

    pass



if __name__ == "__main__":
    main()