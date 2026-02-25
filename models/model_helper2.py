
#### Helper functions to create, load and restart
### models for training
###

import yaml
import torch
import warnings
import re
import os
import sys

### MUST IMPORT ALL POSSIBLE MODEL ARCHITECTURE CLASSES TO BE CALLABLE
### BY CREATE
from models.ET_models.scd_model import ET, CET, CFrad, ETFrad
from models.ET_models import priors

from copy import deepcopy

def create_prior_models(args, dataset=None):
    """Parse the prior_model configuration option and create the prior models.

    The information can be passed in different ways via the args dictionary, which must contain at least the key "prior_model".

    1. A single prior model name and its arguments as a dictionary:

    .. code:: python

      args = {
          "prior_model": "Atomref",
          "prior_args": {"max_z": 100}
      }


    2. A list of prior model names and their arguments as a list of dictionaries:

    .. code:: python

      args = {
          "prior_model": ["Atomref", "D2"],
          "prior_args": [{"max_z": 100}, {"max_z": 100}]
      }


    3. A list of prior model names and their arguments as a dictionary:

    .. code:: python

      args = {
          "prior_model": [{"Atomref": {"max_z": 100}}, {"D2": {"max_z": 100}}]
      }


    Args:
        args (dict): Arguments for the model.
        dataset (torch_geometric.data.Dataset, optional): A dataset from which to extract the atomref values. Defaults to None.

    Returns:
        list: A list of prior models.

    """
    prior_models = []
    if args["prior_model"]:
        prior_model = args["prior_model"]
        prior_names = []
        prior_args = []
        if not isinstance(prior_model, list):
            prior_model = [prior_model]
        for prior in prior_model:
            if isinstance(prior, dict):
                for key, value in prior.items():
                    prior_names.append(key)
                    if value is None:
                        prior_args.append({})
                    else:
                        prior_args.append(value)
            else:
                prior_names.append(prior)
                prior_args.append({})
        if "prior_args" in args and args["prior_args"] is not None:
            prior_args = args["prior_args"]
            if not isinstance(prior_args, list):
                prior_args = [prior_args]
        for name, arg in zip(prior_names, prior_args):
            print(f"Creating prior model: {name} with args {arg}")
            assert hasattr(priors, name), (
                f"Unknown prior model {name}. "
                f"Available models are {', '.join(priors.__all__)}"
            )
            # initialize the prior model
            prior_models.append(getattr(priors, name)(dataset=dataset, **arg))
    return prior_models

def create_model(args, prior_model=None, mean=None, std=None):
    model_config_file = args.get("model_config", None)

    #load config yaml and parse into a dictionary
    if model_config_file is not None:
        with open(model_config_file, "r") as f:
            model_config = yaml.safe_load(f)

        #for any key overlap, use the keys from args
        for k, v in args.items():
            all_keys = model_config.keys()
            if k in all_keys:
                model_config[k] = v

        args.update(model_config)
    else:
        #raise error
        raise ValueError(f"Model config file not found: {model_config_file}")

    #manually set/reset some configs for model
    if args.get('set_head_agg', None) is not None:
        model_config['head_agg'] = args['set_head_agg']

    model_config['mean'] = mean
    model_config['std'] = std

    #check if args contains prior model information
    if args.get('prior_model', None) is not None:
        #TODO make this more robust but switching the check to weaither prior_model is a prior class object
        if isinstance(args['prior_model'], str) or isinstance(args['prior_model'], dict):
            model_config['prior_model'] = create_prior_models(args)
        else:
            #assume prior_model a model object, rather than inscriuctions
            model_config['prior_model'] = args['prior_model']

    #check if args contdains "derivative" key
    # if args.get('derivative', None) is not None:
    #     assert isinstance(args['derivative'], bool), " 'derivative' key must be a boolean"
    #     print(f'---- adjusting model config: derivative ={args.derivative}')
    #     model_config['derivative'] = args['derivative']

    # model_config['prior_model'] = args.prior_model

    # model_config['prior_model'] = prior_model
    #pop off the model architecture key: 'model'
    model_arc = model_config.pop("model", None)
    assert model_arc is not None, "Model architecture not specified. please include a 'model' input"
    model_class = globals().get(model_arc, None)
    assert model_class is not None, f"Model class \"{model_arc}\" not found. Please check the model architecture name."
    model = model_class(**model_config)

    return model


def get_model_checkpoint(model_dir, epoch=-1):
    ### given a directory parse out the checkpoint files, epoch and step

    files_in_dir = os.listdir(model_dir)
    checkpoint_files = [f for f in files_in_dir if f.endswith('.ckpt')]
    if len(checkpoint_files) == 0:
        return None, None, None # in the case that there are no checkpoints found
    
    #TODO: update this to be robust to multiple last-*.ckpt files
    last_ckpt_files = [f for f in checkpoint_files if f.startswith('last')]
    # last_ckpt_files = [checkpoint_files.pop(i) for i in range(len(checkpoint_files)) if checkpoint_files[i].startswith('last')]

    #remove all last_ckpt_files from checkpoint_files
    checkpoint_files = [f for f in checkpoint_files if f not in last_ckpt_files]

    # print('last:', last_ckpt_files)
    # print('other ckpts:', checkpoint_files)
    # for f in checkpoint_files:
    #     if f.startswith('last') and f != 'last.ckpt':
    #         print(f"WARNING: Found non-standard last checkpoint file: {f}. This will be ignored.")

    # if 'last.ckpt' in checkpoint_files:
    #     last_checkpoint = checkpoint_files.pop(checkpoint_files.index('last.ckpt'))
    if 'last.ckpt' in last_ckpt_files:
        last_checkpoint = last_ckpt_files[last_ckpt_files.index('last.ckpt')]
    else:
        last_checkpoint = None
        print("WARNING: No last.ckpt file found in model directory!")

    def str_to_num(string):
        #parse a string of digits into either an int or float
        if '.' in string:
            return float(string)
        return int(string)

    #parse the epoch from the checkpoint file name
    checkpoint_details = []
    for ckp_f in checkpoint_files:
        #drop extension 
        ckp_f, ext = os.path.splitext(ckp_f)
        ckp_f = ckp_f.split('-')
        ckp_dict = {}
        for entry in ckp_f:
            # print(entry)
            k, v = entry.split('=')
            ckp_dict[k] = str_to_num(v)
        checkpoint_details.append(ckp_dict)
        # print(ckp_dict)

    checkpoint_epoch = [int(d['epoch']) for d in checkpoint_details]
    checkpoint_step = [int(d['step']) for d in checkpoint_details]
    epoch = int(epoch)
    # print(f'given ep {epoch}, checkpoint epochs: {checkpoint_epoch}')
    if epoch == -1:
        # Select last epoch
        if last_checkpoint is not None:
            print(f"Loading last checkpoint: {last_checkpoint}")
            checkpoint_path = os.path.join(model_dir, last_checkpoint)

            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_epoch = ckpt["epoch"]
            ckpt_step = ckpt["global_step"]

            return checkpoint_path, ckpt_step, ckpt_epoch


        #choose largest number in checkpoint_epoch
        checkpoint_ep = max(checkpoint_epoch)
        checkpoint_index = checkpoint_epoch.index(checkpoint_ep)
        checkpoint_step = checkpoint_step[checkpoint_index]
        checkpoint_path = os.path.join(model_dir, last_checkpoint)
        epoch = checkpoint_ep 
        # print(f'epoch set {epoch}')

    elif epoch in checkpoint_epoch:
        checkpoint_index = checkpoint_epoch.index(epoch)
        checkpoint_step = checkpoint_step[checkpoint_index]
        checkpoint_path = os.path.join(model_dir, checkpoint_files[checkpoint_index])
    else:
        print(f"Epoch {epoch} not found in available checkpoints: {checkpoint_epoch}")
        print("Available epochs:", sorted(set(checkpoint_epoch)))
        # Handle the error - could return None, use latest, or raise exception
        raise ValueError(f"Epoch {epoch} not found")

    #assert that checkpoint_path exists
    assert os.path.exists(checkpoint_path), f"Checkpoint path does not exist: {checkpoint_path}, epoch {epoch}"
    
    return checkpoint_path, checkpoint_step, epoch

def exists(obj):
    return obj is not None

def is_not_null(obj):
    #check if obj is not None and not the string "None", 'null' or any case variant
    return obj is not None and str(obj).lower() not in ['none', 'null']


def load_model(filepath, args=None, device="cpu",  mean=None, std=None, **kwargs):

    # if restart:
    #     checkpoint_path, _step, _epoch = get_model_checkpoint(filepath, epoch=restart_epoch)
    #     print(f"Restarting from checkpoint: {checkpoint_path}, epoch: {restart_epoch}, step: {_step}")
    #     ckpt = torch.load(checkpoint_path, map_location="cpu")
    #     ckpt_epoch = ckpt["epoch"]
    #     ckpt_step = ckpt["global_step"]

    #     assert ckpt_epoch == _epoch, f"Checkpoint epoch {ckpt_epoch} does not match requested restart epoch {restart_epoch}"
    #     assert ckpt_step == _step, f"Checkpoint step {ckpt_step} does not match expected step {_step}"

    # else:
    #     ckpt = torch.load(filepath, map_location="cpu")

    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
    
    
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)
    ema_model = None

    ema_state_dict = {}
    ckpt_state_dict = {}
    #pop off all keys from the ckpt["state_dict"] that start with "ema_model." 
    for k, v in ckpt["state_dict"].items():
        if k.startswith("ema_model."):
            ema_state_dict[k[len("ema_model."):]] = v
        else:
            ckpt_state_dict[k] = v

    # state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt_state_dict.items()}

    has_ema_model = len(ema_state_dict) > 0

    def load_weights(mod, state):
        # loading_return = model.load_state_dict(state_dict, strict=False)
        loading_return = mod.load_state_dict(state, strict=False)
    
        if len(loading_return.unexpected_keys) > 0:
            # Should only happen if not applying denoising during fine-tuning.
            print(f"WARNING: Unexpected keys in state_dict:\n{loading_return.unexpected_keys}")
            # print(loading_return.unexpected_keys)
            # assert all(("output_model_noise" in k or "pos_normalizer" in k) for k in loading_return.unexpected_keys)
        
        if len(loading_return.missing_keys) > 0:
            #check iff all missing keys are from the prior model
            non_prior_missing = [k for k in loading_return.missing_keys if not k.startswith('prior_model.')]
            if len(non_prior_missing) > 0:
                assert False, f"Missing keys not in prior model: {non_prior_missing}"
            
            # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    def check_arch(mod):
        # expect either bool or str arguments 
        
        # Implement architecture alterations here
        if exists(args.get('reset_head', None)):
            if isinstance(args['reset_head'], bool):
                if args['reset_head']:
                    print('Resetting model head')
                    mod.reset_head()
            elif isinstance(args['reset_head'], str):
                if is_not_null(args['reset_head']):
                    print(f'Resetting model head: {args["reset_head"]}')
                    mod.reset_head(args['reset_head'])

        if exists(args.get('reset_embeddings', None)):
            if isinstance(args['reset_embeddings'], bool):
                if args['reset_embeddings']:
                    print('Resetting model embeddings')
                    mod.reset_embeddings()
            elif isinstance(args['reset_embeddings'], str):
                if is_not_null(args['reset_embeddings']):
                    print(f'Resetting model embeddings: {args["reset_embeddings"]}')
                    mod.reset_embeddings(args['reset_embeddings'])

        if exists(args.get('reset_norms', None)):
            if isinstance(args['reset_norms'], bool):
                if args['reset_norms']:
                    print('---!!!Resetting model norms!!!---')
                    mod.reset_norms()
            elif isinstance(args['reset_norms'], str):
                if is_not_null(args['reset_norms']):
                    print(f'--- !!!Resetting model norms: {args["reset_norms"]}, type: {type(args["reset_norms"])}!!! ---')
                    mod.reset_norms(args['reset_norms'])


    if exists(args.get('load_model', None)):
    # if exists(args.load_model):
        print('Loading weights from restart checkpoint')
        #if loading from a reset then adjust arch before loading weights
        check_arch(model)
        load_weights(model, state_dict)

        if has_ema_model:
            ema_model = deepcopy(model)
            load_weights(ema_model, ema_state_dict)
            ema_model.to(device)
    else:
        print('Loading weights from pretrained model')
        #otherwise if loading from a pretrained model then load weights then adjust arch
        load_weights(model, state_dict)

        if has_ema_model:
            ema_model = deepcopy(model)
            load_weights(ema_model, ema_state_dict)
            check_arch(ema_model)
            ema_model.to(device)

        check_arch(model)
    
    
    if mean:
        model.mean = mean
        if has_ema_model:
            ema_model.mean = mean
    if std:
        model.std = std
        if has_ema_model:
            ema_model.std = std

        print('--- MODEL SUCESSFULLY LOADED ---')

    return model.to(device), ema_model, ckpt