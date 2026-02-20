
from huggingface_hub import HfApi
import os

def upload_checkpoint_and_config(
    checkpoint_path: str,
    config_path: str, 
    repo_name: str,
    private: bool = False
):
    """Upload raw checkpoint and config files to HF Hub"""
    
    api = HfApi()
    
    # Create repository
    try:
        api.create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private
        )
        print(f"Created repository: {repo_name}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload checkpoint file
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="last.ckpt",
        repo_id=repo_name,
        commit_message="Upload model checkpoint"
    )
    print("Uploaded last.ckpt")
    
    # Upload config file
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="input.yaml", 
        repo_id=repo_name,
        commit_message="Upload model config"
    )
    print("Uploaded input.yaml")

if __name__ == "__main__":

    # scdSm-n04_pcq

    # Set your paths
    # repo_name = "Ty-Perez/CET-GEOM01_0"  # Change this
    # experiment_dir = "experiments/scd-n004_geom01_b48n4"

    # repo_name = "Ty-Perez/CET-GEOM10_0"  # Change this
    # experiment_dir = "experiments/scd-n004_geom_b48n4"

    # repo_name = "Ty-Perez/cget-amp20"  # Change this
    # experiment_dir = "experiments/cget-amp20_b32n2_0"

    #### New Set for public use:

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-pcq'
    # experiment_dir = "experiments/scd-s-n004_pcq"

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-geom10'
    # experiment_dir = "experiments/scd-n004_geom_b48n4"

    # ##DONE
    # repo_name = 'Ty-Perez/ct-scd-amp20'
    # experiment_dir = "experiments/cet-amp20_b142n1_0"

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-all'
    # experiment_dir = "experiments/cet-all_b56n4_2"

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-sair'
    # experiment_dir = "experiments/cet-sair_b12n2_0"

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-sair-pocket'
    # experiment_dir = "experiments/cet-sairpocket_b12n2_0"

    ##DONE
    # repo_name = 'Ty-Perez/ct-scd-omol25'
    # experiment_dir = "experiments/ct-scd-omol25_4n"

    ##DONE
    # repo_name = 'Ty-Perez/ct-fe-omol25'
    # experiment_dir = "experiments/null_FT_s2ef_n4_800ksteps_fwdF"

    ##DONE
    # repo_name = 'Ty-Perez/cgt-scd-pcq'
    # experiment_dir = "experiments/scd-frad-n004_pcq_b32n4"
    # model_name=scd-frad-n004_pcq_b32n4 # frad pretrained


    checkpoint_path = os.path.join(experiment_dir, "last.ckpt")
    # experiment_dir = "experiments/scdSs-n004_pcq"
    # checkpoint_path = os.path.join(experiment_dir, "step=405503-epoch=143-val_loss=0.0935-test_loss=0.2179-train_per_step=0.1139.ckpt")
    config_path = os.path.join(experiment_dir, "input.yaml")
    
    # Upload to your repo
    upload_checkpoint_and_config(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        repo_name=repo_name,  # Change this
        private=True  # Set to True for private repo
    )