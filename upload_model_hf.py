
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


    checkpoint_path = os.path.join(experiment_dir, "last.ckpt")
    config_path = os.path.join(experiment_dir, "input.yaml")
    
    # Upload to your repo
    upload_checkpoint_and_config(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        repo_name=repo_name,  # Change this
        private=True  # Set to True for private repo
    )
