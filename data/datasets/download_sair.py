
##### 
# This script exists to download, extract and preprocess the SAIR dataset from Hugging Face
# for simple use
# WARNING: The SAIR dataset contains over 2TB of data, after preprocessing results in ~1M files
# we suggest running this script on as a job on your chosen cluster.

import os
import tarfile
from huggingface_hub import hf_hub_url, list_repo_files, hf_hub_download
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

###### Download functions from SAIR creators #######
def load_sair_parquet(destination_dir: str) -> pd.DataFrame:
    """
    Downloads the sair.parquet file from the SandboxAQ/SAIR dataset and loads it
    into a pandas DataFrame.

    Args:
        destination_dir (str): The local path where the parquet file will be
                               downloaded. The directory will be created if it
                               doesn't exist.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the
                      sair.parquet file.
    """
    # --- 1. Setup and Repository Configuration ---
    repo_id = "SandboxAQ/SAIR"
    parquet_filename = "sair.parquet"

    print(f"Targeting repository: {repo_id}")
    print(f"Targeting file: {parquet_filename}")
    print(f"Destination directory: {destination_dir}")

    # Create the destination directory if it doesn't already exist
    os.makedirs(destination_dir, exist_ok=True)
    print(f"Ensured destination directory exists.")

    # --- 2. Download the Parquet file from the Hugging Face Hub ---
    download_path = os.path.join(destination_dir, parquet_filename)

    print(f"\nDownloading '{parquet_filename}'...")
    try:
        # Use hf_hub_download to get the file
        hf_hub_download(
            repo_id=repo_id,
            filename=parquet_filename,
            repo_type="dataset",
            local_dir=destination_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded to '{download_path}'")
    except Exception as e:
        print(f"An error occurred while downloading '{parquet_filename}': {e}")
        return None

   # --- 3. Load the Parquet file into a pandas DataFrame ---
    try:
        print(f"Loading '{parquet_filename}' into a pandas DataFrame...")
        df = pd.read_parquet(download_path)
        print("Successfully loaded DataFrame.")
        return df
    except Exception as e:
        print(f"Failed to load parquet file '{download_path}': {e}")
        return None


def download_and_extract_sair_structures(
    destination_dir: str,
    file_subset: list[str] = None,
    cleanup: bool = True
):
    """
    Downloads and extracts .tar.gz files from the SandboxAQ/SAIR dataset on Hugging Face.

    This function connects to the specified Hugging Face repository, identifies all
    .tar.gz files within the 'structures_compressed' directory, and downloads
    and extracts them to a local destination. It can download either all files
    or a specified subset.

    Args:
        destination_dir (str): The local path where the files will be downloaded
                               and extracted. The directory will be created if it
                               doesn't exist.
        file_subset (list[str], optional): A list of specific .tar.gz filenames
                                           to download. If None, all .tar.gz files
                                           in the directory will be downloaded.
                                           Defaults to None.
        cleanup (bool, optional): If True, the downloaded .tar.gz archive will be
                                  deleted after successful extraction. Defaults to True.

    Raises:
        ValueError: If any of the files specified in file_subset are not found
                    in the repository.
    """
    # --- 1. Setup and Repository Configuration ---
    repo_id = "SandboxAQ/SAIR"
    repo_folder = "structures_compressed"

    print(f"Targeting repository: {repo_id}")
    print(f"Destination directory: {destination_dir}")

    # Create the destination directory if it doesn't already exist
    os.makedirs(destination_dir, exist_ok=True)
    print(f"Ensured destination directory exists.")

    # --- 2. Get the list of relevant files from the Hugging Face Hub ---
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset")
        # Filter for files within the specified folder that are tar.gz archives
        repo_tars = [
            f.split('/')[-1] for f in all_files
            if f.startswith(repo_folder + '/') and f.endswith(".tar.gz")
        ]
        print(f"Found {len(repo_tars)} total .tar.gz files in '{repo_folder}'.")
    except Exception as e:
        print(f"Error: Could not list files from repository '{repo_id}'. Please check the name and your connection.")
        print(f"Details: {e}")
        return

    # --- 3. Determine which files to download ---
    if file_subset:
        # Validate that all requested files actually exist in the repository
        invalid_files = set(file_subset) - set(repo_tars)
        if invalid_files:
            raise ValueError(f"The following requested files were not found in the repository: {list(invalid_files)}")

        files_to_download = file_subset
        print(f"A subset of {len(files_to_download)} files was specified for download.")
    else:
        files_to_download = repo_tars
        print("No subset specified. All .tar.gz files will be downloaded.")

    # --- 4. Download and Extract each file ---
    for filename in tqdm(files_to_download, desc="Processing files"):
        # Construct the full path within the repository
        repo_filepath = f"{repo_folder}/{filename}"

        download_path = os.path.join(destination_dir, repo_filepath)

        print(f"\nDownloading '{filename}'...")
        try:
            # Download the file from the Hub
            hf_hub_download(
                repo_id=repo_id,
                filename=repo_filepath,
                repo_type="dataset",
                local_dir=destination_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Successfully downloaded to '{download_path}'")

            # Extract the downloaded .tar.gz file
            print(f"Extracting '{filename}'...")
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=destination_dir)
            print(f"Successfully extracted contents to '{destination_dir}'")

        except Exception as e:
            print(f"An error occurred while processing '{filename}': {e}")
            continue

        finally:
            # Clean up the downloaded archive if the flag is set and the file exists
            if cleanup and os.path.exists(download_path):
                os.remove(download_path)
                print(f"Cleaned up (deleted) '{download_path}'")

    print("\nOperation completed.")


################ custom helper functions ################
import multiprocessing
from time import time
from multiprocessing import Pool

def get_available_tars():
    repo_id = "SandboxAQ/SAIR"
    repo_folder = "structures_compressed"

    all_files = list_repo_files(repo_id, repo_type="dataset")
    # Filter for files within the specified folder that are tar.gz archives
    repo_tars = [
        f.split('/')[-1] for f in all_files
        if f.startswith(repo_folder + '/') and f.endswith(".tar.gz")
    ]
    return repo_tars

def get_downloaded_tars(output_directory):
    download_cach_dir = os.path.join(output_directory, '.cache/huggingface/download/structures_compressed')
    assert os.path.exists(download_cach_dir), f"Download cache directory does not exist: {download_cach_dir}"

    all_cached_files = os.listdir(download_cach_dir)

    #remove terminal .lock from file names
    check_ext = '.lock'
    all_cached_files = [os.path.splitext(f)[0] for f in all_cached_files if not f.endswith(check_ext)]

    return all_cached_files

def get_remaining_tars(output_directory):
    avail_chunks = get_available_tars()
    finished_chunks = get_downloaded_tars(output_directory=output_directory)

    remaining_chunks = list(set(avail_chunks) - set(finished_chunks))
    #sort remaining chunks
    remaining_chunks.sort()
    return remaining_chunks


def mp_download_and_extract_tar(tar_files, output_directory):
    if not isinstance(tar_files, list):
        assert isinstance(tar_files, str), "tar_files must be a string or a list of strings"
        tar_files = [tar_files]

    #track and report time taken for each job - then report it an average at end
    job_timer = time()
    ################################################################################
    download_and_extract_sair_structures(destination_dir=output_directory, file_subset=tar_files)
    ################################################################################
    end_job_timer = time()
    elapsed = end_job_timer - job_timer
    
    print(f"\nfinished \t{tar_files}: \t{elapsed} seconds")
    #return time taken for possible averaging later
    return elapsed

def multiprocess_downlaod(output_dir, max_files=None, max_workers=None):
    # Get system info
    available_cpus = multiprocessing.cpu_count()
    remaining_tars = get_remaining_tars(output_directory=output_dir)
    print(f"Total remaining files to download: {len(remaining_tars)}")
    if max_files is not None:
        remaining_tars = remaining_tars[:max_files]

    # For download + extraction: optimize for number of files
    num_workers = min(len(remaining_tars), available_cpus // 2)  # Leave some CPUs for system
    
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)
    
    print(f"Available CPUs: {available_cpus}")
    print(f"Files to download: {len(remaining_tars)}")
    print(f"Using {num_workers} workers")

    if num_workers == 0:
        print("No files remaining to download.")
        return

    process_start_timer = time()

    with Pool(num_workers) as p:
        results = p.starmap(mp_download_and_extract_tar, [(tar_file, output_dir) for tar_file in remaining_tars])

    process_end_timer = time()
    total_elapsed = process_end_timer - process_start_timer
    total_elapsed_mins = total_elapsed / 60
    
    # Calculate average time
    avg_time = sum(results) / len(results) if results else 0
    min_time = min(results) if results else 0
    max_time = max(results) if results else 0
    # print(f"Average time per file: {avg_time:.2f} seconds")
    # print(f"Minimum time per file: {min_time:.2f} seconds")
    # print(f"Maximum time per file: {max_time:.2f} seconds")
    #convert to minutes
    total_time_mins = sum(results) / 60 if results else 0
    print("\n--- COMPLETED ---")
    print(f"Total time: {total_elapsed_mins:.3f} minutes")
    print(f"mp time: {total_time_mins:.3f} minutes")
    print("\n--- Download + Extraction Time Statistics ---")
    print(f"Average time per file: {avg_time/60:.3f} minutes")
    print(f"Minimum time per file: {min_time/60:.3f} minutes")
    print(f"Maximum time per file: {max_time/60:.3f} minutes")

###### Post extracttion preprocessing  ######
from Bio.PDB import MMCIFParser
import pandas as pd
from StructureCloud.chem_tools.atom_properties import ATOMIC_NUMBERS
import torch
import numpy as np

from torch_geometric.data import Data

#convert all keys to lowercase
symbol_2_num = {k.lower(): v for k, v in ATOMIC_NUMBERS.items()}

AA_to_num = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    'LIG': 20, # special code for ligand residues
    'UNK': 21 # unknown residue
}

def extract_cif(file_path) -> dict:
    ''' extract relevant data from a cif file into a numerical dictionary '''

    # Parse the CIF file into nnumerical dictionary
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)

    unknown_elem_val = 0 # Use 0 for unknown atoms

    z = []
    pos = []
    atom_chain_id = []
    atom_res_idx = []
    atom_is_ligand = []
    res_name = []
    res_num = []
    res_pos = []
    res_chain_id = []

    for model in structure:
        for cid, chain in enumerate(model.get_chains()): #one chain is protein the other is ligand
            # print(f" Processing chain: {chain.id}")
            for ridx, res in enumerate(chain.get_residues()):
               
                res_is_lig = res.resname == 'LIG'
                if res_is_lig:
                    ridx = len(z)  # assign unique residue index for ligand residues
                atoms = list(res.get_atoms())

                # print(f"  Processing residue: {res.resname} {res.id}, number of atoms: {len(atoms)}, is_ligand: {res_is_lig}")
                # print(res.resname, res.id[1])
                # elements = np.array([symbol_2_num[atom.element.lower()] for atom in atoms], dtype=np.int32)
                elements = np.array([symbol_2_num.get(atom.element.lower(), unknown_elem_val) for atom in atoms], dtype=np.int32)
                coords = np.array([atom.coord for atom in atoms], dtype=np.float32)
                lig_mask = np.ones((len(atoms),), dtype=np.int32) if res_is_lig else np.zeros((len(atoms),), dtype=np.int32)
                res_idx = np.full((len(atoms),), ridx, dtype=np.int32)

                #stack atomwise data
                z.append(elements)
                pos.append(coords)
                atom_chain_id.extend([chain.id]*len(atoms))
                atom_res_idx.extend(res_idx)
                atom_is_ligand.extend(lig_mask.tolist())
                
                rname = res.resname
                rnum = AA_to_num.get(res.resname)
                rpos = res.center_of_mass()
                rchain = chain.id

                #stack residuewise data
                res_name.append(rname)
                res_num.append(rnum)
                res_pos.append(rpos)
                res_chain_id.append(rchain)

    pos = np.vstack(pos).astype(np.float32)  # [N, 3]
    z = np.hstack(z).astype(np.int32)        # [N, ]
    atom_chain_id = np.array(atom_chain_id)  # [N, ]
    atom_res_idx = np.array(atom_res_idx, dtype=np.int32)  # [N, ]
    atom_is_ligand = np.array(atom_is_ligand, dtype=np.bool)  # [N, ]

    res_pos = np.array(res_pos, dtype=np.float32)  # [R, 3]
    res_num = np.array(res_num, dtype=np.int32)    # [R, ]
    res_name = np.array(res_name)                   # [R, ]
    res_chain_id = np.array(res_chain_id)           # [R, ]
    res_is_ligand = np.array([1 if name == 'LIG' else 0 for name in res_name], dtype=np.bool)  # [R, ]

    data = {
        'pos' : pos, # all positions of atoms [N, 3]
        'z' : z,     # atomic numbers of atoms [N, ]
        'atom_chain_id' : atom_chain_id, # chain IDs of atoms [N, ]
        'atom_res_idx' : atom_res_idx, # indexing of which residue each atom belongs to [N, ]
        'atom_is_ligand' : atom_is_ligand, # ligand atom indices, binary mask [N, ]
        'res_name' : res_name, # residue names [R, str]
        'res_num' : res_num, # residue types as numbers [R, ]
        'res_pos' : res_pos, # residue centroid positions [R, 3]
        'res_chain_id' : res_chain_id, # chain IDs of residues [R, ]
        'res_is_ligand' : res_is_ligand # ligand residue indices, binary mask [R, ]
    }

    #format for torch.save with weights_only=True
    for k in data.keys():
        if isinstance(data[k], np.ndarray):
            # Check if it's a string array
            if data[k].dtype.char == 'U':  # Unicode strings
                # Convert to list for weights_only=True compatibility
                data[k] = data[k].tolist()
            else:
                # Convert numeric arrays to tensors
                data[k] = torch.from_numpy(data[k])


    return data

def sample_name(id, n):
    return f'sample_{id}_model_{n}.cif'

def get_raw_confs(id, source_dir, max_confs=5):
    ''' get all raw conformations for a given sample id from source directory '''
    all_conf_data = {}
    file_paths = []
    for i in range(max_confs):
        path = os.path.join(source_dir, sample_name(id, i))
        if not os.path.exists(path):
            print(f"Sample {id}, conf {i} not found: {path}")
            continue
        data = extract_cif(path)
        all_conf_data[i] = data
        file_paths.append(path)
        
    return all_conf_data, file_paths

def preprocess_sample(id, input_dir, output_dir, max_confs=5):
    ''' preprocess all conformations for a given sample id and save to output directory 
    
    args:
        id: sample id (int)
        input_dir: directory containing raw cif files (str)
        output_dir: directory to save processed sample files (str)
        max_confs: maximum number of conformations to process per sample (int)

    '''

    out_name = f'sample_{id}.pt'
    out_path = os.path.join(output_dir, out_name)

    print(f"--- Sample {id}: Starting preprocessing:")

    #check if output file already exists
    if os.path.exists(out_path):
        sample_data = torch.load(out_path, map_location='cpu', weights_only=True) # a dictionary of dicts with sample data
    
        saved_confs = list(sample_data.keys())
        if len(saved_confs) == max_confs:
            print(f"Sample {id}: Max conformations ({max_confs}) already processed for sample {id}. Skipping.")
            return
        else:
            print(f"Sample {id}: Adding to existing data. {len(saved_confs)} conformations found.")
    else:
        sample_data = {}
        saved_confs = []

    raw_samples, source_paths = get_raw_confs(id, input_dir, max_confs=max_confs)

    for key in raw_samples.keys():
        if key in saved_confs:
            continue
        sample_data[key] = raw_samples[key]

    torch.save(sample_data, out_path)
    print(f"Sample {id}: Saved to {out_path}")
    
    try:#test load saved sample to verify
        loaded_sample = torch.load(out_path, map_location='cpu', weights_only=True)
        print(f"Sample {id}: loading loaded verified! number of conformations: {len(loaded_sample.keys())}")
    except Exception as e:
        print(f"Error loading sample {id}: {e}")
        return

    # delete files at source paths to save space
    for path in source_paths:
        os.remove(path)
        print(f"Deleted source file: {path}")
    
    print(f"--- Sample {id}: COMPLETE")

def mp_preprocess_samples(source_dir, output_dir, dump_dir_name='processed', max_confs=5, max_workers=None, max_samples=None):
    ''' multiprocessing preprocessing of all samples in source directory and save to output directory
    
    NOTE: this will take any downloaded/extracted sair cif files and preproces them into 
    dictionary files containing tensor numerical data. Afterwords the original cif files are deleted.
    this is done to reduce the total file count and speed up future loading.
    
    args:
        source_dir: directory containing raw cif files (str)
        output_dir: directory to create a new folder and save processed sample files (str)
        dump_dir_name (str : 'processed'): name of folder to create in output_dir to save processed files (str)
        max_confs (int : 5): maximum number of conformations to process per sample (int). SAIR provides 5 conformations per sample.
        max_workers (int : None): maximum number of parallel workers to use (int). If None, will use half of available CPUs.
        max_samples (int : None): maximum number of samples to process (int). If None, will process all samples.
    '''

    #create output directory if it doesn't exist
    processed_dir = os.path.join(output_dir, dump_dir_name)
    os.makedirs(processed_dir, exist_ok=True)
    
    #gather all sample ids from structure files
    struc_files = [f for f in os.listdir(source_dir) if f.endswith('.cif')]
    sample_ids = list(set([int(f.split('_')[1]) for f in struc_files]))
    sample_ids.sort()  #sort sample ids

    print(f'Found {len(sample_ids)} unique sample IDs for preprocessing.')
    if len(sample_ids) == 0:
        print("No samples found for preprocessing.")
        return

    if max_samples is not None:
        sample_ids = sample_ids[0:max_samples]

    print(f'Starting multiprocessing preprocessing of {len(sample_ids)} samples...')

    available_cpus = multiprocessing.cpu_count()
    num_workers = min(len(sample_ids), available_cpus // 2)  # Leave some CPUs for system
    
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)
    
    print(f"Available CPUs: {available_cpus}")
    print(f"Samples to process: {len(sample_ids)}")
    print(f"Using {num_workers} workers")

    if num_workers == 0:
        print("No samples to process.")
        return
    
    start_time = time()
    with Pool(num_workers) as p:
        p.starmap(preprocess_sample, [(sid, source_dir, processed_dir, max_confs) for sid in sample_ids])
    end_time = time()
    print("Multiprocessing preprocessing complete.")
    elapsed_mins = (end_time - start_time) / 60.0
    print(f"Total preprocessing time: {elapsed_mins:.3f} minutes")
    seconds_per_sample = (end_time - start_time) / len(sample_ids)
    print(f"Average time per sample: {seconds_per_sample:.3f} seconds")

if __name__ == '__main__':
    # --- Download the parquet dataset ---

    #print a warning about data size
    print("\n--- WARNING: The SAIR dataset is very large (>2TB). Ensure you have sufficient storage space before proceeding. ---\n")
    print("--- to download and preprocess the full dataset, you must uncomment the relevant sections below and run this script as a job on your cluster. ---\n")

    # Define a destination for the data
    output_directory = "/pscratch/sd/t/tyjperez/scd_dump/tmp/sair_data"

    # Call the function to download and load the parquet data
    sair_df = load_sair_parquet(destination_dir=output_directory)

    # # Check if the DataFrame was loaded successfully
    # if sair_df is not None:
    #     print("\n--- DataFrame Info ---")
    #     sair_df.info()

    #     print("\n--- DataFrame Head ---")
    #     print(sair_df.head())

    # --- Download a specific subset of structure tarballs ---
    # print("--- Running Scenario 2: Download a specific subset ---")
    # Define the specific files you want to download
    # Replace this with None to download *all* structures
    # (remember, this is >100 files of ~10GB each!)
    # subset_to_get = [
    #     "sair_structures_0_to_10565.tar.gz"
    #     # "sair_structures_1006049_to_1016517.tar.gz", # finished
    #     # "sair_structures_100623_to_111511.tar.gz",
    # ]
    # download_and_extract_sair_structures(destination_dir=output_directory, file_subset=subset_to_get)

    # rem_tars = get_remaining_tars(output_directory)
    # print(f"Remaining tars to download: {len(rem_tars)}")

    ### multiprocessing download + extraction ###
    # print("--- Running Multiprocessing Download + Extraction ---")
    # multiprocess_downlaod(output_dir=output_directory, max_files=25)

    ### after download, preprocess samples ###
    # print("\n--- Running Multiprocessing Preprocessing ---")
    # source_dir = os.path.join(output_directory, 'structures')
    # output_dir = output_directory
    # mp_preprocess_samples(source_dir, output_dir, max_samples=200000)
