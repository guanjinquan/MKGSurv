import os
import sys
import glob
import h5py
import joblib
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# -------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP & IMPORTS
# -------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Code'))
from modules.base_modules.prototypes import cluster
from modules.base_modules.panther_module import StructuredPANTHER

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC"
CONF = {
    'h5_dir': f'{BASE_DIR}/h5_files',
    'csv_path': f'{BASE_DIR}/processed/kirc_patient_labels.csv',
    'folds_json': f'{BASE_DIR}/processed/kirc_patients_5fold.json',
    
    # 输出路径配置
    'output_dir': f'{BASE_DIR}/processed',
    'backup_dir': f'{BASE_DIR}/processed/backup',
    
    'n_proto': 128,              # Number of prototypes (p)
    'n_patches_sample': 10000000, # Number of patches to sample for clustering
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 2026,               # Consistent with JSON seed
    'cluster_mode': 'faiss',    # 'faiss' or 'kmeans'
    'h5_feature_key': 'features'
}

# -------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------------
def get_file_mapping(csv_path, h5_dir):
    """
    Maps Patient IDs to H5 files.
    Returns:
        pid_to_files: { case_id (UUID): [h5_paths] }
        sid_to_files: { submitter_id (e.g. TCGA-XX-XXXX): [h5_paths] }
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    all_h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))
    h5_lookup = {os.path.basename(f): f for f in all_h5_files}

    pid_to_files = {}
    sid_to_files = {}
    
    print("Mapping IDs to Files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row['cases.case_id']
        submitter_id = row['cases.submitter_id']
        
        # Match by checking if submitter_id is inside the filename
        matched_files = []
        for fname, fpath in h5_lookup.items():
            if submitter_id in fname:
                matched_files.append(fpath)
        
        if matched_files:
            if case_id in pid_to_files:
                pid_to_files[case_id].extend(matched_files)
            else:
                pid_to_files[case_id] = matched_files
                
            if submitter_id in sid_to_files:
                sid_to_files[submitter_id].extend(matched_files)
            else:
                sid_to_files[submitter_id] = matched_files

    print(f"Mapped {len(sid_to_files)} patients.")
    return pid_to_files, sid_to_files

def sample_patches(file_list, n_sample, feature_key='features'):
    """Samples patches uniformly from the provided file list."""
    print(f"Sampling {n_sample} patches from {len(file_list)} training files...")
    all_patches = []
    patches_collected = 0
    
    # Random shuffle
    shuffled_files = list(file_list)
    np.random.shuffle(shuffled_files)
    
    for fpath in tqdm(shuffled_files):
        if patches_collected >= n_sample:
            break
        try:
            with h5py.File(fpath, 'r') as f:
                if feature_key not in f: continue
                data = f[feature_key][:]
                if len(data) == 0: continue
                
                take_n = min(len(data), 1000) # Max 1000 per slide to ensure diversity
                indices = np.random.choice(len(data), take_n, replace=False)
                all_patches.append(data[indices])
                patches_collected += len(indices)
        except:
            continue

    if not all_patches:
        raise ValueError("No patches collected from training set!")

    combined = np.concatenate(all_patches, axis=0)
    if len(combined) > n_sample:
        indices = np.random.choice(len(combined), n_sample, replace=False)
        combined = combined[indices]
        
    return combined

def get_prototypes(fold_idx, train_files, conf):
    """
    Tries to load prototypes from backup. 
    If not found, generates them using ONLY training files and saves to backup.
    """
    os.makedirs(conf['backup_dir'], exist_ok=True)
    
    # Backup filename format: panther_fold{fold_idx}_{nP}.pkl
    backup_path = os.path.join(conf['backup_dir'], f"panther_fold{fold_idx}_{conf['n_proto']}.pkl")
    
    if os.path.exists(backup_path):
        print(f"\n[Fold {fold_idx}] Loading prototypes from backup: {backup_path}")
        prototypes = joblib.load(backup_path)
        # Simple validation
        if prototypes.shape[0] != conf['n_proto']:
            print(f"Warning: Loaded prototypes have shape {prototypes.shape}, expected {conf['n_proto']}. Re-generating.")
        else:
            return prototypes

    print(f"\n[Fold {fold_idx}] Generating new prototypes from {len(train_files)} training files...")
    
    # 1. Sample patches (Training Set Only!)
    patches = sample_patches(train_files, conf['n_patches_sample'], conf['h5_feature_key'])
    
    # 2. Cluster
    prototypes = cluster(
        patches=patches,
        n_proto=conf['n_proto'],
        mode=conf['cluster_mode'],
        n_proto_patches=conf['n_patches_sample']
    )
    
    # 3. Save backup
    print(f"Saving prototypes to {backup_path}")
    joblib.dump(prototypes, backup_path)
    
    return prototypes

def extract_features(model, file_mapping, device, h5_key):
    """
    Runs inference on ALL files per patient.
    CORRECTED LOGIC: Concatenates all patches from all slides of a patient BEFORE inference.
    """
    results = {}
    model.eval()
    
    # file_mapping key should be the Submitter ID (TCGA-XX-XXXX)
    with torch.no_grad():
        for pid, fpaths in tqdm(file_mapping.items(), desc="Extracting Features"):
            try:
                # 1. Collect ALL features from ALL slides for this patient
                all_slide_features = []
                for fpath in fpaths:
                    with h5py.File(fpath, 'r') as f:
                        if h5_key not in f: continue
                        features = f[h5_key][:] # Shape: (N_patches, D)
                        if len(features) > 0:
                            all_slide_features.append(features)
                
                if not all_slide_features:
                    print(f"Warning: No valid features found for {pid}")
                    continue

                # 2. Concatenate along the patch dimension to form one large bag
                # If slide 1 has N1 patches, slide 2 has N2 patches
                # combined_features shape: (N1 + N2 + ..., D)
                combined_features = np.concatenate(all_slide_features, axis=0)
                
                # 3. Prepare Input Tensor
                # Add batch dimension: (1, Total_N, D)
                x = torch.from_numpy(combined_features).float().unsqueeze(0).to(device)
                mask = torch.ones(1, x.shape[1]).to(device)
                
                # 4. Forward Pass (Once per patient)
                # Model output shape: (1, n_proto, 2*D + 1)
                embedding = model(x, mask)
                
                # 5. Save Result
                # Remove batch dimension -> (n_proto, 2*D + 1)
                results[pid] = embedding.squeeze(0).cpu().numpy()
                
            except Exception as e:
                print(f"Error processing {pid}: {e}")
                
    return results

# -------------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------------
def main():
    np.random.seed(CONF['seed'])
    torch.manual_seed(CONF['seed'])
    
    # 1. Prepare Mappings
    # pid_to_files: UUID -> [Paths]
    # sid_to_files: SubmitterID (TCGA-XX-XXXX) -> [Paths]
    pid_to_files, sid_to_files = get_file_mapping(CONF['csv_path'], CONF['h5_dir'])
    
    # 2. Load Folds JSON
    print(f"Loading Folds from: {CONF['folds_json']}")
    with open(CONF['folds_json'], 'r') as f:
        folds_data = json.load(f)
    
    # 3. Iterate Folds
    for fold_info in folds_data['folds']:
        fold_idx = fold_info['fold']
        train_sids = fold_info['train']
        
        print(f"\n{'='*20} Processing Fold {fold_idx} / {CONF['folds_json']} {'='*20}")
        
        # Identify Training Files for this Fold
        train_files = []
        missing_train = 0
        for sid in train_sids:
            if sid in sid_to_files:
                train_files.extend(sid_to_files[sid])
            else:
                missing_train += 1
        
        print(f"Fold {fold_idx}: Found {len(train_files)} training H5 files (Missing: {missing_train})")
        if len(train_files) == 0:
            print("Critical Error: No training files found for this fold. Skipping.")
            continue

        # -----------------------------------------------------------
        # Step A: Get Prototypes (Load Backup or Generate from Train)
        # -----------------------------------------------------------
        prototypes = get_prototypes(fold_idx, train_files, CONF)
        
        # -----------------------------------------------------------
        # Step B: Initialize Model with Fold-Specific Prototypes
        # -----------------------------------------------------------
        feature_dim = prototypes.shape[1]
        print(f"Initializing StructuredPANTHER (dim={feature_dim}, proto={CONF['n_proto']})...")
        
        model = StructuredPANTHER(
            in_dim=feature_dim,
            n_proto=CONF['n_proto'],
            prototypes=prototypes, 
            em_iter=3,
            tau=0.001,
            ot_eps=0.1,
            fix_proto=True
        ).to(CONF['device'])
        
        # -----------------------------------------------------------
        # Step C: Extract Features for ALL Patients (Train/Val/Test)
        # -----------------------------------------------------------
        # Pass sid_to_files (which contains LISTS of files)
        fold_features = extract_features(model, sid_to_files, CONF['device'], CONF['h5_feature_key'])
        
        # -----------------------------------------------------------
        # Step D: Save Fold Features
        # -----------------------------------------------------------
        save_path = os.path.join(CONF['output_dir'], f"features_image_pathology_fold{fold_idx}.pkl")
        print(f"Saving {len(fold_features)} patients' features to: {save_path}")
        joblib.dump(fold_features, save_path)
        
    print("\nAll folds processed successfully.")

if __name__ == "__main__":
    main()