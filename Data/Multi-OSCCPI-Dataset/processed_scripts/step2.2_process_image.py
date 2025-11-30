import os
import sys
import json
import glob
import random
import joblib
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

# -------------------------------------------------------------------------
# 1. ENVIRONMENT & IMPORTS
# -------------------------------------------------------------------------
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    print("Error: Missing libraries for UNI model.")
    print("Please install them using: pip install timm huggingface_hub")
    sys.exit(1)

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Code'))

try:
    from modules.base_modules.prototypes import cluster
    from modules.base_modules.panther_module import StructuredPANTHER
except ImportError as e:
    print("Error: Could not import custom modules (PANTHER/Cluster).")
    print("Please check your sys.path and directory structure.")
    print(f"Details: {e}")
    sys.exit(1)

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
CONF = {
    'input_root': "/home/Guanjq/NewWork/PathoBackup/Multi-OSCCPI-Dataset/Multi-OSCCPI-Images",
    # Changed to the 5-fold OOD split file
    'split_json': "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/split_OOD_5fold.json",
    
    # 这里的 backup_dir 是根目录，代码会自动在下面创建/查找 h5_files 子文件夹
    'backup_dir': "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/backup",
    
    # 最终结果输出路径
    'output_dir': "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed",
    
    # UNI 本地权重文件夹路径
    'uni_weight_path': "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/UNI",

    'patch_size': 256,        
    'model_img_size': 224,    # UNI 标准输入大小
    'batch_size': 64,         # 显存允许的话可以调大
    'n_proto': 128,           
    'n_patches_sample': 2000000, 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 2026,
    'cluster_mode': 'faiss'
}

# -------------------------------------------------------------------------
# 3. HELPER CLASS: FEATURE EXTRACTOR (UNI v1)
# -------------------------------------------------------------------------
class UNIFeatureExtractor(nn.Module):
    def __init__(self, local_path=None):
        super(UNIFeatureExtractor, self).__init__()
        
        model_path = local_path if local_path else CONF.get('uni_weight_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Local UNI path not found: {model_path}")
            
        print(f"Loading UNI model from local path: {model_path}")

        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            candidates = glob.glob(os.path.join(model_path, "*.bin"))
            if candidates:
                bin_path = candidates[0]
                print(f"Warning: 'pytorch_model.bin' not found, using found candidate: {bin_path}")
            else:
                raise FileNotFoundError(f"No checkpoint file (.bin) found in {model_path}")

        print(f"Loading weights from {bin_path} ...")
        state_dict = torch.load(bin_path, map_location="cpu")
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        msg = self.model.load_state_dict(state_dict, strict=True)
        print(f"Weights loaded successfully. {msg}")

    def forward(self, x):
        return self.model(x)

def get_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# -------------------------------------------------------------------------
# 4. HELPER FUNCTIONS: DATA PROCESSING
# -------------------------------------------------------------------------

def get_patches_from_image(img_path, patch_size=256):
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return []

    if w < patch_size or h < patch_size:
        return []

    patches = []
    x_steps = list(range(0, w - patch_size + 1, patch_size))
    if w % patch_size != 0:
        x_steps.append(w - patch_size) # Overlap
    y_steps = list(range(0, h - patch_size + 1, patch_size))
    if h % patch_size != 0:
        y_steps.append(h - patch_size)

    for y in y_steps:
        for x in x_steps:
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches

def get_patient_image_paths(input_root, patient_id):
    patient_path = os.path.join(input_root, str(patient_id))
    image_names = [
        "01_2X.jpg", "01_4X.jpg", "01_10X.jpg",
        "02_2X.jpg", "02_4X.jpg", "02_10X.jpg"
    ]
    exist_paths = []
    for name in image_names:
        img_path = os.path.join(patient_path, name)
        if os.path.exists(img_path):
            exist_paths.append(img_path)
    if not exist_paths:
        return []

    final_image_paths = []
    for idx, name in enumerate(image_names):
        img_path = os.path.join(patient_path, name)
        if not os.path.exists(img_path):
            fallback_name = image_names[(idx + 3) % 6]
            fallback_path = os.path.join(patient_path, fallback_name)
            if os.path.exists(fallback_path):
                final_image_paths.append(fallback_path)
            else:
                final_image_paths.append(random.choice(exist_paths))
        else:
            final_image_paths.append(img_path)
    return final_image_paths

# 保存特征到 backup/h5_files 子目录
def save_patient_features(pid, features, backup_dir):
    if len(features) == 0:
        return
    
    h5_subdir = os.path.join(backup_dir, 'h5_files')
    os.makedirs(h5_subdir, exist_ok=True)
    
    h5_path = os.path.join(h5_subdir, f"{pid}.h5")
    
    # 转换为 numpy 数组
    data = np.stack(features, axis=0) # (N, Dim)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('features', data=data)

# 创建一个空特征文件的辅助函数 (用于无图患者)
def save_empty_features(pid, backup_dir):
    h5_subdir = os.path.join(backup_dir, 'h5_files')
    os.makedirs(h5_subdir, exist_ok=True)
    h5_path = os.path.join(h5_subdir, f"{pid}.h5")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('features', data=np.array([]))

# -------------------------------------------------------------------------
# 5. MAIN LOGIC
# -------------------------------------------------------------------------

def main():
    os.makedirs(CONF['output_dir'], exist_ok=True)
    os.makedirs(CONF['backup_dir'], exist_ok=True)
    
    # 确保存放 h5 的子目录存在
    h5_dir = os.path.join(CONF['backup_dir'], 'h5_files')
    os.makedirs(h5_dir, exist_ok=True)
    
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])
    torch.manual_seed(CONF['seed'])
    device = torch.device(CONF['device'])

    # Load Split JSON
    print(f"Loading Split: {CONF['split_json']}")
    with open(CONF['split_json'], 'r') as f:
        split_data = json.load(f)
    
    # 遍历所有 Fold，获取所有唯一 Patient ID 用于 Step 0 (避免重复提取)
    all_unique_ids = set()
    folds_to_process = []

    # 识别 fold_1 到 fold_5
    for key in split_data.keys():
        if key.startswith("fold_"):
            folds_to_process.append(key)
            train_ids = split_data[key].get('train', [])
            valid_ids = split_data[key].get('valid', [])
            test_ids = split_data[key].get('test', [])
            all_unique_ids.update(train_ids + valid_ids + test_ids)
    
    folds_to_process.sort() # 确保顺序: fold_1, fold_2...
    print(f"Found {len(folds_to_process)} folds: {folds_to_process}")
    print(f"Total unique patients across all folds: {len(all_unique_ids)}")

    # -----------------------------------------------------------
    # Step 0: Global Stream Feature Extraction (Optimized Batching)
    # -----------------------------------------------------------
    # 这一步是全局的，只要提取了一次，所有 Fold 都可以共用
    print(f"\n{'='*10} Step 0: Global Stream Feature Extraction {'='*10}")
    
    all_ids_list = list(all_unique_ids)
    pending_ids = [pid for pid in all_ids_list if not os.path.exists(os.path.join(h5_dir, f"{pid}.h5"))]
    
    skipped_count = len(all_ids_list) - len(pending_ids)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed patients found in {h5_dir}.")
    
    if len(pending_ids) > 0:
        print(f"Found {len(pending_ids)} new patients needing extraction.")
        print(f"Loading Feature Extractor (UNI) on {device}...")
        try:
            extractor = UNIFeatureExtractor(local_path=CONF['uni_weight_path']).to(device)
            extractor.eval()
            transform = get_transforms()
        except Exception as e:
            print(f"Failed to load UNI model: {e}")
            return
            
        tensor_buffer = []  
        pid_buffer = []     
        results_cache = defaultdict(list)
        active_pids_in_results = set()
        
        pbar = tqdm(pending_ids, desc="Processing Stream")
        
        def flush_buffer():
            if not tensor_buffer:
                return
            batch_tensors = torch.stack(tensor_buffer).to(device)
            with torch.no_grad():
                feats = extractor(batch_tensors).cpu().numpy()
            for i, feat in enumerate(feats):
                pid = pid_buffer[i]
                results_cache[pid].append(feat)
                active_pids_in_results.add(pid)
            tensor_buffer.clear()
            pid_buffer.clear()

        for current_pid in pbar:
            img_paths = get_patient_image_paths(CONF['input_root'], current_pid)
            patient_patches = []
            if img_paths:
                for p_path in img_paths:
                    patient_patches.extend(get_patches_from_image(p_path, CONF['patch_size']))
            
            if not patient_patches:
                save_empty_features(current_pid, CONF['backup_dir'])
                continue

            for patch in patient_patches:
                t = transform(patch)
                tensor_buffer.append(t)
                pid_buffer.append(current_pid)
                if len(tensor_buffer) >= CONF['batch_size']:
                    flush_buffer()

            pids_waiting_in_buffer = set(pid_buffer)
            pids_to_save = []
            for pid in list(active_pids_in_results):
                if pid in pids_waiting_in_buffer:
                    continue
                pids_to_save.append(pid)
            
            for pid in pids_to_save:
                save_patient_features(pid, results_cache[pid], CONF['backup_dir'])
                del results_cache[pid]
                active_pids_in_results.remove(pid)
                
        if len(tensor_buffer) > 0:
            flush_buffer()
            
        for pid, feats in results_cache.items():
            save_patient_features(pid, feats, CONF['backup_dir'])
            
        print("Feature extraction complete. Cleaning up UNI model...")
        del extractor
        torch.cuda.empty_cache()
    else:
        print(f"All features already exist in {h5_dir}. Skipping extraction step.")

    # -----------------------------------------------------------
    # Loop over Folds
    # -----------------------------------------------------------
    for fold_name in folds_to_process:
        print(f"\n\n{'#'*30}")
        print(f" Processing {fold_name}")
        print(f"{'#'*30}")

        train_ids = split_data[fold_name].get('train', [])
        valid_ids = split_data[fold_name].get('valid', [])
        test_ids = split_data[fold_name].get('test', [])
        current_fold_all_ids = train_ids + valid_ids + test_ids

        # -----------------------------------------------------------
        # Step A: Prototype Generation (Specific to current Fold)
        # -----------------------------------------------------------
        print(f"\n--- Step A: Prototype Generation for {fold_name} ---")
        
        # 文件名包含 fold_name，避免覆盖
        proto_filename = f"prototypes_{fold_name}_num={CONF['n_proto']}.pkl"
        proto_save_path = os.path.join(CONF['backup_dir'], proto_filename)
        
        prototypes = None
        if os.path.exists(proto_save_path):
            print(f"Loading existing prototypes for {fold_name} from {proto_save_path}")
            prototypes = joblib.load(proto_save_path)
        else:
            print(f"Generating Prototypes using TRAIN set of {fold_name}")
            sampled_features = []
            total_collected = 0
            
            shuffled_train_ids = list(train_ids)
            random.shuffle(shuffled_train_ids)
            
            pbar = tqdm(shuffled_train_ids, desc=f"Sampling Train Patches ({fold_name})")
            
            for pid in pbar:
                if total_collected >= CONF['n_patches_sample']:
                    break
                
                h5_path = os.path.join(h5_dir, f"{pid}.h5")
                if not os.path.exists(h5_path):
                    continue
                    
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if 'features' not in f or f['features'].shape[0] == 0:
                            continue
                        feats = f['features'][:]
                        
                    if len(feats) > 0:
                        n_take = min(len(feats), 500) 
                        indices = np.random.choice(len(feats), n_take, replace=False)
                        sampled_features.append(feats[indices])
                        total_collected += n_take
                        pbar.set_postfix({'collected': total_collected})
                except Exception as e:
                    print(f"Error reading {h5_path}: {e}")

            if not sampled_features:
                print(f"Warning: Could not extract features for {fold_name}. Skipping this fold.")
                continue
            else:
                all_sampled = np.concatenate(sampled_features, axis=0)
                print(f"Clustering {all_sampled.shape[0]} patches into {CONF['n_proto']} prototypes...")
                
                prototypes = cluster(
                    patches=all_sampled,
                    n_proto=CONF['n_proto'],
                    mode=CONF['cluster_mode'],
                    n_proto_patches=CONF['n_patches_sample']
                )
                joblib.dump(prototypes, proto_save_path)
                print(f"Prototypes saved to {proto_save_path}")

        # -----------------------------------------------------------
        # Step B & C: PANTHER Inference (Specific to current Fold)
        # -----------------------------------------------------------
        print(f"\n--- Step B/C: PANTHER Inference for {fold_name} ---")
        
        feature_dim = prototypes.shape[1]
        
        # 初始化新的 PANTHER 模型 (因为 Prototypes 变了)
        panther_model = StructuredPANTHER(
            in_dim=feature_dim,
            n_proto=CONF['n_proto'],
            prototypes=prototypes, 
            em_iter=3,
            tau=0.001,
            ot_eps=0.1,
            fix_proto=True
        ).to(device)
        panther_model.eval()

        final_results = {}
        missing_ids_in_fold = [] # 记录本 fold 中确实的患者ID
        
        process_pbar = tqdm(current_fold_all_ids, desc=f"Extracting features ({fold_name})")
        
        for pid in process_pbar:
            h5_path = os.path.join(h5_dir, f"{pid}.h5")
            if not os.path.exists(h5_path):
                missing_ids_in_fold.append(pid)
                continue
                
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'features' not in f or f['features'].shape[0] == 0:
                        # 特征为空，也视为 Missing，不加入 final_results
                        missing_ids_in_fold.append(pid)
                        continue
                    patch_features = f['features'][:]
                
                # Inference
                x = torch.from_numpy(patch_features).float().unsqueeze(0).to(device)
                mask = torch.ones(1, x.shape[1]).to(device)
                
                with torch.no_grad():
                    embedding = panther_model(x, mask)
                    
                embedding_np = embedding.squeeze(0).cpu().numpy()
                final_results[str(pid)] = embedding_np
                
            except Exception as e:
                print(f"Error in PANTHER inference for {pid}: {e}")
                missing_ids_in_fold.append(pid)

        # -----------------------------------------------------------
        # CHECK: Validation of Completeness
        # -----------------------------------------------------------
        expected_count = len(current_fold_all_ids)
        actual_count = len(final_results)
        
        if missing_ids_in_fold:
            print(f"\n[WARNING] Fold {fold_name}: {len(missing_ids_in_fold)} patients skipped due to missing/empty features:")
            print(f"Skipped IDs: {missing_ids_in_fold}")
        
        if expected_count != actual_count:
            print(f"\n[CHECK FAILED] Fold {fold_name}: Missing {expected_count - actual_count} patients in output.")
            print(f"Expected {expected_count} (Train+Valid+Test), got {actual_count}.")
        else:
            print(f"\n[CHECK PASSED] Fold {fold_name}: All {expected_count} patients (Train/Valid/Test) successfully processed.")

        # -----------------------------------------------------------
        # Step D: Save Final Output for this Fold
        # -----------------------------------------------------------
        # 文件名包含 fold_name
        output_pkl = os.path.join(CONF['output_dir'], f"feature_image_pathology_{fold_name}.pkl")
        print(f"Saving features for {len(final_results)} patients ({fold_name}) to: {output_pkl}")
        joblib.dump(final_results, output_pkl)
        
        # 清理内存，为下一个 fold 做准备
        del panther_model
        torch.cuda.empty_cache()

    print("\nAll folds processed successfully!")

if __name__ == "__main__":
    main()