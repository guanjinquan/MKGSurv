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
    'split_json': "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/split_seed=2024.json",
    
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
    
    train_ids = split_data.get('train', [])
    valid_ids = split_data.get('valid', [])
    test_ids = split_data.get('test', [])
    all_ids = train_ids + valid_ids + test_ids
    print(f"IDs - Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}")

    # -----------------------------------------------------------
    # Step 0: Stream Feature Extraction (Optimized Batching)
    # -----------------------------------------------------------
    print(f"\n{'='*10} Step 0: Stream Feature Extraction {'='*10}")
    
    # 1. 严格筛选出需要提取特征的 Patients (检查 h5_files 子目录)
    # 只有当 h5_dir 下不存在该患者的 .h5 文件时，才加入待处理列表
    pending_ids = [pid for pid in all_ids if not os.path.exists(os.path.join(h5_dir, f"{pid}.h5"))]
    
    # 打印跳过信息，让用户放心
    skipped_count = len(all_ids) - len(pending_ids)
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
            
        # 缓冲区
        tensor_buffer = []  # 存放 Tensor
        pid_buffer = []     # 存放对应的 PID
        
        # 结果缓存: {pid: [feat1, feat2, ...]}
        results_cache = defaultdict(list)
        
        # 记录正在处理的 PID 集合，用于判断是否可以保存
        active_pids_in_results = set()
        
        pbar = tqdm(pending_ids, desc="Processing Stream")
        
        # ------------------------------------------
        # 内部函数：执行 Batch 推理并分发结果
        # ------------------------------------------
        def flush_buffer():
            if not tensor_buffer:
                return
            
            # 堆叠 Batch
            batch_tensors = torch.stack(tensor_buffer).to(device)
            
            with torch.no_grad():
                # Inference
                feats = extractor(batch_tensors).cpu().numpy()
            
            # 分发结果
            for i, feat in enumerate(feats):
                pid = pid_buffer[i]
                results_cache[pid].append(feat)
                active_pids_in_results.add(pid)
            
            # 清空缓冲区
            tensor_buffer.clear()
            pid_buffer.clear()

        # ------------------------------------------
        # 主循环：遍历患者 -> 收集 Patch -> 凑 Batch
        # ------------------------------------------
        for current_pid in pbar:
            # 1. 获取该患者所有 Patch
            img_paths = get_patient_image_paths(CONF['input_root'], current_pid)
            patient_patches = []
            if img_paths:
                for p_path in img_paths:
                    patient_patches.extend(get_patches_from_image(p_path, CONF['patch_size']))
            
            # 如果该患者没有图片，记录空文件或跳过
            if not patient_patches:
                save_empty_features(current_pid, CONF['backup_dir'])
                continue

            # 2. 逐个 Patch 加入全局缓冲区
            for patch in patient_patches:
                # Transform to Tensor (3, 224, 224)
                t = transform(patch)
                tensor_buffer.append(t)
                pid_buffer.append(current_pid)
                
                # 如果缓冲区满了，执行推理
                if len(tensor_buffer) >= CONF['batch_size']:
                    flush_buffer()

            # 3. 检查并保存已完成的患者
            # 找出所有在 cache 中有数据，但不在 buffer 中等待的 pid
            pids_waiting_in_buffer = set(pid_buffer)
            
            # 准备保存列表 (转换成 list 避免运行时修改 dict error)
            pids_to_save = []
            for pid in list(active_pids_in_results):
                # 如果 pid 还有残余数据在 buffer 里，不能保存
                if pid in pids_waiting_in_buffer:
                    continue
                
                # 如果 pid 是当前正在处理的 patient，且循环还没真正结束，这里因为代码是串行的，
                # 所以一旦 current_pid 不在 buffer 里，说明它的所有 patch 都推理完了。
                pids_to_save.append(pid)
            
            for pid in pids_to_save:
                save_patient_features(pid, results_cache[pid], CONF['backup_dir'])
                del results_cache[pid]
                active_pids_in_results.remove(pid)
                
        # ------------------------------------------
        # 循环结束：清理剩余的 Buffer
        # ------------------------------------------
        if len(tensor_buffer) > 0:
            flush_buffer()
            
        # 保存所有剩余的 results
        for pid, feats in results_cache.items():
            save_patient_features(pid, feats, CONF['backup_dir'])
            
        print("Feature extraction complete. Cleaning up UNI model...")
        del extractor
        torch.cuda.empty_cache()
        
    else:
        print(f"All features already exist in {h5_dir}. Skipping extraction step.")

    # -----------------------------------------------------------
    # Step A: Prototype Generation (Load from Disk)
    # -----------------------------------------------------------
    print(f"\n{'='*10} Step A: Prototype Generation {'='*10}")
    
    # [修改点]：prototypes 保存路径在 backup 根目录下
    proto_filename = f"prototypes_num={CONF['n_proto']}.pkl"
    proto_save_path = os.path.join(CONF['backup_dir'], proto_filename)
    
    if os.path.exists(proto_save_path):
        print(f"Loading existing UNI prototypes from {proto_save_path}")
        prototypes = joblib.load(proto_save_path)
    else:
        print(f"Generating Prototypes from Training Set H5 files in: {h5_dir}")
        sampled_features = []
        total_collected = 0
        
        shuffled_train_ids = list(train_ids)
        random.shuffle(shuffled_train_ids)
        
        pbar = tqdm(shuffled_train_ids, desc="Sampling Train Patches")
        
        debug_path_printed = False 

        for pid in pbar:
            if total_collected >= CONF['n_patches_sample']:
                break
            
            # 读取路径调整为 h5_dir 子目录
            h5_path = os.path.join(h5_dir, f"{pid}.h5")
            if not os.path.exists(h5_path):
                if not debug_path_printed:
                    print(f"\n[DEBUG WARNING] File not found: {h5_path}")
                    print("Please ensure Step 0 ran successfully.")
                    debug_path_printed = True
                continue
                
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Handle empty datasets
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
            print("Warning: Could not extract any features from training set for prototypes.")
            print("Please check your input paths and data.")
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
    # Step B: Initialize PANTHER
    # -----------------------------------------------------------
    # Check if prototypes exist
    if not os.path.exists(proto_save_path):
        print("Prototypes not found, skipping PANTHER step.")
        return

    feature_dim = prototypes.shape[1]
    print(f"Initializing StructuredPANTHER (dim={feature_dim}, proto={CONF['n_proto']})...")
    
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

    # -----------------------------------------------------------
    # Step C: Process All Patients (Load from Disk)
    # -----------------------------------------------------------
    print(f"\n{'='*10} Step C: PANTHER Inference {'='*10}")
    final_results = {}
    
    process_pbar = tqdm(all_ids, desc="Extracting PANTHER Features")
    
    for pid in process_pbar:
        # 读取路径调整为 h5_dir 子目录
        h5_path = os.path.join(h5_dir, f"{pid}.h5")
        if not os.path.exists(h5_path):
            continue
            
        try:
            # Load Features
            with h5py.File(h5_path, 'r') as f:
                if 'features' not in f or f['features'].shape[0] == 0:
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

    # -----------------------------------------------------------
    # Step D: Save Final Output
    # -----------------------------------------------------------
    output_pkl = os.path.join(CONF['output_dir'], f"feature_image_pathology.pkl")
    print(f"Saving features for {len(final_results)} patients to: {output_pkl}")
    joblib.dump(final_results, output_pkl)
    print("Done!")

if __name__ == "__main__":
    main()