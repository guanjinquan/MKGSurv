import joblib
import torch
import os

def quick_check():
    file_paths = [
        '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_image_pathology_fold1.pkl',
        '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_image_pathology_fold2.pkl',
        '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_image_pathology_fold3.pkl',
        '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_image_pathology_fold4.pkl',
        '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_image_pathology_fold5.pkl'
    ]
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        # with open(file_path, 'rb') as f:
        #     data = .load(f)
        data = joblib.load(file_path)
        
        all_good = True
        for pid, tensor in list(data.items()):  # 只检查前10个样本
            if isinstance(tensor, torch.Tensor):
                dim = tensor.shape[0]
            else:
                dim = tensor.shape[0]
            
            if dim != 128:
                print(f"{os.path.basename(file_path)} - {pid}: ❌ shape[0] = {dim}")
                all_good = False
                break  # 发现一个错误就停止
        
        if all_good:
            print(f"{os.path.basename(file_path)}: ✅ 所有tensor的shape[0]都是128 (抽查前10个样本)")

if __name__ == "__main__":
    quick_check()