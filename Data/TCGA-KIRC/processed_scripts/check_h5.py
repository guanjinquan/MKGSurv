import json
import os
import glob

# 读取JSON文件
json_path = '/home/Zhengzx/MedAlignFusion/Data/TCGA-BRCA/processed/brca_patients_5fold.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# 获取所有H5文件
h5_dir = '/home/Zhengzx/MedAlignFusion/Data/TCGA-BRCA/h5_files'
h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

# 创建H5文件名集合（只包含患者ID部分）
h5_patient_ids = set()
for h5_file in h5_files:
    filename = os.path.basename(h5_file)
    # 提取患者ID部分（例如从 "TCGA-E2-A1IJ-01Z-00-DX1.h5" 提取 "TCGA-E2-A1IJ"）
    patient_id = '-'.join(filename.split('-')[:3])
    h5_patient_ids.add(patient_id)

# 查找在JSON中但不在H5文件中的患者ID
missing_patients = []

# 遍历所有折叠的数据
for fold in data['folds']:
    # 检查训练集
    for patient_id in fold['train']:
        if patient_id not in h5_patient_ids:
            missing_patients.append((patient_id, f"Fold {fold['fold']} Train"))
    
    # 检查验证集
    for patient_id in fold['valid']:
        if patient_id not in h5_patient_ids:
            missing_patients.append((patient_id, f"Fold {fold['fold']} Valid"))
    
    # 检查测试集
    for patient_id in fold['test']:
        if patient_id not in h5_patient_ids:
            missing_patients.append((patient_id, f"Fold {fold['fold']} Test"))

# 打印结果
print(f"总共找到 {len(missing_patients)} 个在JSON中存在但在H5目录中缺少的患者ID:")
for patient_id, group in missing_patients:
    print(f"  {patient_id} - {group}")

# 按组分类打印
print("\n按组分类:")
groups = {}
for patient_id, group in missing_patients:
    if group not in groups:
        groups[group] = []
    groups[group].append(patient_id)

for group, patient_ids in groups.items():
    print(f"\n{group}:")
    for patient_id in patient_ids:
        print(f"  {patient_id}")