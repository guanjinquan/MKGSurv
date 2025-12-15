import pandas as pd
import pickle

# 使用DIMAF中的临床数据文件
clinical_file = '/home/Guanjq/NewWork/DIMAF/src/data/data_files/tcga_kirc/clinical_data_all.csv'
output_path = '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC/source/kirc_patients.pkl'

# 读取临床数据
df = pd.read_csv(clinical_file, sep=',', low_memory=False)

# 提取患者ID
if 'case_id' in df.columns:
    patient_ids = df['case_id'].dropna().unique()
    patient_list = sorted(patient_ids.tolist())
    
    # 保存为pickle文件
    with open(output_path, 'wb') as f:
        pickle.dump(patient_list, f)
    
    print(f"提取了 {len(patient_list)} 个KIRC患者ID")
    print(f"已保存到: {output_path}")
else:
    print("未找到 case_id 列")