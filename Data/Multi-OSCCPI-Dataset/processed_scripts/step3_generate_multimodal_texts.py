import pandas as pd
import os
import re
import csv
import numpy as np

# ================= 配置路径 =================
# 输入文件路径
CLINICAL_DATA_PATH = '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/clinical_data.csv'
# 输出文件路径
OUTPUT_CSV_PATH = '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/multimodal_texts.csv'

# ================= 1. 定义固定元数据 (Pathology Header) =================
# 这段话将始终出现在 Pathology 列的最开头
PATHOLOGY_META_INFO = (
    "Image Metadata: H&E stained tissue sections from lesion core and boundary. "
    "Captured at ×200, ×400, ×1000 magnification using Olympus microscope. "
    "Fixed with formalin, paraffin-embedded."
)

# ================= 2. 定义列分组 (避免列名在不同组重复) =================

# --- A. Clinical (临床基线、查体、既往史、血液检查) ---
cols_clinical = [
    # Metadata
    "Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)",
    # History
    "AlcoholHistory(0no/1yes)", "SmokingHistory(0no/1yes)", 
    "BetelNutHistory(0no/1yes)", "PreoperativeHistory(0no/1yes)", 
    "PreoperativeHistoryDetails",
    "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
    "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)",
    # Physical Exam
    "NeckMass(+)", "TumorLocation",
    # Blood Work (术前血检)
    "PreopWBC", "PreopHemoglobin", "PreopPotassium", "PreopAlbumin", "PreopVitaminD"
]

# --- B. Treatment (治疗方案) ---
cols_treatment = [
    "SurgicalMethod",
    "Radiotherapy(0no/1yes)", 
    "Chemotherapy(0no/1yes)"
]

# --- C. Pathology (病理分期、免疫组化、镜下所见、病理报告) ---
# 注意：TNM分期通常以术后病理为准，因此归类在此；若为临床分期(cTNM)可移至Clinical
cols_pathology = [
    # Staging (TNM & Differentiation)
    "TumorT", "TumorN", "TumorM", "TumorDifferentiation(1high/2med/3low)",
    # Invasion & Spread
    "CancerThrombus(0/1)", "SurroundingTissueInvasion(0/1)", 
    "LNM(0/1)", "VascularInvasion(+)", "PerineuralInvasion(+)", 
    "AccessoryChain(+)", "Metastasis(0no/1yes)",
    # Lymph Nodes Details
    "IA(+)", "IB(+)", "IIA(+)", "IIB(+)", "III(+)",
    # IHC / Molecular Markers (免疫组化)
    "Ki-67", "CK5_6(0/1)", "P63(0/1)", "P16(0/1)", "HPV(0/1)", "PD_L1",
    # Text Reports
    "Pathology", "Flap"
]

# --- D. Genomics (基因组学) ---
# 如果源数据中有WES/WGS结果或特定基因突变列，请在此添加。
# 目前留空或仅作为占位符，如果 PD_L1/HPV 算作基因组学也可以移到这里，
# 但通常它们属于病理免疫组化。
cols_genomics = [
    # "GeneMutation_TP53", "GeneMutation_NOTCH1", etc.
] 

def clean_text(text):
    """
    清洗文本：移除制表符、换行符、回车符，处理分号。
    """
    if pd.isna(text):
        return ""
    text_str = str(text)
    
    # 将内容中原有的分号替换为逗号，防止与我们的字段分隔符混淆
    text_str = text_str.replace(';', ',')
    
    # 使用正则表达式将所有空白字符（包括 \t, \n, \r）替换为单个空格
    cleaned = re.sub(r'\s+', ' ', text_str)
    
    # 移除首尾空白及无效字符
    cleaned = cleaned.strip()
    
    if cleaned.lower() in ['nan', 'null', 'none', '/']:
        return ""
        
    return cleaned

def generate_section_text(row, target_cols, prefix_text=""):
    """
    将指定列的数据转换为文本格式 "Key: Value; Key: Value"
    可以指定 prefix_text (如 Metadata) 放在最前面
    """
    data_descriptions = []
    
    # 1. 如果有前缀文本，先加入
    if prefix_text:
        data_descriptions.append(prefix_text)
    
    # 2. 遍历列名提取数据
    for col in target_cols:
        if col in row.index:
            val = row[col]
            # 只有当值非空且有效时才处理
            if pd.notna(val):
                clean_col = clean_text(col)
                clean_val = clean_text(val)
                
                if clean_val:
                    # 格式化为 "Key: Value"
                    data_descriptions.append(f"{clean_col}: {clean_val}")
    
    if not data_descriptions:
        return ""
        
    # 3. 使用分号连接所有信息
    # 注意：如果 prefix_text 存在，它已经是列表第一个元素，join 会自动处理连接
    return "; ".join(data_descriptions)

def main():
    print(">>> 开始处理多模态文本数据...")

    # 1. 读取 Clinical Data
    if not os.path.exists(CLINICAL_DATA_PATH):
        print(f"Error: 找不到源文件 {CLINICAL_DATA_PATH}")
        return

    print(f"Read: {CLINICAL_DATA_PATH}")
    # 显式指定 dtype 防止 PID 读取为数字
    df_clinical = pd.read_csv(CLINICAL_DATA_PATH, dtype={'PID': str})
    
    # 清洗 PID
    if 'PID' in df_clinical.columns:
        df_clinical['PID'] = df_clinical['PID'].str.strip()
        # 去重：保留第一次出现的 PID
        if df_clinical['PID'].duplicated().any():
            print(f"Warning: 检测到重复 PID，将剔除重复项。行数变化: {len(df_clinical)} -> {len(df_clinical.drop_duplicates(subset=['PID']))}")
            df_clinical = df_clinical.drop_duplicates(subset=['PID'])
        df_clinical.set_index('PID', inplace=True)
    else:
        print("Error: Clinical Data 中缺少 'PID' 列")
        return

    # 2. 生成各科室文本数据
    print("Processing: Generating text columns...")

    # A. Clinical
    series_clinical = df_clinical.apply(lambda row: generate_section_text(row, cols_clinical), axis=1)
    
    # B. Treatment
    series_treatment = df_clinical.apply(lambda row: generate_section_text(row, cols_treatment), axis=1)
    
    # C. Pathology (包含 Metadata + Tabular + Text)
    series_pathology = df_clinical.apply(
        lambda row: generate_section_text(row, cols_pathology, prefix_text=PATHOLOGY_META_INFO), 
        axis=1
    )
    
    # D. Genomics (目前可能为空，根据是否有列生成)
    series_genomics = df_clinical.apply(lambda row: generate_section_text(row, cols_genomics), axis=1)

    # 3. 构建最终 DataFrame
    df_result = pd.DataFrame({
        'Clinical': series_clinical,
        'Treatment': series_treatment,
        'Pathology': series_pathology,
        'Genomics': series_genomics
    }, index=df_clinical.index)

    # 4. 保存结果
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Saving to: {OUTPUT_CSV_PATH}")
    
    # 使用 quote_all=True 确保每个文本字段都被引号包裹，防止内容中的特殊字符破坏CSV结构
    df_result.reset_index().to_csv(
        OUTPUT_CSV_PATH, 
        index=False, 
        encoding='utf-8-sig', 
        quoting=csv.QUOTE_ALL
    )
    
    print(">>> 处理完成！")
    print("生成的列: PID, Clinical, Treatment, Pathology, Genomics")
    print(f"Pathology列示例 (前50字符): {df_result['Pathology'].iloc[0][:50]}...")

if __name__ == "__main__":
    main()