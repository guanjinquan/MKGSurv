import pandas as pd
import numpy as np
import os
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# 1. 路径与列定义 (Configuration)
# =============================================================================

BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC"
SOURCE_DIR = os.path.join(BASE_DIR, "source")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# --- 原始数据源 (Raw Data TSVs) ---
BIOSPECIMEN_DIR = os.path.join(SOURCE_DIR, "biospecimen.project-tcga-lusc.2025-11-08")
CLINICAL_DIR = os.path.join(SOURCE_DIR, "clinical.project-tcga-lusc.2025-11-08")

PATH_SLIDE_TSV = os.path.join(BIOSPECIMEN_DIR, "slide.tsv")
PATH_SAMPLE_TSV = os.path.join(BIOSPECIMEN_DIR, "sample.tsv")
PATH_PORTION_TSV = os.path.join(BIOSPECIMEN_DIR, "portion.tsv")

# --- 处理后的文件 (仅用于获取 Schema) ---
PATH_CLINICAL_AGG = os.path.join(PROCESSED_DIR, "clinical_data_aggregated.csv")
PATH_TREATMENT_AGG = os.path.join(PROCESSED_DIR, "treatment_data_aggregated.csv")
PATH_TREATMENT_TEXT = os.path.join(PROCESSED_DIR, "text_treatment.csv")
PATH_PATHOLOGY_AGG = os.path.join(PROCESSED_DIR, "pathology_aggregated.csv")

# --- 其他资源 ---
PATH_LABELS = os.path.join(PROCESSED_DIR, "lusc_patient_labels.csv")
PATH_REPORTS = os.path.join(PROCESSED_DIR, "tcga_lusc_reports.csv")
PATH_RNA = os.path.join(SOURCE_DIR, "HiSeqV2_PANCAN")
PATH_HALLMARK = "/home/Guanjq/NewWork/MedAlignFusion/Data/MMP_hallmarks_signatures.csv"

# --- 输出 ---
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "multimodal_texts.csv")

# =============================================================================
# [User Defined Columns] 显式定义的 Biospecimen 元数据列
# =============================================================================

# 1. Slide Data (Cellular composition)
COLS_SLIDE = [
    'slides.percent_tumor_cells',
    'slides.percent_tumor_nuclei',
    'slides.percent_necrosis',
    'slides.percent_normal_cells',
    'slides.percent_stromal_cells',
    'slides.percent_inflam_infiltration',
    'slides.percent_lymphocyte_infiltration',
    'slides.percent_monocyte_infiltration',
    'slides.percent_neutrophil_infiltration',
    'slides.section_location'
]

# 2. Sample Data (Tissue descriptors)
COLS_SAMPLE = [
    'samples.sample_type',
    'samples.tissue_type',
    'samples.tumor_descriptor',
    'samples.preservation_method'
]

# 3. Portion Data (Physical properties)
COLS_PORTION = [
    'portions.is_ffpe',
    'portions.weight'
]

# =============================================================================
# 2. 核心工具函数
# =============================================================================

def get_cols_from_processed(filepath):
    """从处理好的 CSV 文件中只读取列名 (作为 Schema)。"""
    if not os.path.exists(filepath):
        return []
    try:
        df_head = pd.read_csv(filepath, nrows=0)
        cols = df_head.columns.tolist()
        ignore_list = ['cases.submitter_id', 'cases.case_id', 'submitter_id', 'case_id', 'Unnamed: 0']
        valid_cols = [c for c in cols if c not in ignore_list]
        return valid_cols
    except Exception:
        return []

def load_raw_tsv(filepath):
    """鲁棒地读取 TSV 文件。"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath, sep='\t', low_memory=False)
    except Exception:
        try:
            return pd.read_csv(filepath, sep='\t', quoting=3, on_bad_lines='skip', low_memory=False)
        except Exception:
            return pd.DataFrame()

def clean_value(val):
    """清洗单元格数据"""
    if pd.isna(val): return None
    s = str(val).strip()
    # 扩展无效值列表
    invalid_tokens = [
        '--', 'nan', 'null', 'none', 'not reported', 'unknown', 
        'not applicable', '.', '?', "'--", '"--"', 'cannot be determined'
    ]
    if s.lower() in invalid_tokens: return None
    s = s.strip("'").strip('"')
    if not s: return None
    return s

def row_to_text(row, target_cols):
    """将一行数据转换为文本"""
    parts = []
    # 预计算 row 中存在的列
    row_cols = set(row.index)
    
    for col in target_cols:
        if col in row_cols:
            val = clean_value(row[col])
            if val:
                # 移除前缀并格式化
                key_name = col.split('.')[-1].replace('_', ' ').title()
                parts.append(f"{key_name}: {val}")
    return "; ".join(parts) if parts else ""

# =============================================================================
# 3. 专用处理模块
# =============================================================================

def process_biospecimen_metadata(target_ids):
    """
    专门处理 Slide, Sample, Portion 的元数据。
    返回一个 Series，索引为 patient_id，值为文本。
    """
    print("\n   Processing Biospecimen Metadata (Slide/Sample/Portion)...")
    
    # 容器
    results = pd.DataFrame(index=target_ids)
    results['text'] = ""
    
    # 辅助函数：处理单个 TSV 并聚合
    def process_tsv(filepath, target_cols, label):
        if not os.path.exists(filepath):
            print(f"      Warning: {filepath} not found.")
            return pd.Series(dtype=object)
            
        df = load_raw_tsv(filepath)
        if df.empty: return pd.Series(dtype=object)
        
        # 确定 Patient ID
        if 'cases.submitter_id' in df.columns:
            df['patient_id'] = df['cases.submitter_id'].astype(str).str.strip()
        elif 'submitter_id' in df.columns: # 有些文件列名可能不同，尝试通用名
             # 通常 submitter_id 是 barcode，前12位是 patient
             df['patient_id'] = df.iloc[:, 0].astype(str).str.slice(0, 12)
        else:
             # 尝试第一列作为 Barcode
             df['patient_id'] = df.iloc[:, 0].astype(str).str.slice(0, 12)

        # 过滤需要的列
        available_cols = [c for c in target_cols if c in df.columns]
        if not available_cols:
            return pd.Series(dtype=object)
            
        # 聚合函数：同一病人可能有多行（多个切片/样本），取所有唯一值
        def agg_meta(g):
            txt_parts = []
            for col in available_cols:
                # 获取该列所有非空的唯一值
                vals = set()
                # 检查列是否存在于子DataFrame中
                if col in g.columns:
                    for x in g[col]:
                        v = clean_value(x)
                        if v: vals.add(v)
                
                if vals:
                    key = col.split('.')[-1].replace('_', ' ').title()
                    val_str = ", ".join(sorted(vals))
                    txt_parts.append(f"{key}: {val_str}")
            return "; ".join(txt_parts)

        # 按病人聚合
        # FIX: Explicitly select columns to avoid FutureWarning regarding grouping keys
        series = df.groupby('patient_id')[available_cols].apply(agg_meta)
        
        # 添加标签前缀
        series = series[series != ""]
        return series.apply(lambda x: f"[{label}]: {x}")

    # 1. Slide Data
    s_slide = process_tsv(PATH_SLIDE_TSV, COLS_SLIDE, "Slide Meta")
    
    # 2. Sample Data
    s_sample = process_tsv(PATH_SAMPLE_TSV, COLS_SAMPLE, "Sample Meta")
    
    # 3. Portion Data
    s_portion = process_tsv(PATH_PORTION_TSV, COLS_PORTION, "Portion Meta")
    
    # 合并
    combined = pd.DataFrame({'slide': s_slide, 'sample': s_sample, 'portion': s_portion})
    # 只保留 target_ids 中的行
    combined = combined.reindex(target_ids).fillna("")
    
    # 拼接文本
    final_text = (combined['slide'] + "\n" + combined['sample'] + "\n" + combined['portion']).str.strip()
    
    return final_text

def process_genomics(target_ids):
    """处理 genomics 数据"""
    if not os.path.exists(PATH_RNA) or not os.path.exists(PATH_HALLMARK):
        return pd.Series(index=target_ids, dtype=object).fillna("")
    try:
        hallmark = pd.read_csv(PATH_HALLMARK)
        h_map = {c: hallmark[c].dropna().astype(str).str.strip().tolist() for c in hallmark.columns}
        
        rna = pd.read_csv(PATH_RNA, sep='\t')
        if 'sample' in rna.columns: rna.set_index('sample', inplace=True)
        rna = rna.T
        rna.index = rna.index.str.slice(0, 12) # 截取 Patient ID
        rna = rna.groupby(rna.index).mean()
        rna.columns = rna.columns.astype(str).str.strip()
        
        valid = list(set(target_ids) & set(rna.index))
        rna = rna.loc[valid]
        
        results = {}
        for pid, row in tqdm(rna.iterrows(), total=len(rna), desc="genomics"):
            parts = []
            for pname, genes in h_map.items():
                v_genes = [g for g in genes if g in rna.columns]
                if v_genes:
                    top = row[v_genes].sort_values(ascending=False).head(50).index.tolist()
                    if top: parts.append(f"Pathway {pname}: {', '.join(top)}.")
            results[pid] = " ".join(parts)
        return pd.Series(results).reindex(target_ids).fillna("")
    except Exception:
        return pd.Series(index=target_ids, dtype=object).fillna("")

# =============================================================================
# 4. 主流程
# =============================================================================

def main():
    print("=== Multimodal Text Generation (Updated with Biospecimen Meta) ===")
    
    # 1. 确定目标病人列表
    if not os.path.exists(PATH_LABELS):
        print("Error: Label file not found.")
        return
    labels_df = pd.read_csv(PATH_LABELS)
    target_ids = labels_df['cases.submitter_id'].astype(str).str.strip().unique()
    print(f"Target Patients: {len(target_ids)}")
    
    # 初始化结果 DataFrame
    final_df = pd.DataFrame(index=target_ids)
    final_df.index.name = 'cases.submitter_id'
    
    # 2. 获取 Schema (用于 Generic Tabular 处理)
    print("\n[Step 1] Loading Schemas for General Tabular Data...")
    cols_clinical = get_cols_from_processed(PATH_CLINICAL_AGG)
    cols_treatment = get_cols_from_processed(PATH_TREATMENT_AGG) + get_cols_from_processed(PATH_TREATMENT_TEXT)
    cols_path_tabular = get_cols_from_processed(PATH_PATHOLOGY_AGG) # 仅包含分期等诊断信息
    
    # 3. 加载并聚合所有 Raw TSV (用于 General Tabular)
    print("\n[Step 2] Scanning Raw TSVs for General Data...")
    raw_files = []
    if os.path.exists(CLINICAL_DIR):
        raw_files.extend([os.path.join(CLINICAL_DIR, f) for f in os.listdir(CLINICAL_DIR) if f.endswith('.tsv')])
    if os.path.exists(BIOSPECIMEN_DIR): # 依然扫描，用于提取可能的其他列
        raw_files.extend([os.path.join(BIOSPECIMEN_DIR, f) for f in os.listdir(BIOSPECIMEN_DIR) if f.endswith('.tsv')])
        
    dfs = []
    id_map = {} # case_id -> submitter_id
    for f in tqdm(raw_files, desc="Loading TSVs"):
        df = load_raw_tsv(f)
        if not df.empty:
            if 'cases.case_id' in df.columns and 'cases.submitter_id' in df.columns:
                temp = df[['cases.case_id', 'cases.submitter_id']].dropna().drop_duplicates()
                for c, s in temp.values: id_map[c] = s
            dfs.append(df)
            
    if dfs:
        full_concat = pd.concat(dfs, axis=0, ignore_index=True)
        # ID 补全
        if 'cases.submitter_id' in full_concat.columns:
            full_concat['cases.submitter_id'] = full_concat['cases.submitter_id'].fillna(full_concat.get('cases.case_id', pd.Series()).map(id_map))
        full_concat = full_concat.dropna(subset=['cases.submitter_id'])
        full_concat['cases.submitter_id'] = full_concat['cases.submitter_id'].astype(str).str.strip()
        
        # 筛选目标病人
        full_concat = full_concat[full_concat['cases.submitter_id'].isin(target_ids)]
        
        # 聚合通用数据 (clinical, Treatment, pathology Tabular)
        print("   Aggregating General Tabular Data...")
        def agg_func(s):
            vals = {clean_value(x) for x in s if clean_value(x)}
            return "; ".join(sorted(vals)) if vals else ""
            
        # 仅保留相关列加速
        relevant_cols = list(set(cols_clinical + cols_treatment + cols_path_tabular) & set(full_concat.columns))
        relevant_cols.append('cases.submitter_id')
        grouped = full_concat[relevant_cols].groupby('cases.submitter_id').agg(agg_func)
        
        # 填充到 Final DF
        final_df['clinical'] = grouped.apply(lambda r: row_to_text(r, cols_clinical), axis=1).reindex(target_ids).fillna("")
        final_df['Treatment'] = grouped.apply(lambda r: row_to_text(r, cols_treatment), axis=1).reindex(target_ids).fillna("")
        path_tabular = grouped.apply(lambda r: row_to_text(r, cols_path_tabular), axis=1).reindex(target_ids).fillna("")
    else:
        print("Warning: No raw data loaded for tabular sections.")
        path_tabular = pd.Series("", index=target_ids)
        final_df['clinical'] = ""
        final_df['Treatment'] = ""

    # 4. Biospecimen Metadata (Slide/Sample/Portion) - [NEW]
    print("\n[Step 3] Processing Biospecimen Metadata (The Missing Link)...")
    path_meta = process_biospecimen_metadata(target_ids)

    # 5. pathology Reports
    print("\n[Step 4] Processing pathology Reports...")
    path_reports = pd.Series("", index=target_ids)
    if os.path.exists(PATH_REPORTS):
        try:
            rep_df = pd.read_csv(PATH_REPORTS)
            id_col = 'cases.submitter_id' if 'cases.submitter_id' in rep_df.columns else 'patient_id'
            if id_col in rep_df.columns:
                rep_df[id_col] = rep_df[id_col].astype(str).str.strip()
                def combine_rep(g):
                    texts = [f"{str(r.get('report_text','')).replace('nan','').strip()}\n{str(r.get('annotation_text','')).replace('nan','').strip()}".strip() for _, r in g.iterrows()]
                    return "\n[Diagnostic Report]:\n".join([t for t in texts if t])
                
                # FIX: Explicitly select report columns to avoid FutureWarning regarding grouping keys
                report_cols = [c for c in ['report_text', 'annotation_text'] if c in rep_df.columns]
                path_reports = rep_df.groupby(id_col)[report_cols].apply(combine_rep).reindex(target_ids).fillna("")
        except Exception as e:
            print(f"Error loading reports: {e}")

    # 6. 合并所有 pathology 信息
    print("   Merging pathology Sections...")
    final_df['pathology'] = (
        path_tabular + "\n\n" + 
        path_reports + "\n\n" + 
        path_meta
    ).str.strip()

    # 7. genomics
    print("\n[Step 5] Processing genomics...")
    final_df['genomics'] = process_genomics(target_ids)

    # 8. 保存
    print(f"\n[Step 6] Saving to {OUTPUT_FILE}...")
    final_df = final_df.fillna("")
    final_df.to_csv(OUTPUT_FILE, quoting=csv.QUOTE_ALL)
    
    # --- 9. 最终统计 (Data Completeness Check) ---
    print("\n" + "="*50)
    print("   📊 数据完整性统计 (Data Completeness Check)")
    print("="*50)
    
    total_patients = len(final_df)
    print(f"总目标患者数 (Total Target Patients): {total_patients}")
    
    check_cols = ['clinical', 'Treatment', 'pathology', 'genomics']
    all_complete = True
    
    for col in check_cols:
        if col not in final_df.columns:
            print(f"❌ 列 {col} 不存在！")
            continue
            
        # 统计空值 (空字符串)
        # 注意: final_df 已经 fillna("")，所以直接检查是否为空字符串
        missing_mask = (final_df[col].astype(str).str.strip() == "")
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            print(f"✅ {col:<10}: 数据完整 (无缺失)")
        else:
            all_complete = False
            pct = (missing_count / total_patients) * 100
            print(f"⚠️ {col:<10}: 缺失 {missing_count} 例 ({pct:.1f}%)")
            
    if all_complete:
        print("\n✨ 完美！所有目标患者的四模态数据均已齐全。")
    else:
        print("\n📝 部分模态存在缺失，请检查原始数据源是否覆盖了所有 label 患者。")
    print("="*50)
    
    print("Done.")

if __name__ == "__main__":
    main()