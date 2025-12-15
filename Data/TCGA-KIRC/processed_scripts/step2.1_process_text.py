import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
import random
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

# =================配置路径=================
BASE_DIR = "/home/Zhengzx/MedAlignFusion/Data/TCGA-KIRC/processed"

# 输入文件路径c
# 注意：这里更新为包含详细治疗信息的CSV路径
PATH_TREATMENT_CSV = os.path.join(BASE_DIR, "text_treatment.csv")
PATH_REPORTS_CSV = os.path.join(BASE_DIR, "tcga_kirc_reports.csv")

# 输出文件路径
OUTPUT_PATHOLOGY_PKL = os.path.join(BASE_DIR, "features_text_pathology.pkl")
OUTPUT_TREATMENT_PKL = os.path.join(BASE_DIR, "features_text_treatment.pkl")

# =================核心编码器类=================
class ClinicalBertEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.embed_dim = 768 
        
        print("Initializing Text Encoder (ClinicalBERT)...")
        self.text_model_name = "medicalai/ClinicalBERT"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(self.text_model_name)
        except Exception as e:
            print(f"Error loading model from HF Mirror: {e}")
            print("Trying local load or check internet connection.")
            raise e
            
        self.to(self.device)
        self.eval() 

    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def _encode_text(self, texts_list: List[str]) -> List[torch.Tensor]:
        """
        输入: 一个包含多个文本字符串的列表 (例如 [原文本, 增强文本1, 增强文本2...])
        输出: 一个列表, 包含对应的特征 Tensor。
              每个 Tensor 的形状为 (N_chunks, 768), 已移动到 CPU。
        """
        batch_size = len(texts_list)
        chunk_payload = 510 # BERT max 512 - 2 (CLS/SEP)
        
        all_chunks = []
        mapping_info = [] 
        
        # 1. 预处理与分块 (Tokenization & Chunking)
        for i, text in enumerate(texts_list):
            item_specific_chunks = [] 
            
            if isinstance(text, str) and text.strip():
                # 简单的清理，防止空字符
                clean_text = text.strip()
                token_ids = self.tokenizer.encode(clean_text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)
            
            # 记录该样本对应的 chunk 数量
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
            else:
                mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
                all_chunks.extend(item_specific_chunks)

        # 如果没有任何有效 chunk，返回空 tensor 列表
        if not all_chunks:
            return [torch.zeros(0, self.embed_dim) for _ in range(batch_size)]

        # 2. 批量编码 (Batch Encoding)
        # 将所有样本的所有 chunk 展平成一个大 batch 处理
        # 使用 convert_ids_to_tokens 再 join 可能不如直接 decode 准确，但为了对齐 chunk 逻辑，
        # 这里直接 pad token_ids 可能更高效，或者 decode 回 string。
        # 为保持稳健性，这里 decode 回 string 再 encode，虽然稍慢但兼容性好。
        chunk_texts = [self.tokenizer.decode(c, clean_up_tokenization_spaces=True) for c in all_chunks]
        
        # 防止 OOM，如果 chunk 数量巨大，可以分批处理 BERT inference
        bert_batch_size = 32
        pooled_outputs = []
        
        for i in range(0, len(chunk_texts), bert_batch_size):
            batch_texts = chunk_texts[i : i + bert_batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                bert_outputs = self.bert(**inputs)
            
            # 取 CLS token
            batch_pooled = bert_outputs.last_hidden_state[:, 0, :]
            pooled_outputs.append(batch_pooled.cpu())
            
        # 合并所有 batch 的结果
        if pooled_outputs:
            pooled = torch.cat(pooled_outputs, dim=0)
        else:
            pooled = torch.zeros(0, self.embed_dim)

        # 3. 重组结果 (Reconstruct)
        output_list = []
        chunk_cursor = 0
        
        for i in range(batch_size):
            info = next((m for m in mapping_info if m['index'] == i), None)
            n_chunks = info['n'] if info else 0
            
            if n_chunks > 0:
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks]
                output_list.append(valid_features)
                chunk_cursor += n_chunks
            else:
                output_list.append(torch.zeros((0, self.embed_dim)))

        return output_list

# =================数据增强工具函数=================

def clean_term(term):
    """清理单个术语，去除 nan, Unspecified 等无效信息"""
    if not isinstance(term, str):
        return None
    term = term.strip()
    if not term:
        return None
    lower_term = term.lower()
    if lower_term in ['nan', 'null', 'unspecified', 'not reported', 'unknown']:
        return None
    return term

def augment_treatment_info(row: pd.Series, max_augment=10) -> List[str]:
    """
    针对治疗信息进行多列融合与随机 Shuffle 增强。
    
    Args:
        row: DataFrame 的一行
        max_augment: 最大增强数量
        
    Returns:
        List[str]: [原始组合文本, 增强文本1, 增强文本2, ...]
    """
    # 需要合并的列名
    target_cols = [
        'treatments.therapeutic_agents',
        'treatments.treatment_intent_type',
        'treatments.treatment_or_therapy',
        'treatments.treatment_type'
    ]
    
    # 1. 提取所有有效概念 (Terms)
    all_terms = []
    
    for col in target_cols:
        if col in row:
            val = str(row[col])
            # TCGA 数据常用 + 号连接
            parts = val.split('+')
            for p in parts:
                clean = clean_term(p)
                if clean:
                    all_terms.append(clean)
    
    # 去除完全重复的项 (可选，如果想保留频次信息则不去重，这里选择去重以精简)
    # all_terms = sorted(list(set(all_terms))) 
    # 不排序不去重可能更能反映原始记录的权重，但为了 Shuffle 效果，列表即可
    
    if not all_terms:
        return [""] # 无有效信息
    
    # 2. 构建原始文本 (保持列的读取顺序或列表顺序)
    original_text = ", ".join(all_terms)
    results = [original_text]
    
    # 3. 构建增强文本 (随机 Shuffle 列表顺序)
    # 只有当术语数量大于1时，Shuffle才有意义
    if len(all_terms) > 1:
        seen_texts = {original_text}
        
        for _ in range(max_augment * 2): # 尝试更多次以防重复
            shuffled_terms = all_terms.copy()
            random.shuffle(shuffled_terms)
            new_text = ", ".join(shuffled_terms)
            
            if new_text not in seen_texts:
                results.append(new_text)
                seen_texts.add(new_text)
                
            if len(results) >= max_augment + 1:
                break
                
    return results

def augment_pathology_text(text: str) -> List[str]:
    """
    针对病理文本的简单增强（基于标点切分Shuffle）
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    versions = [text]
    
    # 尝试基于句号或加号切分
    for delimiter in ['. ', '+']:
        if delimiter in text:
            parts = [p.strip() for p in text.split(delimiter) if p.strip()]
            if len(parts) > 1:
                for _ in range(10):
                    random.shuffle(parts)
                    # 重新组合
                    new_txt = f"{delimiter}".join(parts)
                    if new_txt not in versions:
                        versions.append(new_txt)
                break # 只用一种分隔符处理一次即可
                
    return versions

# =================主处理流程=================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(BASE_DIR):
        print(f"Base dir {BASE_DIR} does not exist. Creating...")
        os.makedirs(BASE_DIR, exist_ok=True)

    encoder = ClinicalBertEncoder(device=device)

    # ---------------------------------------------------------
    # 任务 1: 处理 Treatment/Clinical Labels (多列融合 + Shuffle 增强)
    # ---------------------------------------------------------
    if os.path.exists(PATH_TREATMENT_CSV):
        print(f"Processing Treatment CSV: {PATH_TREATMENT_CSV}")
        df_treat = pd.read_csv(PATH_TREATMENT_CSV)
        
        # 确保 ID 列是字符串并去空
        df_treat['cases.submitter_id'] = df_treat['cases.submitter_id'].astype(str).str.strip()
        
        patient_treatment_dict = {} # PID -> List[Tensor]
        patient_ids = df_treat['cases.submitter_id'].tolist()
        
        print("Generating patient treatment features with Shuffle Augmentation...")
        total = len(df_treat)
        
        for idx, row in df_treat.iterrows():
            pid = row['cases.submitter_id']
            
            # === 生成增强文本列表 ===
            # text_versions: [Original, Aug1, Aug2... Aug10]
            text_versions = augment_treatment_info(row, max_augment=10)
            
            # === 编码 ===
            feat_list = encoder._encode_text(text_versions)
            
            patient_treatment_dict[pid] = feat_list
            
            if idx % 100 == 0:
                print(f"Processed Treatments: {idx}/{total}")

        with open(OUTPUT_TREATMENT_PKL, 'wb') as f:
            pickle.dump(patient_treatment_dict, f)
        print(f"Saved {OUTPUT_TREATMENT_PKL}, patients count: {len(patient_treatment_dict)}")
        
        # 检查样本
        if len(patient_treatment_dict) > 0:
            first_pid = list(patient_treatment_dict.keys())[0]
            first_feats = patient_treatment_dict[first_pid]
            print(f"Sample Treatment [{first_pid}]: {len(first_feats)} variants (Original + Aug). Shape: {first_feats[0].shape}")

    else:
        print(f"Warning: {PATH_TREATMENT_CSV} not found. Skipping treatment processing.")

    # ---------------------------------------------------------
    # 任务 2: 处理 Pathology Reports (文本增强)
    # ---------------------------------------------------------
    if os.path.exists(PATH_REPORTS_CSV):
        print(f"\nProcessing Reports CSV: {PATH_REPORTS_CSV}")
        df_reports = pd.read_csv(PATH_REPORTS_CSV)
        
        pathology_dict = {} 
        valid_data = [] 
        
        # 预处理数据列表
        for idx, row in df_reports.iterrows():
            pid = str(row['patient_id']).strip()
            report_txt = str(row['report_text']) if pd.notna(row['report_text']) else ""
            annot_txt = str(row['annotation_text']) if pd.notna(row['annotation_text']) else ""
            
            # 简单拼接
            combined_text = (report_txt + " " + annot_txt).strip()
            
            if combined_text:
                valid_data.append((pid, combined_text))
        
        print(f"Found {len(valid_data)} valid pathology reports. Encoding...")

        for idx, (pid, txt) in enumerate(valid_data):
            # === 数据增强 ===
            text_versions = augment_pathology_text(txt)
            
            # === 编码 ===
            feat_list = encoder._encode_text(text_versions)
            
            pathology_dict[pid] = feat_list
            
            if idx % 50 == 0:
                print(f"Processed Pathology: {idx}/{len(valid_data)}")

        with open(OUTPUT_PATHOLOGY_PKL, 'wb') as f:
            pickle.dump(pathology_dict, f)
        print(f"Saved {OUTPUT_PATHOLOGY_PKL}, patients count: {len(pathology_dict)}")
        
    else:
        print(f"Warning: {PATH_REPORTS_CSV} not found. Skipping pathology processing.")

if __name__ == "__main__":
    main()