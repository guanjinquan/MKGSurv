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
BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed"

# 输入文件路径
PATH_TREATMENT_CSV = os.path.join(BASE_DIR, "text_treatment.csv")
PATH_REPORTS_CSV = os.path.join(BASE_DIR, "tcga_brca_reports.csv")

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
        输入: 一个包含多个文本字符串的列表
        输出: 一个列表, 包含对应的特征 Tensor。
        """
        batch_size = len(texts_list)
        chunk_payload = 510 
        
        all_chunks = []
        mapping_info = [] 
        
        # 1. 预处理与分块
        for i, text in enumerate(texts_list):
            item_specific_chunks = [] 
            
            if isinstance(text, str) and text.strip():
                clean_text = text.strip()
                token_ids = self.tokenizer.encode(clean_text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)
            
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
            else:
                mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
                all_chunks.extend(item_specific_chunks)

        if not all_chunks:
            # 如果全是空文本，返回一批空张量
            return [torch.zeros(0, self.embed_dim) for _ in range(batch_size)]

        # 2. 批量编码
        chunk_texts = [self.tokenizer.decode(c, clean_up_tokenization_spaces=True) for c in all_chunks]
        
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
            
            batch_pooled = bert_outputs.last_hidden_state[:, 0, :]
            pooled_outputs.append(batch_pooled.cpu())
            
        if pooled_outputs:
            pooled = torch.cat(pooled_outputs, dim=0)
        else:
            pooled = torch.zeros(0, self.embed_dim)

        # 3. 重组结果
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
                # 这是一个潜在的风险点：这里返回了 (0, 768) 的空张量
                output_list.append(torch.zeros((0, self.embed_dim)))

        return output_list

# =================数据增强工具函数=================

def clean_term(term):
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
    target_cols = [
        'treatments.therapeutic_agents',
        'treatments.treatment_intent_type',
        'treatments.treatment_or_therapy',
        'treatments.treatment_type'
    ]
    
    all_terms = []
    for col in target_cols:
        if col in row:
            val = str(row[col])
            parts = val.split('+')
            for p in parts:
                clean = clean_term(p)
                if clean:
                    all_terms.append(clean)
    
    if not all_terms:
        return [""] 
    
    original_text = ", ".join(all_terms)
    results = [original_text]
    
    if len(all_terms) > 1:
        seen_texts = {original_text}
        for _ in range(max_augment * 2): 
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
    if not isinstance(text, str) or not text.strip():
        return []
    
    versions = [text]
    for delimiter in ['. ', '+', '\n']:
        if delimiter in text:
            parts = [p.strip() for p in text.split(delimiter) if p.strip()]
            if len(parts) > 1:
                for _ in range(10):
                    random.shuffle(parts)
                    new_txt = f"{delimiter}".join(parts)
                    if new_txt not in versions:
                        versions.append(new_txt)
                break 
                
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
            combined_text = "N/A"

            if "llm_polished_report" in row:
                combined_text = row['llm_polished_report']
            else:
                print("Warning!! Column llm_polished_report not found")
            
            # 这里简单过滤一下非字符串
            if not isinstance(combined_text, str):
                combined_text = ""
                
            valid_data.append((pid, combined_text))
        
        print(f"Found {len(valid_data)} valid pathology reports. Encoding...")

        for idx, (pid, txt) in enumerate(valid_data):
            # === 数据增强 ===
            text_versions = augment_pathology_text(txt)
            
            # 如果增强后列表为空（原始文本也是空的），手动加一个空字符串占位
            if not text_versions:
                text_versions = [""]

            # === 编码 ===
            feat_list = encoder._encode_text(text_versions)

            # ========================================================
            # TODO: Debug Logic Added Here
            # ========================================================
            clean_feat_list = []
            
            for i, feat in enumerate(feat_list):
                # 检查 1: 是否为 None
                if feat is None:
                    print(f"❌ [DEBUG] PID {pid}: Feature index {i} is None. Skipping.")
                    continue
                
                # 检查 2: 是否为空张量 (numel=0)
                # 这通常发生在输入文本为空字符串时，ClinicalBertEncoder 返回 (0, 768)
                if feat.numel() == 0:
                    # 这是一个非常严格的过滤：如果特征为空，说明并没有提取到有效信息
                    # 为了防止训练时维度报错，这里我们不加入这个空特征
                    # print(f"⚠️ [DEBUG] PID {pid}: Feature index {i} is empty tensor {feat.shape}. Skipping.")
                    continue
                
                # 检查 3: 维度一致性
                if feat.shape[-1] != encoder.embed_dim:
                    print(f"❌ [DEBUG] PID {pid}: Feature index {i} has wrong dim {feat.shape}. Expected {encoder.embed_dim}. Skipping.")
                    continue
                
                clean_feat_list.append(feat)
            
            # 补救措施：如果清洗后列表为空（说明该病人没有任何有效文本特征）
            # 我们必须人为制造一个全是 0 的特征向量 (1, 768)
            # 否则后续 DataLoader 处理时会报错，或者训练时 input_dim 变为 0
            if len(clean_feat_list) == 0:
                print(f"🚨 [CRITICAL WARNING] PID {pid}: No valid features found after cleaning! Creating fallback zero-vector.")
                fallback_feat = torch.zeros((1, encoder.embed_dim)) # (1, 768)
                clean_feat_list.append(fallback_feat)
            
            # 更新为清洗后的列表
            feat_list = clean_feat_list
            # ========================================================
            
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