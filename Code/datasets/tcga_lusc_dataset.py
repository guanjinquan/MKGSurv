import os
import json
from typing import Dict, Any, List, Tuple, Optional
import sys

# specific path as requested
sys.path.append("/home/Guanjq/NewWork/MedAlignFusion/Code") 

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import random
import joblib
from datasets.dataset_base import MultiModalDataset
import copy 



class TCGA_LUSC_Dataset(MultiModalDataset):

    VALID_MODALITIES = [
        "tabular-clinical-9", 
        "genomics-genomics",
        "image-pathology", 
  
        "text-pathology", 
        "text-treatment",
        "tabular-treatment-7", 
        "tabular-pathology-22", 
    ]

    def _read_pickle(self, path: str) -> Any:
        """
        Helper to load pickle/joblib files.
        """
        if not os.path.exists(path):
            print(f"Warning: Pickle file not found at: {path}")
            # Do not raise here, allow soft failure (returns None implicitly)
            return None
 
        try:
            data = joblib.load(path)
            return data
        except Exception as e:
            print(f"Error loading data file {path}: {e}")
            raise

    def __init__(self, args, mode: str = "train", modalities: str = "all", fold: int = None):
        """
        Args:
            mode (str): 'train', 'valid', or 'test'.
            modalities (str): Comma-separated string or "all".
            fold (int): The fold index (0-4).
        """
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"
        assert fold is not None, "Fold ID must be specified."
        assert 0 <= fold <= 4, "Fold ID must be between 0 and 4"

        random.seed(42)
        self.args = args
        self.mode = mode
        self.fold = fold
        
        # --- Path Construction ---
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, "../../")) 
        
        self.dataset_dir = os.path.join(project_root, "Data", "TCGA-LUSC")
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        
        print(f"Path Debugging:")
        print(f"  - Script Location: {current_file_dir}")
        print(f"  - Calculated Dataset Dir: {self.dataset_dir}")

        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed directory does not exist: {self.processed_dir}")
        
        
        # --- 0. Parse Modalities ---
        self.modalities = self.parse_modalities(modalities)
        self.do_mixup = getattr(args, 'do_mixup', False) and self.mode == "train"
        self.mixup_alpha = getattr(args, 'mixup_alpha', 1.0) # 默认为1.0

        print(f"Active modalities: {self.modalities}")

        # --- 2. Load Patient Split ---
        split_file = os.path.join(self.processed_dir, "lusc_patients_5fold.json") 
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)
            if 'folds' not in splits:
                raise KeyError(f"JSON structure error: 'folds' key missing in {split_file}")
            self.patient_ids = splits['folds'][fold][mode]
        
        print(f"Mode: {mode} | Fold: {fold} | Patients: {len(self.patient_ids)}")

        # --- 3. Load Data Sources (Pickles & CSVs) ---
        self.loaded_features = {} 

        self.pickle_map = {
            "genomics-genomics": "features_rna.pkl",
            "text-pathology": "features_text_pathology.pkl",
            "text-treatment": "features_text_treatment.pkl",
        }

        for mod in self.modalities:
            if mod == "image-pathology":
                fold_idx_for_file = self.fold + 1 
                pkl_name = f"features_image_pathology_fold{fold_idx_for_file}.pkl"
                pkl_path = os.path.join(self.processed_dir, pkl_name)
                # Note: If file doesn't exist, _read_pickle returns None
                data = self._read_pickle(pkl_path)
                if data is not None:
                    self.loaded_features[mod] = data
            
            elif mod in self.pickle_map:
                pkl_name = self.pickle_map[mod]
                pkl_path = os.path.join(self.processed_dir, pkl_name)
                data = self._read_pickle(pkl_path)
                if data is not None:
                    self.loaded_features[mod] = data
        
        # Knowledge Features
        knowledge_file = os.path.join(self.processed_dir, "features_medical_knowledge.pkl")
        self.knowledge_dict = self._read_pickle(knowledge_file)

        # Load CSVs (Tabular Data & Labels)
        self._load_tabular_and_labels()

        # ---Calculate Global Statistics for H-Mixup Weights ---
        # pi_star: 原始数据集的事件发生率
        # pi_hat:  H-Mixup 增强后预期的事件发生率 (通过模拟计算)
        self.pi_star = 0.5
        self.pi_hat = 0.5
        
        if self.mode == 'train':
            self._calculate_statistics()

    def _calculate_statistics(self):
        """
        计算原始事件率(pi_star)并模拟一次Mixup过程以估算增强后的事件率(pi_hat)。
        这样可以在 getitem 中直接返回全局校正后的权重。
        """
        # 1. 提取所有样本的 Time 和 Event
        times = []
        events = []
        
        for pid, info in self.patient_labels.items():
            event = info.get('DFS_event', 0)
            time = info.get('DFS_time', -1)
            
            if time >= 0:
                times.append(time)
                events.append(event)
        
        times = np.array(times)
        events = np.array(events)
        
        # 2. 计算原始事件率 pi_star
        self.pi_star = np.mean(events)
        self.pi_star = np.clip(self.pi_star, 1e-6, 1 - 1e-6)

        # 3. 模拟 H-Mixup 过程估算 pi_hat
        # 我们进行 N 次随机配对模拟，N = len(dataset) * 5 以保证统计稳定性
        n_sim = len(events) * 5
        sim_events = []
        
        for _ in range(n_sim):
            # 随机采样两个索引
            idx1 = np.random.randint(0, len(events))
            idx2 = np.random.randint(0, len(events))
            
            t1, e1 = times[idx1], events[idx1]
            t2, e2 = times[idx2], events[idx2]
            
            # 生成 lambda
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = np.clip(lam, 1e-4, 1 - 1e-4)
            
            # H-Mixup 逻辑: Time Scaling + Min
            t_scale_1 = t1 / lam
            t_scale_2 = t2 / (1 - lam)
            
            if t_scale_1 < t_scale_2:
                sim_events.append(e1)
            else:
                sim_events.append(e2)
        
        # 计算估算的增强后事件率 pi_hat
        self.pi_hat = np.mean(sim_events)
        self.pi_hat = np.clip(self.pi_hat, 1e-6, 1 - 1e-6)

    def _load_tabular_and_labels(self):
        try:
            self.clinical_df = pd.read_csv(os.path.join(self.processed_dir, "clinical_data_aggregated.csv"), dtype=str)
            self.treatment_df = pd.read_csv(os.path.join(self.processed_dir, "treatment_data_aggregated.csv"), dtype=str)
            self.pathology_df = pd.read_csv(os.path.join(self.processed_dir, "pathology_aggregated.csv"), dtype=str)
            self.labels_df = pd.read_csv(os.path.join(self.processed_dir, "lusc_patient_labels.csv"), dtype=str)
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise

        def _process_row(row, columns):
            tabular_data = []
            columns = sorted(columns)
            for col in columns:
                value = row[col]
                numeric_val = pd.to_numeric(value, errors='coerce')
                if pd.isna(numeric_val):
                    tabular_data.append(-1.0)
                else:
                    tabular_data.append(float(numeric_val))
            return tabular_data

        # Process Clinical Tabular
        self.clinical_tabular_dict = {}
        exclude_cols = ['cases.case_id', 'cases.submitter_id']
        
        if "tabular-clinical-9" in self.modalities:
            cols = [c for c in self.clinical_df.columns if c not in exclude_cols]
            for _, row in self.clinical_df.iterrows():
                pid = row['cases.submitter_id']
                self.clinical_tabular_dict[pid] = _process_row(row, cols)

        # Process Treatment Tabular
        self.treatment_tabular_dict = {}
        if "tabular-treatment-7" in self.modalities:
            cols = [c for c in self.treatment_df.columns if c not in exclude_cols]
            for _, row in self.treatment_df.iterrows():
                pid = row['cases.submitter_id']
                self.treatment_tabular_dict[pid] = _process_row(row, cols)

        self.pathology_tabular_dict = {}
        if "tabular-pathology-22" in self.modalities:
            cols = [c for c in self.pathology_df.columns if c not in exclude_cols]
            for _, row in self.pathology_df.iterrows():
                pid = row['cases.submitter_id']
                self.pathology_tabular_dict[pid] = _process_row(row, cols)

        # Process Labels
        self.patient_labels = {}
        for _, row in self.labels_df.iterrows():
            pid = row['cases.submitter_id']
            self.patient_labels[pid] = {
                "DFS_time": float(row["DFS_time"]),
                "DFS_event": float(row["DFS_event"]),
            }

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_sample(idx)

    def get_sample(self, requested_idx: int) -> Dict[str, Any]:
        do_mixup = requested_idx >= len(self) and self.do_mixup   
        idx = requested_idx % len(self) 

        patient_id = self.patient_ids[idx]

        # --- 1. Labels ---
        try:
            survival_info = self.patient_labels[patient_id]
            event = int(survival_info['DFS_event'])
            time_days = float(survival_info['DFS_time'])
        except KeyError:
            # Skip patient if labels are missing
            # print(f"Warning: Labels missing for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        output_dict = {
            "pid": patient_id,
            "labels": {
                'label_time': time_days,
                'label_event': event,
                'sample_weight': 1.0,
            },
        }

        if self.args.use_medical_knowledge:
            output_dict["medical-knowledge"] = self.knowledge_dict.get(patient_id, None)
        else:
            kdata = self.knowledge_dict.get(patient_id, None)
            output_dict["medical-knowledge"] = {}
            for k, v in kdata.items():
                output_dict["medical-knowledge"][k] = {
                    "score": v['score'] if self.mode != 'train' else 0.0,
                    "knowledge": torch.randn_like(v['knowledge'])
                }
                
        # --- 2. Load Modalities ---
        modalities_found = []

        for mod in self.modalities:
            feature_data = None

            # Case A: Tabular
            if mod == "tabular-clinical-9":
                feature_data = self.clinical_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            elif mod == "tabular-treatment-7":
                feature_data = self.treatment_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            elif mod == "tabular-pathology-22":
                feature_data = self.pathology_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            # Case B: Pickles
            elif mod in self.loaded_features:
                data_dict = self.loaded_features[mod]
                if patient_id in data_dict:
                    raw_feat = data_dict[patient_id]
                    
                    if isinstance(raw_feat, list) and len(raw_feat) > 0 and torch.is_tensor(raw_feat[0]):
                        if self.mode == 'train':  # Select one of the augmented feature
                            feature_data = random.choice(raw_feat)
                        else:
                            feature_data = raw_feat[0]
                    elif isinstance(raw_feat, (np.ndarray, list)):
                        feature_data = torch.tensor(raw_feat, dtype=torch.float32)
                    elif isinstance(raw_feat, torch.Tensor):
                        feature_data = raw_feat.float()
                    else:
                        raise ValueError(f"Unsupported feature type: {type(raw_feat)}")
                else:
                    feature_data = None

            # Assign to output
            output_dict[mod] = feature_data
            
            # Count valid
            if feature_data is not None:
                if isinstance(feature_data, torch.Tensor):
                    if feature_data.numel() > 0:
                        modalities_found.append(mod)
                else:
                    modalities_found.append(mod)
        
        # --- 3. Integrity Check ---
        if len(modalities_found) == 0:
            # print(f"Warning: No valid modalities found for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        if do_mixup:
            other_item_idx = random.randint(0, len(self) - 1)
            if other_item_idx == idx: 
                other_item_idx = (idx + 1) % len(self)
            output_dict = self.mixup_data(output_dict, self.get_sample(other_item_idx))

        return output_dict

    def mixup_data(self, ori_data, other_data):
        """
        Implementation of H-Mixup with Variable Length Padding and Weight Calculation.
        """
        # 1. 生成 Mixup 系数
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = np.clip(lam, 1e-4, 1 - 1e-4)

        # 2. 特征混合 (Variable Length Padding)
        for mod in self.modalities:
            if mod not in ori_data or ori_data[mod] is None: continue
            
            feat_a = ori_data[mod]
            feat_b = other_data.get(mod)

            if feat_b is None: continue

            if isinstance(feat_a, torch.Tensor) and isinstance(feat_b, torch.Tensor):
                if feat_a.shape != feat_b.shape:
                    len_a = feat_a.shape[0]
                    len_b = feat_b.shape[0]
                    max_len = max(len_a, len_b)
                    
                    pad_a = torch.zeros((max_len, feat_a.shape[1]), dtype=feat_a.dtype)
                    pad_a[:len_a] = feat_a
                    
                    pad_b = torch.zeros((max_len, feat_b.shape[1]), dtype=feat_b.dtype)
                    pad_b[:len_b] = feat_b
                    
                    mixed_feat = lam * pad_a + (1 - lam) * pad_b
                    ori_data[mod] = mixed_feat
                else:
                    ori_data[mod] = lam * feat_a + (1 - lam) * feat_b

        # 3. 标签混合 (H-Mixup Logic)
        t1 = float(ori_data['labels']['label_time'])
        e1 = int(ori_data['labels']['label_event'])
        t2 = float(other_data['labels']['label_time'])
        e2 = int(other_data['labels']['label_event'])

        t_scale_1 = t1 / lam
        t_scale_2 = t2 / (1 - lam)

        if t_scale_1 < t_scale_2:
            label_time = t_scale_1
            label_event = e1
        else:
            label_time = t_scale_2
            label_event = e2

        # 4. 计算样本权重 (Based on Global Statistics)
        # Weight = P*(y) / P_hat(y)
        # 如果是 Event (1): Weight = pi_star / pi_hat
        # 如果是 Censor(0): Weight = (1 - pi_star) / (1 - pi_hat)
        
        if label_event == 1:
            weight = self.pi_star / self.pi_hat
        else:
            weight = (1.0 - self.pi_star) / (1.0 - self.pi_hat)

        # 5. 更新 Labels
        ori_data['labels'] = {
            "label_event": label_event,
            "label_time": label_time,
            "sample_weight": float(weight)  # 直接返回计算好的权重
        }

        return ori_data

    def get_survival_bins(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        """
        self.num_time_bins = 4 
        self.observed_years = 20 * 365.0
        self.time_bins = np.linspace(0, self.observed_years, self.num_time_bins + 1)

        labels_y = []
        for patient_id in self.patient_ids:
            if patient_id not in self.patient_labels:
                continue
            survival_info = self.patient_labels[patient_id]
            time_days = float(survival_info['DFS_time']) 
            event_status = int(survival_info['DFS_event']) 

            event_time = min(time_days, self.observed_years)
            time_bin = np.digitize(event_time, self.time_bins) - 1
            time_bin = max(0, min(time_bin, self.num_time_bins - 1))
            
            labels_y.append((int(time_bin), event_status))

        return labels_y 

    def parse_modalities(self, modalities_str: str) -> List[str]:
        if modalities_str == "all":
            return sorted(list(self.VALID_MODALITIES))
        
        requested_modalities = modalities_str.split(',')
        parsed_list = []
        for mod in requested_modalities:
            mod = mod.strip()
            if mod in self.VALID_MODALITIES:
                parsed_list.append(mod)
            else:
                print(f"Warning: Modality '{mod}' not recognized and will be skipped.")
        return parsed_list

    def get_active_modalities(self):
        return self.modalities
    