import os
import json
import random
import joblib
import sys
import numpy as np
import pandas as pd
import torch
import copy
from typing import List, Dict, Any
sys.path.append("/home/Guanjq/NewWork/MedAlignFusion/Code")
from datasets.dataset_base import MultiModalDataset




class OSCCSurvInHouseDataset(MultiModalDataset):

    VALID_MODALITIES = [
        "image-pathology",            
        "text-clinical",            
        "tabular-clinical-metadata-4",        #  (From CSV)
        "tabular-clinical-history-9",         #  (From CSV)
        "tabular-clinical-blood-5",           #  (From CSV)     
 
        "text-pathology",          
        "text-treatment",           
        "tabular-pathology-cell-16",        # (From CSV)
        "tabular-pathology-immunohistochemic-5", # (From CSV)  
    ]

    def _read_pickle(self, path: str) -> Any:
        if not os.path.exists(path):
            print(f"Warning: Pickle file not found at: {path}")
            return None
        try:
            data = joblib.load(path)
            return data
        except Exception as e:
            print(f"Error loading data file {path}: {e}")
            return None

    def __init__(self, args, mode="train", modalities="all", fold=None):
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"
        assert 0 <= fold <= 4, "fold must be an integer between 0 and 4"

        self.args = args
        self.mode = mode
        self.fold = fold
        
        # --- Path Definitions ---
        self.dataset_dir = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        
        # --- 1. Parse Modalities ---
        self.modalities = self.parse_modalities(modalities)
        # 仅在训练模式且模态数大于1时启用 Mixup
        self.do_mixup = getattr(args, 'do_mixup', False) and self.mode == "train"
        self.mixup_alpha = getattr(args, 'mixup_alpha', 1.0) # 默认为1.0

        # --- 2. Load Clinical Data (CSV) ---
        self._load_clinical_csv()

        # --- 3. Load Patient Split ---
        self._load_split_file()

        # --- 4. Load Pre-extracted Features (Pickles) ---
        self.loaded_features = {}
        self.pickle_map = {
            "image-pathology": f"feature_image_pathology_fold_{fold+1}.pkl",
            "text-clinical": "features_text_clinical.pkl",
            "text-pathology": "features_text_pathology.pkl",
            "text-treatment": "features_text_treatment.pkl"
        }

        for mod in self.modalities:
            if mod in self.pickle_map:
                pkl_path = os.path.join(self.processed_dir, self.pickle_map[mod])
                data = self._read_pickle(pkl_path)
                if data is not None:
                    self.loaded_features[mod] = data
        
        # --- 5. Pre-process Tabular Data ---
        self.tabular_features = {} 
        self._preprocess_tabular_data()

        # --- 6. Knowledge Features ---
        knowledge_file = os.path.join(self.processed_dir, "features_medical_knowledge.pkl")
        self.knowledge_dict = self._read_pickle(knowledge_file)

        # --- 7. Calculate Global Statistics for H-Mixup Weights ---
        # pi_star: 原始数据集的事件发生率
        # pi_hat:  H-Mixup 增强后预期的事件发生率 (通过模拟计算)
        self.pi_star = 0.5
        self.pi_hat = 0.5
        
        if self.mode == 'train':
            self._calculate_statistics()

        print(f"Dataset initialized: mode='{self.mode}'. Count: {len(self.items)}.")

    def _calculate_statistics(self):
        """
        计算原始事件率(pi_star)并模拟一次Mixup过程以估算增强后的事件率(pi_hat)。
        这样可以在 getitem 中直接返回全局校正后的权重。
        """
        # 1. 提取所有样本的 Time 和 Event
        times = []
        events = []
        
        for item in self.items:
            has_recurrence = item.get('recurrence') == 'yes'
            t_rec = item.get('days_to_recurrence')
            t_last = item.get('days_to_last_information')
            
            e, t = 0, -1.0
            if has_recurrence and pd.notna(t_rec):
                e, t = 1, float(t_rec)
            elif pd.notna(t_last):
                e, t = 0, float(t_last)
            
            if t >= 0:
                times.append(t)
                events.append(e)
        
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, requested_idx: int) -> Dict[str, Any]:
        # 1. 确定是否进行 Mixup
        # 只有当索引超出原始长度时才认为是要做 Mixup
        do_mixup = requested_idx >= len(self) and self.do_mixup
        idx = requested_idx % len(self)
        
        item_info = self.items[idx]
        pid_str = str(item_info['pid'])
        
        # 2. 获取基础 Label
        has_recurrence = item_info.get('recurrence') == 'yes'
        time_to_recurrence = item_info.get('days_to_recurrence')
        time_to_last_info = item_info.get('days_to_last_information')

        event_time = -1.0
        event_flag = 0 

        if has_recurrence and pd.notna(time_to_recurrence):
            event_flag = 1
            event_time = float(time_to_recurrence)
        elif pd.notna(time_to_last_info):
            event_flag = 0
            event_time = float(time_to_last_info)

        # 3. 初始化输出，默认权重为 1.0 (无 Mixup)
        output_dict = {
            "pid": pid_str,
            "labels": {
                'label_time': event_time,
                'label_event': event_flag,
                'sample_weight': 1.0,
            },
        }

        if self.args.use_medical_knowledge:
            output_dict["medical-knowledge"] = self.knowledge_dict.get(pid_str, None)
        else:
            kdata = self.knowledge_dict.get(pid_str, None)
            output_dict["medical-knowledge"] = {}
            for k, v in kdata.items():
                output_dict["medical-knowledge"][k] = {
                    "score": v['score'] if self.mode != 'train' else 0.0,
                    "knowledge": torch.randn_like(v['knowledge'])
                }

        # 4. 加载特征 
        modalities_found = 0
        for mod in self.modalities:
            feature_data = None

            if mod in self.loaded_features:
                data_dict = self.loaded_features[mod]
                if pid_str in data_dict:
                    raw_feat = data_dict[pid_str]
                    if isinstance(raw_feat, list) and len(raw_feat) > 0 and torch.is_tensor(raw_feat[0]):
                        feature_data = random.choice(raw_feat) if self.mode == 'train' else raw_feat[0]
                    elif isinstance(raw_feat, (np.ndarray, list)):
                        feature_data = torch.tensor(raw_feat, dtype=torch.float32)
                    elif isinstance(raw_feat, torch.Tensor):
                        feature_data = raw_feat.float()

            elif mod in self.tabular_features:
                if pid_str in self.tabular_features[mod]:
                    feature_data = self.tabular_features[mod][pid_str]

            if isinstance(feature_data, torch.Tensor):
                if feature_data.dim() == 1: feature_data = feature_data.unsqueeze(0)
            
            output_dict[mod] = feature_data
            if feature_data is not None: 
                modalities_found += 1

        if modalities_found == 0:
            return self.get_sample((idx + 1) % len(self))

        # 5. 执行 Mixup
        if do_mixup:
            other_item_idx = random.randint(0, len(self) - 1)
            if other_item_idx == idx: 
                other_item_idx = (idx + 1) % len(self)
            other_sample = self.get_sample(other_item_idx)
            
            # Mixup 数据并计算权重
            output_dict = self.mixup_data(output_dict, other_sample)

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
    def _load_clinical_csv(self):
        csv_path = os.path.join(self.dataset_dir, "clinical_data.csv")
        try:
            self.clinical_df = pd.read_csv(csv_path)
            # Ensure PID is treated as string for consistency
            if 'PID' in self.clinical_df.columns:
                # Handle float-like strings "123.0" -> "123"
                self.clinical_df['PID'] = self.clinical_df['PID'].astype(str).apply(lambda x: str(int(float(x))) if x.replace('.', '', 1).isdigit() else x)
                self.clinical_df.set_index('PID', inplace=True)
            print("Successfully loaded clinical data CSV.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Clinical data file not found at {csv_path}")

    def _load_split_file(self):
        metadata_path = os.path.join(self.dataset_dir, "oscc_recurrence_survival_data.json")
        split_path = os.path.join(self.dataset_dir, "split_OOD_5fold.json")
        
        with open(metadata_path, 'r') as f:
            all_patients_info = {str(item['pid']): item for item in json.load(f)}

        with open(split_path, 'r') as f:
            split_data = json.load(f)[f'fold_{self.fold+1}']

        target_pids = [str(pid) for pid in split_data[self.mode]]
        
        self.items = []
        for pid in target_pids:
            if pid in all_patients_info and pid in self.clinical_df.index:
                # 基本数据来自 JSON，但扩展信息来自 CSV
                self.items.append(all_patients_info[pid])
        
        if self.mode == "train":
            random.shuffle(self.items)

    def _preprocess_tabular_data(self):
        sources_with_columns = {
            "tabular-clinical-metadata-4": [
                "Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)"
            ],
            "tabular-clinical-history-9": [
                "AlcoholHistory(0no/1yes)", "SmokingHistory(0no/1yes)", "BetelNutHistory(0no/1yes)", 
                "PreoperativeHistory(0no/1yes)", "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
                "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)", "NeckMass(+)"
            ],
            "tabular-clinical-blood-9": [
                "PreopWBC", "PreopHemoglobin", "PreopPotassium", "PreopAlbumin", "PreopVitaminD",
                "PostopWBC", "PostopHemoglobin", "PostopPotassium", "PostopAlbumin"
            ],
            "tabular-pathology-cell-16": [
                "TumorT", "TumorN", "TumorM", "TumorDifferentiation(1high/2med/3low)",
                "CancerThrombus(0/1)", "SurroundingTissueInvasion(0/1)",  "LNM(0/1)", 
                "AccessoryChain(+)", "VascularInvasion(+)", "PerineuralInvasion(+)", 
                "Metastasis(0no/1yes)", "IA(+)", "IB(+)", "IIA(+)", "IIB(+)", "III(+)"
            ],
            "tabular-pathology-immunohistochemic-5": [
                "Ki-67", "CK5_6(0/1)", "P63(0/1)", "P16(0/1)", "HPV(0/1)"
            ]
        }

        def process_special_column(column, value):
            if column == "TumorT":
                try:
                    return float(value)
                except:
                    if str(value) == '4a': return 4.0
                    elif str(value) == 'Tis': return 0.0
                    else: return -1.0
            return value

        active_tabular = [m for m in self.modalities if "tabular" in m]

        for mod_name in active_tabular:
            if mod_name not in sources_with_columns:
                continue
            
            self.tabular_features[mod_name] = {}
            columns = sorted(sources_with_columns[mod_name])

            for pid in self.clinical_df.index:
                row = self.clinical_df.loc[pid]
                vec = []
                for col in columns:
                    val = row.get(col, -1)
                    val = process_special_column(col, val)
                    try:
                        f_val = float(val)
                        if pd.isna(f_val): vec.append(-1.0)
                        else: vec.append(f_val)
                    except:
                        vec.append(-1.0)
                
                tensor = torch.tensor(vec, dtype=torch.float32)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                self.tabular_features[mod_name][str(pid)] = tensor

    def parse_modalities(self, modalities_str: str) -> List[str]:
        if modalities_str == "all":
            return sorted(list(self.VALID_MODALITIES))
        parsed = [m.strip() for m in modalities_str.split(',')]
        return [m for m in parsed if m in self.VALID_MODALITIES]

    def get_survival_bins(self):
        self.num_time_bins = 4
        self.observed_years = 20 * 365.0
        self.time_bins = np.linspace(0, self.observed_years, self.num_time_bins + 1)

        labels_y = []
        for index in range(len(self.items)):
            label_info = self.items[index]
            
            has_recurrence = label_info.get('recurrence') == 'yes'
            time_to_recurrence = label_info.get('days_to_recurrence')
            time_to_last_info = label_info.get('days_to_last_information')
            
            event_time = -1.0
            
            if has_recurrence and pd.notna(time_to_recurrence):
                if time_to_recurrence < self.observed_years:
                    event_time = time_to_recurrence
                else:
                    event_time = self.observed_years
            elif pd.notna(time_to_last_info):
                event_time = min(time_to_last_info, self.observed_years)
            
            time_bin = -1
            if event_time >= 0:
                time_bin = np.digitize(event_time, self.time_bins) - 1
                time_bin = min(time_bin, self.num_time_bins - 1)
            
            labels_y.append((int(time_bin), (1 if has_recurrence else 0)))
            
        return labels_y
    
    def get_active_modalities(self):
        return self.modalities

