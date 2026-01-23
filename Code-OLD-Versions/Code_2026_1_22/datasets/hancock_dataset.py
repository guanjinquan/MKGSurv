import os
import json
from typing import Dict, Any, List, Tuple, Optional
import sys

sys.path.append("/home/Zhengzx/MedAlignFusion/Code")

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import copy
import joblib
from datasets.dataset_base import MultiModalDataset


class HANCOCKDataset(MultiModalDataset):
    # VALID_MODALITIES = [
    #     "image-pathology",
    #     "text-clinical",
    #     "text-treatment",
    #     "tabular-clinical-52",
    #     "tabular-pathology-17"
    # ]
    PRE_OP_MODALITIES = [
        "text-clinical",
        "tabular-clinical-52", 
        "image-pathology", 
    ]

    POST_OP_MODALITIES = [
        "text-treatment",
        "tabular-pathology-17", 
    ]

    VALID_MODALITIES = PRE_OP_MODALITIES + POST_OP_MODALITIES

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
            # print(f"Successfully loaded {path}, type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")  # 添加这行
            return data
        except Exception as e:
            print(f"Error loading data file {path}: {e}")
            raise

    def __init__(self, args=None, mode: str = "train", modalities: str = "all", fold: int = None):
        """
        Initializes the dataset.

        Args:
            mode (str): The dataset mode, one of 'train', 'valid', or 'test'.
            modalities (str): A comma-separated string of modalities to load,
                              e.g., "image-pathology,text-clinical". "all" loads all available.
            fold (int): The fold index (0-4) for cross-validation.
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

        self.dataset_dir = os.path.join(project_root, "Data", "HANCOCK")
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        self.text_data_dir = os.path.join(self.dataset_dir, "TextData")
        self.structured_data_dir = os.path.join(self.dataset_dir, "StructuredData")

        print(f"Path Debugging:")
        print(f"  - Script Location: {current_file_dir}")
        print(f"  - Calculated Dataset Dir: {self.dataset_dir}")

        for dir_path in [self.processed_dir, self.structured_data_dir, self.text_data_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
        # --- Parse Modalities ---
        self.modalities = self.parse_modalities(modalities)
        # 设置do_mixup标志，与TCGA-LUAD保持一致
        self.do_mixup = (getattr(args, 'do_mixup', False) or getattr(args, 'do_mixup_only_treatment', False)) and len(self.modalities) > 1 and self.mode == "train"
        print(f"Active modalities: {self.modalities}")

        # --- Load Patient Split ---
        split_file = os.path.join(self.processed_dir, "hancock_patients_5fold.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)
            if 'folds' not in splits:
                raise KeyError(f"JSON structure error: 'folds' key missing in {split_file}")
            self.patient_ids = splits['folds'][fold][mode]

        print(f"Mode: {mode} | Fold: {fold} | Patients: {len(self.patient_ids)}")

        # --- Load Data Sources ---
        self.loaded_features = {}

        self.pickle_map = {
            "text-clinical": "features_text_clinical.pkl",
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

        # Load CSVs (Tabular Data & Labels)
        self._load_tabular_and_labels()
        
        # 添加虚拟治疗选项以满足tester要求
        # 即使数据集没有实际的治疗方案，也需要这些属性来避免tester出错
        self._setup_dummy_treatment_options()

    def _setup_dummy_treatment_options(self):
        """
        Setup dummy treatment options for compatibility with tester module.
        Even though HANCOCK dataset doesn't have actual treatment options,
        these attributes are needed to prevent errors in the tester.
        """
        # 定义虚拟的治疗选项
        self.TREATMENT_OPTIONS = ["Standard_Care"]
        # 创建对应的one-hot编码
        self.TREATMENT_OPTIONS_ONEHOT = [np.array([1.0])]
        # 创建对应的特征嵌入（虚拟的零向量）
        self.TREATMENT_OPTIONS_FEAT = [np.zeros(768)]  # 假设嵌入维度为768

    def _load_tabular_and_labels(self):
        try:
            # 只使用一种方法读取JSON，先尝试标准格式，失败后再试lines=True格式
            clinical_path = os.path.join(self.structured_data_dir, "clinical_data.json")
            pathological_path = os.path.join(self.structured_data_dir, "pathological_data.json") 
            blood_path = os.path.join(self.structured_data_dir, "blood_data.json")
            
            # 尝试读取临床数据
            try:
                self.clinical_df = pd.read_json(clinical_path)
            except ValueError:
                self.clinical_df = pd.read_json(clinical_path, lines=True)
            
            # 尝试读取病理数据
            try:
                self.pathological_df = pd.read_json(pathological_path)
            except ValueError:
                self.pathological_df = pd.read_json(pathological_path, lines=True)
                
            # 尝试读取血液检查数据
            try:
                self.blood_df = pd.read_json(blood_path)
            except ValueError:
                self.blood_df = pd.read_json(blood_path, lines=True)

            # 统一 Patient ID 类型为字符串
            self.clinical_df['patient_id'] = self.clinical_df['patient_id'].astype(str)
            self.pathological_df['patient_id'] = self.pathological_df['patient_id'].astype(str)
            self.blood_df['patient_id'] = self.blood_df['patient_id'].astype(str)
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            raise

        def _process_row(row, columns):
            tabular_data = []
            columns = sorted(columns)
            for col in columns:
                value = row.get(col)
                if value is None:
                    tabular_data.append(-1.0)
                else:
                    numeric_val = pd.to_numeric(value, errors='coerce')
                    if pd.isna(numeric_val):
                        tabular_data.append(-1.0)
                    else:
                        tabular_data.append(float(numeric_val))
            return tabular_data

        # Process Clinical Tabular
        self.clinical_tabular_dict = {}
        if "tabular-clinical-52" in self.modalities:
            self.clinical_tabular_columns = [
                "year_of_initial_diagnosis",
                "age_at_initial_diagnosis",
                "sex",
                "smoking_status",
                "primarily_metastasis",
                "first_treatment_intent",
                "first_treatment_modality",
                "days_to_first_treatment",
                "adjuvant_treatment_intent",
                "adjuvant_radiotherapy",
                "adjuvant_radiotherapy_modality",
                "adjuvant_systemic_therapy",
                "adjuvant_systemic_therapy_modality",
                "adjuvant_radiochemotherapy"
            ]

            # Add blood test columns
            blood_columns = ['Basophils', 'Basophils %', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Eosinophils',
                             'Eosinophils %', 'Erythrocytes', 'Glomerular filtration rate', 'Glucose', 'Granulocytes',
                             'Granulocytes %', 'Hematocrit', 'Hemoglobin', 'INR', 'Immature Granulocytyes',
                             'Leukocytes', 'Lymphocytes', 'Lymphocytes %', 'MCH', 'MCV', 'MHCH', 'MPV', 'Magnesium',
                             'Monocytes', 'Monocytes %', 'Normoblasts', 'PDW', 'PLCR', 'PT', 'Platelets', 'Potassium',
                             'RDW', 'Sodium', 'Thrombin time', 'Urea', 'aPPT']

            all_clinical_columns = self.clinical_tabular_columns + blood_columns

            for _, row in self.clinical_df.iterrows():
                pid = str(row['patient_id'])  # 确保PID是字符串类型
                # Merge clinical and blood data for this patient
                clinical_data = row.to_dict()

                # Get blood data for this patient
                patient_blood_data = self.blood_df[self.blood_df['patient_id'] == pid]
                blood_values = {}
                for _, blood_row in patient_blood_data.iterrows():
                    analyte = blood_row['analyte_name']
                    value = blood_row['value']
                    if analyte in blood_columns:
                        blood_values[analyte] = value

                # Combine clinical and blood data
                combined_data = {**clinical_data, **blood_values}
                self.clinical_tabular_dict[pid] = _process_row(combined_data, all_clinical_columns)

        # Process Pathological Tabular
        self.pathology_tabular_dict = {}
        if "tabular-pathology-17" in self.modalities:
            self.pathology_tabular_columns = [
                "primary_tumor_site",
                "pT_stage",
                "pN_stage",
                "grading",
                "hpv_association_p16",
                "number_of_positive_lymph_nodes",
                "number_of_resected_lymph_nodes",
                "perinodal_invasion",
                "lymphovascular_invasion_L",
                "vascular_invasion_V",
                "perineural_invasion_Pn",
                "resection_status",
                "resection_status_carcinoma_in_situ",
                "carcinoma_in_situ",
                "closest_resection_margin_in_cm",
                "histologic_type",
                "infiltration_depth_in_mm"
            ]

            for _, row in self.pathological_df.iterrows():
                pid = str(row['patient_id'])  # 确保PID是字符串类型
                self.pathology_tabular_dict[pid] = _process_row(row, self.pathology_tabular_columns)

        # Process Labels
        self.patient_labels = {}
        for _, row in self.clinical_df.iterrows():
            pid = str(row['patient_id'])  # 确保PID是字符串类型
            # Extract recurrence information
            has_recurrence = row.get('recurrence') == 'yes'
            time_to_recurrence = row.get('days_to_recurrence')
            time_to_last_info = row.get('days_to_last_information')

            event_time = -1.0
            event_flag = 0  # 0 means censored, 1 means event occurred

            if has_recurrence and pd.notna(time_to_recurrence):
                # The patient had a recurrence event
                event_flag = 1
                event_time = float(time_to_recurrence)
            elif pd.notna(time_to_last_info):
                # No recurrence event, use last follow-up time. Always censored.
                event_flag = 0
                event_time = float(time_to_last_info)

            self.patient_labels[pid] = {
                'label_time': event_time,
                'label_event': event_flag
            }

    def __len__(self) -> int:
        """Returns the total number of patients in the dataset."""
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_sample(idx)

    def get_sample(self, requested_idx: int) -> Dict[str, Any]:
        do_mixup = requested_idx >= len(self) and self.do_mixup   # if idx >= len(self), then we're doing mixup augmentation
        idx = requested_idx % len(self) 

        patient_id = str(self.patient_ids[idx])  # 确保patient_id是字符串类型
        output_dict = {"pid": patient_id}

        # --- Labels ---
        try:
            survival_info = self.patient_labels[patient_id]
            event = int(survival_info['label_event'])
            time_days = float(survival_info['label_time'])
        except KeyError:
            # Skip patient if labels are missing
            # print(f"Warning: Labels missing for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        output_dict['labels'] = {
            'do_mixup': do_mixup,  
            'label_time': time_days,
            'label_event': event,
        }

        # --- Load Modalities ---
        modalities_found = 0

        for mod in self.modalities:
            feature_data = None

            # Case A: Tabular
            if mod == "tabular-clinical-52":
                feature_data = self.clinical_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            elif mod == "tabular-pathology-17":
                feature_data = self.pathology_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            # Case B: Pickles
            elif mod in self.loaded_features:
                data_dict = self.loaded_features[mod]
                if patient_id in data_dict:
                    raw_feat = data_dict[patient_id]
                    
                    if isinstance(raw_feat, list) and len(raw_feat) > 0 and torch.is_tensor(raw_feat[0]):
                        if self.mode == 'train':  # Select one of the augmented features
                            feature_data = random.choice(raw_feat)
                        else:
                            feature_data = raw_feat[0]
                    elif isinstance(raw_feat, (np.ndarray, list)):
                        feature_data = torch.tensor(raw_feat, dtype=torch.float32)
                    elif isinstance(raw_feat, torch.Tensor):
                        feature_data = raw_feat.float()
                    elif isinstance(raw_feat, dict):
                        # 处理字典类型的特征，提取值并转换为张量
                        # 假设字典包含嵌套的特征数据
                        if 'features' in raw_feat:
                            feat_value = raw_feat['features']
                            if isinstance(feat_value, (np.ndarray, list)):
                                feature_data = torch.tensor(feat_value, dtype=torch.float32)
                            elif isinstance(feat_value, torch.Tensor):
                                feature_data = feat_value.float()
                            else:
                                # 如果无法识别的数据类型，则跳过该模态
                                feature_data = None
                        elif 'embeddings' in raw_feat:
                            # 处理文本特征，其使用'embeddings'键而不是'features'键
                            feat_value = raw_feat['embeddings']
                            if isinstance(feat_value, (np.ndarray, list)):
                                feature_data = torch.tensor(feat_value, dtype=torch.float32)
                            elif isinstance(feat_value, torch.Tensor):
                                feature_data = feat_value.float()
                            else:
                                # 如果无法识别的数据类型，则跳过该模态
                                feature_data = None
                        else:
                            # 如果字典不包含'features'键，尝试直接将值转换为张量
                            dict_values = list(raw_feat.values())
                            if len(dict_values) > 0:
                                try:
                                    feature_data = torch.tensor(dict_values, dtype=torch.float32)
                                except (TypeError, ValueError):
                                    # 如果无法转换为张量，则跳过该模态
                                    feature_data = None
                            else:
                                feature_data = None
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
                        modalities_found += 1
                else:
                    modalities_found += 1
        
        # --- Integrity Check ---
        if modalities_found == 0:
            # print(f"Warning: No valid modalities found for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        if do_mixup:
            other_item_idx = random.randint(0, len(self) - 1)
            if other_item_idx == idx: 
                other_item_idx = (idx + 1) % len(self)
            output_dict = self.mixup_data(output_dict, self.get_sample(other_item_idx))

        # 添加治疗相关信息到标签中（即使数据集本身没有真实的治疗信息）
        if "text-treatment" in self.modalities:
            output_dict['labels']['treatment_type'] = "Standard_Care"
            output_dict['labels']['treatment_type_onehot'] = torch.tensor([1.0], dtype=torch.float32)

        return output_dict

    def mixup_data(self, ori_data, other_data):
        """
        Performs multimodal mixup by swapping a subset of modalities from 'other_data' to 'ori_data'.
        Critically, it assigns the labels corresponding to the higher risk patient.
        """
        # 1. Determine which modalities to swap
        mixup_modalities = set()

        # 不再判断do_mixup_only_treatment，对所有模态选择max k进行mixup
        # Handle edge case where only 1 modality exists (randint(1, 0) would fail)
        max_k = max(1, len(self.modalities) - 1)
        k = random.randint(1, max_k)
        # Select k modalities to swap from other_data -> ori_data
        mixup_modalities.update(random.sample(self.modalities, k=k))

        # 2. Swap Features
        for mod in mixup_modalities:
            # Only swap if the other patient actually has data for this modality
            if other_data.get(mod) is not None:
                ori_data[mod] = other_data[mod].clone() if hasattr(other_data[mod], 'clone') else copy.deepcopy(other_data[mod])
            else:
                ori_data[mod] = None  # Drop the data for this modality
        
        # 3. Swap Labels (Risk-Based Selection)
        # Definition of Higher Risk:
        #   1. Event (1) > Censored (0)
        #   2. If events are same, Shorter Time > Longer Time
        
        t1 = ori_data['labels']['label_time']
        e1 = ori_data['labels']['label_event']
        
        t2 = other_data['labels']['label_time']
        e2 = other_data['labels']['label_event']
        
        use_other_labels = False

        if e1 == 1 and e2 == 0:
            # Patient 1 has event, Patient 2 censored. 1 is riskier. Keep 1.
            use_other_labels = False
        elif e1 == 0 and e2 == 1:
            # Patient 2 has event, Patient 1 censored. 2 is riskier. Swap.
            use_other_labels = True
        elif e1 == e2:
            # Both Event or Both Censored. 
            # The one with SHORTER time is considered higher risk (died sooner) 
            # or more conservative for censored (less info, assume riskier).
            if t2 < t1:
                use_other_labels = True
            else:
                use_other_labels = False
        else:
            raise ValueError("Invalid label combination")
        
        if use_other_labels:
            ori_data['labels'] = copy.deepcopy(other_data['labels'])
            
        ori_data['labels']['do_mixup'] = True
        return ori_data

    def get_survival_bins(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        """
        self.observed_years = 20 * 365.0
        self.num_time_bins = 20
        self.time_bins = np.linspace(0, self.observed_years, self.num_time_bins + 1)

        labels_y = []
        for patient_id in self.patient_ids:
            if patient_id not in self.patient_labels:
                continue
            survival_info = self.patient_labels[patient_id]
            time_days = float(survival_info['label_time'])
            event_status = int(survival_info['label_event'])

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


# # 创建一个简单的测试脚本用于调试
# def test_dataset():
#     print("Testing HANCOCK Dataset")
    
#     # 创建一个简单的args对象用于测试
#     class Args:
#     def __init__(self):
#         self.do_mixup = False
#         self.do_mixup_only_treatment = False
    
#     args = Args()
    
#     try:
#         dataset = HANCOCKDataset(args=args, mode="train", modalities="all", fold=0)
#         print(f"Dataset loaded successfully with {len(dataset)} patients")
        
#         # 测试获取第一个样本
#         print("Testing data retrieval...")
#         sample = dataset[0]
#         print("Sample keys:", list(sample.keys()))
#         print("Sample retrieved successfully")
        
#         # 检查标签
#         if 'labels' in sample:
#             print("Labels:", sample['labels'])
            
#         # 检查模态数据
#         for modality in dataset.modalities:
#             if modality in sample:
#                 data = sample[modality]
#                 if data is not None:
#                     if isinstance(data, torch.Tensor):
#                         print(f"{modality}: Tensor with shape {data.shape}")
#                     else:
#                         print(f"{modality}: {type(data)}")
#                 else:
#                     print(f"{modality}: None")
#             else:
#                 print(f"{modality}: Not present in sample")
                
#     except Exception as e:
#         print(f"Error during testing: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     test_dataset()