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



class TCGA_LUAD_Dataset(MultiModalDataset):

    PRE_OP_MODALITIES = [
        "tabular-clinical-9", 
        "genomics-genomics",
        "image-pathology", 
    ]

    POST_OP_MODALITIES = [
        "text-pathology", 
        "text-treatment",
        "tabular-treatment-9", 
        "tabular-pathology-21", 
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
        
        self.dataset_dir = os.path.join(project_root, "Data", "TCGA-LUAD")
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        
        print(f"Path Debugging:")
        print(f"  - Script Location: {current_file_dir}")
        print(f"  - Calculated Dataset Dir: {self.dataset_dir}")

        if not os.path.exists(self.processed_dir):
            raise FileNotFoundError(f"Processed directory does not exist: {self.processed_dir}")
        
        
        # --- 0. Parse Modalities ---
        self.modalities = self.parse_modalities(modalities)
        self.do_mixup = (args.do_mixup or args.do_mixup_only_treatment) and len(self.modalities) > 1 and self.mode == "train"
        print(f"Active modalities: {self.modalities}")

        # --- 2. Load Patient Split ---
        split_file = os.path.join(self.processed_dir, "luad_patients_5fold.json") 
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

        # Load CSVs (Tabular Data & Labels)
        self._load_tabular_and_labels()

    def _load_tabular_and_labels(self):
        try:
            self.clinical_df = pd.read_csv(os.path.join(self.processed_dir, "clinical_data_aggregated.csv"), dtype=str)
            self.treatment_df = pd.read_csv(os.path.join(self.processed_dir, "treatment_data_aggregated.csv"), dtype=str)
            self.pathology_df = pd.read_csv(os.path.join(self.processed_dir, "pathology_aggregated.csv"), dtype=str)
            self.labels_df = pd.read_csv(os.path.join(self.processed_dir, "luad_patient_labels.csv"), dtype=str)
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
        if "tabular-treatment-9" in self.modalities:
            cols = [c for c in self.treatment_df.columns if c not in exclude_cols]
            for _, row in self.treatment_df.iterrows():
                pid = row['cases.submitter_id']
                self.treatment_tabular_dict[pid] = _process_row(row, cols)

        self.pathology_tabular_dict = {}
        if "tabular-pathology-21" in self.modalities:
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
        do_mixup = requested_idx >= len(self) and self.do_mixup   # if idx >= len(self), then we're doing mixup augmentation
        # print("Request IDX", requested_idx, "Do Mixup:", do_mixup, "Dataset length = ", len(self))
        idx = requested_idx % len(self) 

        patient_id = self.patient_ids[idx]
        output_dict = {"pid": patient_id}

        # --- 1. Labels ---
        try:
            survival_info = self.patient_labels[patient_id]
            event = int(survival_info['DFS_event'])
            time_days = float(survival_info['DFS_time'])
        except KeyError:
            # Skip patient if labels are missing
            # print(f"Warning: Labels missing for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        output_dict['labels'] = {
            'do_mixup': do_mixup,  
            'label_time': time_days,
            'label_event': event,
        }

        # --- 2. Load Modalities ---
        modalities_found = 0

        for mod in self.modalities:
            feature_data = None

            # Case A: Tabular
            if mod == "tabular-clinical-9":
                feature_data = self.clinical_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            elif mod == "tabular-treatment-9":
                feature_data = self.treatment_tabular_dict.get(patient_id)
                if feature_data is not None:
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

            elif mod == "tabular-pathology-21":
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
                        modalities_found += 1
                else:
                    modalities_found += 1
        
        # --- 3. Integrity Check ---
        if modalities_found == 0:
            # print(f"Warning: No valid modalities found for {patient_id}, skipping...")
            return self.get_sample((idx + 1) % len(self))

        if do_mixup:
            other_item_idx = random.randint(0, len(self) - 1)
            if other_item_idx == idx: 
                other_item_idx = (idx + 1) % len(self)
            output_dict = self.mixup_data(output_dict, self.get_sample(other_item_idx))

        return output_dict

    def mixup_data(self, ori_data, other_data):
        mixup_modalities = set()

        # Handle edge case where only 1 modality exists (randint(1, 0) would fail)
        max_k = max(1, len(self.modalities) - 1)
        k = random.randint(1, max_k)
        mixup_modalities.update(random.sample(self.modalities, k=k))  # Select k modalities to swap from other_data -> ori_data

        token_num_1 = sum([v.shape[0] for k, v in ori_data.items() if k != 'labels' and k != 'pid' and v is not None])
        token_num_2 = 0

        for mod in mixup_modalities:
            token_num_1 -= ori_data[mod].shape[0] if isinstance(ori_data[mod], torch.Tensor) else 0
            if other_data.get(mod) is not None:
                ori_data[mod] = other_data[mod].clone() if isinstance(other_data[mod], torch.Tensor) else other_data[mod]
                token_num_2 += other_data[mod].shape[0]
            else:
                ori_data[mod] = None

        t1 = ori_data['labels']['label_time']
        e1 = ori_data['labels']['label_event']
        t2 = other_data['labels']['label_time']
        e2 = other_data['labels']['label_event']

        if e1 == e2:
            label_event = e1
            label_time = t2 if t2 < t1 else t1
        else:
            ratio = token_num_1 / (token_num_1 + token_num_2)
            label_event = ratio * e1 + (1 - ratio) * e2
            label_time = ratio * t1 + (1 - ratio) * t2
   
        ori_data['labels'] = {
            "do_mixup": True,
            "label_event": label_event,
            "label_time": label_time,
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
    