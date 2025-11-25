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
    TREATMENT_OPTIONS = None
    TREATMENT_OPTIONS_ONEHOT = None
    TREATMENT_OPTIONS_FEAT = None

    PRE_OP_MODALITIES = [
        "image-pathology",           # pre-treatment (Feature from PKL)
        "text-clinical",             # pre-treatment (Feature from PKL)
        "tabular-metadata-4",        # pre-treatment (From CSV)
        "tabular-history-9",         # pre-treatment (From CSV)
        "tabular-blood-5",           # pre-treatment (From CSV)
    ]

    POST_OP_MODALITIES = [
        "text-pathology",            # with-treat (Feature from PKL)
        "text-treatment",            # with-treat (Feature from PKL)
        "tabular-pathology-16",      # with-treat (From CSV)
        "tabular-immunohistochemic-5", # with-treat (From CSV)
        "tabular-posop-blood-4",     # with-treat (From CSV)
    ]

    VALID_MODALITIES = PRE_OP_MODALITIES + POST_OP_MODALITIES

    def _read_pickle(self, path: str) -> Any:
        """Helper to load pickle/joblib files safely."""
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
        
        self.args = args
        self.mode = mode
        
        # --- Path Definitions ---
        self.dataset_dir = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"
        self.processed_dir = os.path.join(self.dataset_dir, "processed")
        
        # --- 1. Parse Modalities ---
        self.modalities = self.parse_modalities(modalities)
        self.do_mixup = (getattr(args, 'do_mixup', False) or getattr(args, 'do_mixup_only_treatment', False)) and len(self.modalities) > 1 and self.mode == "train"
        print(f"Dataset initialized for mode='{self.mode}'. Active modalities: {self.modalities}")

        # --- 2. Load Clinical Data (CSV) ---
        self._load_clinical_csv()

        # --- 3. Load Patient Split ---
        self._load_split_file()

        # --- 4. Load Pre-extracted Features (Pickles) ---
        self.loaded_features = {}
        
        # Map modality names to your specific file paths
        self.pickle_map = {
            "image-pathology": "feature_image_pathology.pkl",
            "text-clinical": "features_text_clinical.pkl",
            "text-pathology": "features_text_pathology.pkl",
            "text-treatment": "features_text_treatment.pkl"
        }

        # Load Treatment Options
        self._load_treatment_options()

        # Load Feature Files
        for mod in self.modalities:
            if mod in self.pickle_map:
                pkl_path = os.path.join(self.processed_dir, self.pickle_map[mod])
                data = self._read_pickle(pkl_path)
                if data is not None:
                    self.loaded_features[mod] = data
        
        # --- 5. Pre-process Tabular Data ---
        # Since tabular data comes from CSV, we pre-process it into dicts here for speed
        self.tabular_features = {} 
        self._preprocess_tabular_data()

        print(f"Dataset loaded: mode='{self.mode}'. Final valid patient count: {len(self.items)}")

    def _load_clinical_csv(self):
        csv_path = os.path.join(self.dataset_dir, "clinical_data.csv")
        try:
            self.clinical_df = pd.read_csv(csv_path)
            # Ensure PID is treated as string for consistency with JSON keys
            if 'PID' in self.clinical_df.columns:
                self.clinical_df['PID'] = self.clinical_df['PID'].astype(str)
                self.clinical_df.set_index('PID', inplace=True)
            print("Successfully loaded clinical data CSV.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Clinical data file not found at {csv_path}")

    def _load_split_file(self):
        # Load Metadata to get label info
        metadata_path = os.path.join(self.dataset_dir, "oscc_recurrence_survival_data.json")
        split_path = os.path.join(self.dataset_dir, "split_OOD.json")
        
        with open(metadata_path, 'r') as f:
            all_patients_info = {str(item['pid']): item for item in json.load(f)}

        with open(split_path, 'r') as f:
            split_data = json.load(f)

        target_pids = [str(pid) for pid in split_data[self.mode]]
        
        # Filter items
        self.items = []
        for pid in target_pids:
            if pid in all_patients_info and pid in self.clinical_df.index:
                self.items.append(all_patients_info[pid])
        
        # Shuffle for train
        if self.mode == "train":
            random.shuffle(self.items)

    def _load_treatment_options(self):
        clinical_path = os.path.join(self.dataset_dir, "clinical_data.csv")
        trt_opt_path = os.path.join(self.processed_dir, "features_all_treatment_options.pkl")
        clinical_df = pd.read_csv(clinical_path)
        data = self._read_pickle(trt_opt_path)

        # Load Features 
        options_to_feat = {}
        for option, feat in zip(data.get("ALL_TREATMENT_OPTIONS_STR", []), data.get("ALL_TREATMENT_OPTIONS_FEAT", [])):
            options_to_feat[option] = feat

        # Set options and onehot encoding
        options_to_onehot = {}
        for idx, row in clinical_df.iterrows():
            patient_option = row['12_treatment_type']
            if patient_option in options_to_onehot:
                continue
            options_to_onehot[patient_option] = np.zeros((12, ))
            patient_onehot = row['12_treatment_type_id']
            for id in patient_onehot.split(","):
                options_to_onehot[patient_option][int(id)] = 1
        
        self.TREATMENT_OPTIONS = list(options_to_onehot.keys())
        self.TREATMENT_OPTIONS_ONEHOT = [options_to_onehot[option] for option in self.TREATMENT_OPTIONS]
        self.TREATMENT_OPTIONS_FEAT = [options_to_feat[option] for option in self.TREATMENT_OPTIONS]
        assert len(self.TREATMENT_OPTIONS) == len(self.TREATMENT_OPTIONS_FEAT), f"Mismatch between treatment options and features, {len(self.TREATMENT_OPTIONS)} vs {len(self.TREATMENT_OPTIONS_FEAT)}"
        assert len(self.TREATMENT_OPTIONS) == len(self.TREATMENT_OPTIONS_ONEHOT), f"Mismatch between treatment options and onehot, {len(self.TREATMENT_OPTIONS)} vs {len(self.TRWEIGHT_OPTIONS_ONEHOT)}"
        assert len(self.TREATMENT_OPTIONS) > 0, "No treatment options found. Please check your data."


    def _preprocess_tabular_data(self):
        """
        Pre-extracts tabular vectors for all patients based on the requested modalities.
        Ensures output shape is (1, D) for consistency (n_i, d_i).
        """
        # Define the column mappings
        sources_with_columns = {
            "tabular-metadata-4": ["Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)"],
            "tabular-history-9": [
                "AlcoholHistory(0no/1yes)", "SmokingHistory(0no/1yes)", "BetelNutHistory(0no/1yes)", 
                "PreoperativeHistory(0no/1yes)", "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
                "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)", "NeckMass(+)"
            ],
            "tabular-blood-5": ["PreopWBC", "PreopHemoglobin", "PreopPotassium", "PreopAlbumin", "PreopVitaminD"],
            "tabular-posop-blood-4": ["PostopWBC", "PostopHemoglobin", "PostopPotassium", "PostopAlbumin"],
            "tabular-pathology-16": [
                "TumorT", "TumorN", "TumorM", "TumorDifferentiation(1high/2med/3low)",
                "CancerThrombus(0/1)", "SurroundingTissueInvasion(0/1)",  "LNM(0/1)", 
                "AccessoryChain(+)", "VascularInvasion(+)", "PerineuralInvasion(+)", 
                "Metastasis(0no/1yes)", "IA(+)", "IB(+)", "IIA(+)", "IIB(+)", "III(+)"
            ],
            "tabular-immunohistochemic-5": ["Ki-67", "CK5_6(0/1)", "P63(0/1)", "P16(0/1)", "HPV(0/1)"]
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
                        if pd.isna(f_val):
                            vec.append(-1.0)
                        else:
                            vec.append(f_val)
                    except:
                        vec.append(-1.0)
                
                # IMPORTANT: Convert to tensor and ensure shape is (1, D)
                # n_i = 1, d_i = number of columns
                tensor = torch.tensor(vec, dtype=torch.float32)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                
                self.tabular_features[mod_name][str(pid)] = tensor

    def parse_modalities(self, modalities_str: str) -> List[str]:
        if modalities_str == "all":
            return sorted(list(self.VALID_MODALITIES))
        
        parsed = [m.strip() for m in modalities_str.split(',')]
        valid = []
        for m in parsed:
            if m in self.VALID_MODALITIES:
                valid.append(m)
            else:
                print(f"Warning: Modality '{m}' not recognized/supported.")
        return valid

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, requested_idx: int) -> Dict[str, Any]:
        do_mixup = requested_idx >= len(self) and self.do_mixup
        idx = requested_idx % len(self)
        
        item_info = self.items[idx]
        pid_str = str(item_info['pid'])
        
        # --- 1. Survival Labels ---
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

        try:
            patient_series = self.clinical_df.loc[pid_str]
            treat_type_str = patient_series['12_treatment_type']
            treat_type_id = str(patient_series['12_treatment_type_id'])
        except KeyError:
            return self.get_sample((idx + 1) % len(self))

        output_dict = {
            "pid": pid_str,
            "labels": {
                'do_mixup': do_mixup,
                'label_time': event_time,
                'label_event': event_flag,
                "treatment_type": treat_type_str,
                "treatment_type_onehot": torch.tensor([1 if str(i) in treat_type_id else 0 for i in range(12)], dtype=torch.float32),
            }
        }

        # --- 2. Load Features ---
        modalities_found = 0

        for mod in self.modalities:
            feature_data = None
            
            # Case A: Feature from Pickle (Image / Text)
            if mod in self.loaded_features:
                data_dict = self.loaded_features[mod]
                if pid_str in data_dict:
                    raw_feat = data_dict[pid_str]
                    
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

            # Case B: Tabular from CSV
            elif mod in self.tabular_features:
                if pid_str in self.tabular_features[mod]:
                    feature_data = self.tabular_features[mod][pid_str]

            # --- Final Shape Enforcement (n_i, d_i) ---
            if isinstance(feature_data, torch.Tensor):
                if feature_data.dim() == 1:
                    feature_data = feature_data.unsqueeze(0) # (D,) -> (1, D)
            
            output_dict[mod] = feature_data
            
            if feature_data is not None:
                modalities_found += 1

        # --- 3. Integrity Check ---
        if modalities_found == 0:
            return self.get_sample((idx + 1) % len(self))

        # --- 4. Mixup Application ---
        if do_mixup:
            other_item_idx = random.randint(0, len(self) - 1)
            if other_item_idx == idx: 
                other_item_idx = (idx + 1) % len(self)
            output_dict = self.mixup_data(output_dict, self.get_sample(other_item_idx))

        return output_dict

    def mixup_data(self, ori_data, other_data):
        mixup_modalities = set()

        if not self.args.do_mixup_only_treatment:
            # Handle edge case where only 1 modality exists (randint(1, 0) would fail)
            max_k = max(1, min(len(self.modalities) // 2, len(self.modalities) - 1))
            k = random.randint(1, max_k)
            # Select k modalities to swap from other_data -> ori_data
            mixup_modalities.update(random.sample(self.modalities, k=k))
        elif "text-treatment" in self.modalities:
            mixup_modalities.add("text-treatment")

        for mod in mixup_modalities:
            if other_data.get(mod) is not None:
                ori_data[mod] = other_data[mod].clone() if isinstance(other_data[mod], torch.Tensor) else other_data[mod]
            else:
                ori_data[mod] = None

        t1 = ori_data['labels']['label_time']
        e1 = ori_data['labels']['label_event']
        t2 = other_data['labels']['label_time']
        e2 = other_data['labels']['label_event']

        use_other_labels = False
        if e1 == 1 and e2 == 0: 
            use_other_labels = False
        elif e1 == 0 and e2 == 1: 
            use_other_labels = True
        elif e1 == e2:
            use_other_labels = t2 < t1
        else:
            raise ValueError("Invalid event combination")
        
        if use_other_labels:
            ori_data['labels'] = copy.deepcopy(other_data['labels'])
        
        ori_data['labels']['do_mixup'] = True
        return ori_data

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

