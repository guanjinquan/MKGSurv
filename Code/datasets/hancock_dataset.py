import os
import json
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import copy


class HANCOCKDataset(Dataset):

    def __init__(self, mode: str = "train", modalities: str = "all"):
        """
        Initializes the dataset.

        Args:
            mode (str): The dataset mode, one of 'train', 'valid', or 'test'.
            modalities (str): A comma-separated string of modalities to load,
                              e.g., "image,strong_related_text". "all" loads all available.
                              Supported: 'image', 'strong_related_text', 'weak_related_text'.
        """
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"

        random.seed(42)
        self.mode = mode
        self.dataset_dir = os.path.join(os.getcwd(), "../Data/HANCOCK")
        self.wsi_encodings_dir = os.path.join(self.dataset_dir, "WSI_UNI_encodings")
        self.text_data_dir = os.path.join(self.dataset_dir, "TextData")
        self.structured_data_dir = os.path.join(self.dataset_dir, "StructuredData")

        # --- Parse modalities ---
        self.modalities = self._parse_modalities(modalities)
        print(f"Dataset will be initialized for modalities: {self.modalities}")

        # --- Dataframes and Dictionaries to hold the data ---
        self.clinical_df_str = None
        self.clinical_df_encoded = None
        self.pathological_df_str = None
        self.pathological_df_encoded = None
        self.blood_data_map_str = {}
        self.blood_data_map_encoded = {}
        self.blood_ref_df = None
        self.pat_to_wsi_embeddings = {}
        self.patient_ids = []
        self.target_patids = []

        # --- Columns to exclude from features (labels and identifiers) ---
        self.label_leakage_columns = [
            "patient_id", "survival_status", "survival_status_with_cause",
            "days_to_last_information", "recurrence", "days_to_recurrence",
            "progress_1", "days_to_progress_1", "progress_2", "days_to_progress_2",
            "metastasis_1_locations", "days_to_metastasis_1",
            "metastasis_2_locations", "days_to_metastasis_2",
            "metastasis_3_locations", "days_to_metastasis_3",
            "metastasis_4_locations", "days_to_metastasis_4"
        ]

        # --- Load and preprocess all data sources ---
        self._load_data()
        print(f"Dataset for mode '{self.mode}' initialized. Found {len(self.patient_ids)} patients.")

    def _get_survival_bins(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        """
        self.observed_years = 20 * 365.0
        self.num_time_bins = 20
        self.time_bins = np.linspace(0, self.observed_years, self.num_time_bins + 1)

        labels_y = []
        for patient_id in self.patient_ids:
            label_info = self.clinical_df_encoded.loc[patient_id]
            
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
            
            labels_y.append(int(time_bin))
            
        return labels_y

    def __len__(self) -> int:
        """Returns the total number of patients in the dataset."""
        return len(self.patient_ids)

    def _parse_modalities(self, modalities_str: str) -> List[str]:
        """Parses the comma-separated modalities string into a list."""
        if modalities_str == "all":
            return [
                "image-pathology", 
                "text-clinical",
                "tabular-pathology-17",
                "tabular-clinical-52"
            ]

        valid_modalities = {
            "image-pathology", 
            "text-clinical",
            "tabular-pathology-17",
            "tabular-clinical-52"
        }
        modalities = [m.strip() for m in modalities_str.split(',')]

        for m in modalities:
            if m not in valid_modalities:
                raise ValueError(f"Unknown modality '{m}'. Valid options are {valid_modalities}")

        return modalities

    def _load_json_to_df(self, file_path: str, target_pids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Helper to load a JSON file and filter it for target patient IDs.
        
        Returns:
            Tuple containing:
            - df_str: DataFrame with all values converted to strings
            - df_encoded: DataFrame with all values converted to numeric types (int/float)
        """
        try:
            index_col = "patient_id"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filtered_data = [item for item in data if item.get(index_col) in target_pids]
            if not filtered_data:
                return pd.DataFrame(), pd.DataFrame()

            # Create original DataFrame
            df_original = pd.DataFrame(filtered_data)
            if index_col in df_original.columns:
                df_original.set_index(index_col, inplace=True)

            # Create string DataFrame - convert all values to strings
            df_str = df_original.astype(str)
            
            # Create encoded DataFrame - convert all values to numeric types
            df_encoded = df_original.copy()
            
            # --- FIX 1: 定义要豁免编码的列 ---
            not_encode_columns = ["analyte_name"]
            
            for col in df_encoded.columns:
                # Try to convert to numeric first
                try:
                    # Attempt direct numeric conversion
                    df_encoded[col] = pd.to_numeric(df_encoded[col])
                except:
                    pass
                
                # --- FIX 2: 如果列在豁免列表中，则跳过后续的标签编码 ---
                if any(no_encode_col in col for no_encode_col in not_encode_columns):
                    continue
                    
                # Handle specific data types
                if df_encoded[col].dtype == 'object':
                    try:
                        # Try to convert to numeric first
                        converted = pd.to_numeric(df_encoded[col])
                        if not converted.isna().all():  # If successful conversion
                            df_encoded[col] = converted
                        else:
                            # Use label encoding for categorical strings
                            unique_values = df_encoded[col].dropna().unique()
                            if len(unique_values) > 0:
                                unique_values = sorted(unique_values)
                                mapping = {value: idx for idx, value in enumerate(unique_values)}
                                df_encoded[col] = df_encoded[col].map(mapping)
                    except:
                        # Fallback: use label encoding
                        unique_values = df_encoded[col].dropna().unique()
                        if len(unique_values) > 0:
                            unique_values = sorted(unique_values)
                            mapping = {value: idx for idx, value in enumerate(unique_values)}
                            df_encoded[col] = df_encoded[col].map(mapping)
                
                elif df_encoded[col].dtype == 'bool':
                    df_encoded[col] = df_encoded[col].astype(int)
                
                elif 'datetime' in str(df_encoded[col].dtype):
                    df_encoded[col] = pd.to_datetime(df_encoded[col]).astype('int64') // 10**9  # Convert to Unix timestamp
                
                elif 'timedelta' in str(df_encoded[col].dtype):
                    df_encoded[col] = df_encoded[col].dt.total_seconds()

            # --- FIX 3: 移除多余的 "Final pass" 循环 ---
            # (原先 77-87 行的代码已被删除，因为上面的循环已正确处理)
            
            # Convert all columns to numeric, filling NaN with 0 for any remaining non-numeric values
            for col in df_encoded.columns:
                # 这部分逻辑是正确的
                if all(no_encode_col not in col for no_encode_col in not_encode_columns):
                    df_encoded[col] = pd.to_numeric(df_encoded[col]).fillna(-1)

            return df_str, df_encoded

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Data file not found at {file_path}")

        except Exception as e:
            raise RuntimeError(f"Error loading or processing {file_path}: {e}")
        

    def _load_data(self):
        """Loads all data sources based on the requested modalities and data split."""
        # 1. Load data splits to identify target patients for train/valid/test
        split_file = os.path.join(self.dataset_dir, "DataSplits_DataDictionaries", "dataset_split_train_valid_test.json")
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        self.target_patids = [item['patient_id'] for item in split_data if item['dataset'] == self.mode]
        self.patient_ids = self.target_patids

        # Always load clinical data for labels and basic info
        self.clinical_df_str, self.clinical_df_encoded = self._load_json_to_df(
            os.path.join(self.structured_data_dir, "clinical_data.json"), self.target_patids
        )
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

        # 2. Conditionally load data for text modalities
        if any("text" in modal for modal in self.modalities) or any("tabular" in modal for modal in self.modalities):
            # Load Pathological Data
            self.pathological_df_str, self.pathological_df_encoded = self._load_json_to_df(
                os.path.join(self.structured_data_dir, "pathological_data.json"), self.target_patids
            )
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

            # Load Blood Data
            self.blood_ref_df = pd.read_json(os.path.join(self.structured_data_dir, "blood_data_reference_ranges.json"))
            # 找到所有包含 _min 或 _max 的列
            ref_cols = [col for col in self.blood_ref_df.columns if '_min' in col or '_max' in col]
            for col in ref_cols:
                # 强制将它们转换为数值类型，无法转换的变为 NaN
                self.blood_ref_df[col] = pd.to_numeric(self.blood_ref_df[col])


            self.blood_df_str, self.blood_df_encoded = self._load_json_to_df(
                os.path.join(self.structured_data_dir, "blood_data.json"), self.target_patids
            )

            assert self.blood_df_str.empty is False and self.blood_df_encoded.empty is False, "Blood data is empty"

            for patient_id, group in self.blood_df_str.reset_index().groupby('patient_id'):
                self.blood_data_map_str[patient_id] = group.to_dict('records')
            for patient_id, group in self.blood_df_encoded.reset_index().groupby('patient_id'):
                self.blood_data_map_encoded[patient_id] = group.to_dict('records')
            # analyte_name
            # self.blood_tabular_columns = sorted(list(set(self.blood_df_encoded["analyte_name"].dropna().unique())))
            self.blood_tabular_columns = ['Basophils', 'Basophils %', 'CRP', 'Calcium', 'Chloride', 'Creatinine', 'Eosinophils', 'Eosinophils %', 'Erythrocytes', 'Glomerular filtration rate', 'Glucose', 'Granulocytes', 'Granulocytes %', 'Hematocrit', 'Hemoglobin', 'INR', 'Immature Granulocytyes', 'Leukocytes', 'Lymphocytes', 'Lymphocytes %', 'MCH', 'MCV', 'MHCH', 'MPV', 'Magnesium', 'Monocytes', 'Monocytes %', 'Normoblasts', 'PDW', 'PLCR', 'PT', 'Platelets', 'Potassium', 'RDW', 'Sodium', 'Thrombin time', 'Urea', 'aPPT']
            # print("Blood Tabular Columns: ", self.blood_tabular_columns)
            # print("Blood Tabular Columns Number: ", len(self.blood_tabular_columns))  # 38


        # 3. Conditionally load WSI embeddings for the image modality
        if "image-pathology" in self.modalities:
            if os.path.exists(self.wsi_encodings_dir):
                for root, _, file in os.walk(self.wsi_encodings_dir):
                    for fname in file:
                        if fname.endswith('.h5'):
                            # Assuming a file naming convention like '..._{pid}.h5'
                            pid = fname.split('_')[-1].replace('.h5', '')
                            if pid in self.target_patids:
                                file_path = os.path.join(root, fname)
                                try:
                                    with h5py.File(file_path, 'r') as f:
                                        if 'features' in f:
                                            features = f['features'][:]
                                            if pid not in self.pat_to_wsi_embeddings:
                                                self.pat_to_wsi_embeddings[pid] = []
                                            self.pat_to_wsi_embeddings[pid].append(torch.from_numpy(features))
                                except Exception as e:
                                    print(f"Warning: Could not load WSI embedding for patient {pid} from {file_path}. Error: {e}")
            else:
                print(f"Warning: WSI encodings directory not found at {self.wsi_encodings_dir}")

    def _read_text_file(self, subdir: str, file_pattern: str) -> str:
        """Safely reads a text file for a given patient."""
        file_path = os.path.join(self.text_data_dir, subdir, file_pattern)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError as e:
            # print(f"Warning: Text file not found for patient: {subdir} {file_pattern}")
            return ""

    def _generate_blood_summary(self, patient_id: str) -> str:
        """Generates a natural language summary of the patient's blood data."""
        if patient_id not in self.blood_data_map_str or self.clinical_df_str is None or self.blood_ref_df is None:
            return "No blood data available."

        patient_sex = self.clinical_df_str.loc[patient_id].get('sex', 'unknown')
        patient_blood_records = self.blood_data_map_str[patient_id]
        patient_blood_records_encoded = self.blood_data_map_encoded[patient_id]

        summary_lines = []
        for record, record_encoded in zip(patient_blood_records, patient_blood_records_encoded):
            analyte, value_str, unit = record.get('analyte_name'), record.get('value'), record.get('unit', '')
            analyte, value_float = record_encoded.get('analyte_name'), record_encoded.get('value')
            if analyte is None or value_float is None or value_str is None: continue

            ref_range_row = self.blood_ref_df[self.blood_ref_df['analyte_name'] == analyte]
            line = f"- {analyte}: {value_str} {unit if unit else ''}"
            if not ref_range_row.empty:
                ref = ref_range_row.iloc[0]
                min_val, max_val = ref.get(f'normal_{patient_sex}_min', None), ref.get(f'normal_{patient_sex}_max', None)
                if min_val and max_val and pd.notna(min_val) and pd.notna(max_val):
                    status = "normal"
                    if value_float < min_val: status = "low"
                    elif value_float > max_val: status = "high"
                    line += f" (Status: {status}, Reference: {min_val}-{max_val} {ref.get('unit', '')})"
            summary_lines.append(line)

        return "\n".join(summary_lines) if summary_lines else ""
    
    def _generate_pathology_tabular_data(self, patient_id):
        # Get pathological features in dict format
        pathological_features = self.pathological_df_encoded .loc[patient_id].to_dict() \
            if self.pathological_df_encoded  is not None and patient_id in self.pathological_df_encoded .index else {}
        
        # dict to tabular sorted the key
        self.pathology_tabular_columns = sorted(list(set(self.pathology_tabular_columns)))
        tabular_pathology_data = [pathological_features.get(column, -1) for column in self.pathology_tabular_columns]

        assert len(tabular_pathology_data) == len(self.pathology_tabular_columns), "Pathology Tabular Columns Number is not equal to {}".format(len(self.pathology_tabular_columns))
        assert len(tabular_pathology_data) == 17, "Pathology Tabular Columns Number is not 17, but got {}".format(len(tabular_pathology_data))
        return tabular_pathology_data
    
    def _generate_clinical_tabular_data(self, patient_id):
        # return tabular data in order blood_tabular_columns
        CLINICAL_TABULAR_LENGTH = 52
        if patient_id not in self.blood_data_map_encoded or self.clinical_df_encoded is None:
            return None
        
        patient_blood_records = self.blood_data_map_encoded[patient_id]
        # print("patient_blood_records = ", patient_blood_records)
        tabular_data = []

        # blood tabular  length = 38
        self.blood_tabular_columns = sorted(list(set(self.blood_tabular_columns)))
        for column in self.blood_tabular_columns:
            value = next(record['analyte_name'] == column for record in patient_blood_records)
            if value is None:
                tabular_data.append(-1)  # padding nan with -1
            else:
                tabular_data.append(value)

        # clinical tabular  length = 14
        self.clinical_tabular_columns = sorted(list(set(self.clinical_tabular_columns)))
        for column_name in self.clinical_tabular_columns:
            value = self.clinical_df_encoded.loc[patient_id, column_name]
            if pd.isna(value):
                tabular_data.append(-1)  # padding nan with -1
            else:
                tabular_data.append(value)

        assert len(tabular_data) == len(self.clinical_tabular_columns) + len(self.blood_tabular_columns), "Clinical Tabular Columns Number is not equal to {}".format(len(self.clinical_tabular_columns) + len(self.blood_tabular_columns))
        assert len(tabular_data) == CLINICAL_TABULAR_LENGTH, f"Expected 52 tabular data columns, got {len(tabular_data)}"
        return tabular_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single patient's data, including survival labels `label_Y` and `label_c`.
        """
        patient_id = self.patient_ids[index]
        output_dict = {"pid": patient_id}

        # --- Image Modality ---
        if "image-pathology" in self.modalities:
            wsi_embeddings = self.pat_to_wsi_embeddings.get(patient_id)
            if wsi_embeddings:
                wsi_tensor = torch.cat(wsi_embeddings, dim=0) if len(wsi_embeddings) > 1 else wsi_embeddings[0]

                # --- DATA AUGMENTATION ---
                # For the training set, with 50% probability, drop ~x% of image tokens.
                if self.mode == 'train' and torch.rand(1) < 0.5:
                    num_tokens = wsi_tensor.shape[0]
                    if num_tokens > 1:
                        # Number of tokens to drop: random between 10% to 50%
                        drop_ratio = random.uniform(0.1, 0.5)
                        num_to_keep = int(num_tokens * drop_ratio)
                        # Create a random indices to keep (in sorted order)
                        keep_indices = torch.randperm(num_tokens)[:num_to_keep].sort().values
                        # Update the tensor
                        wsi_tensor = wsi_tensor[keep_indices]

                output_dict["image-pathology"] = wsi_tensor
            else:
                output_dict["image-pathology"] = None

        # --- Text Modalities ---
        if "text-clinical" in self.modalities:
            radiology_report = self._read_text_file('reports_english', f'SurgeryReport_{patient_id}.txt')
            clinical_features = self.clinical_df_str.loc[patient_id].to_dict()
            for col_leak in self.label_leakage_columns:
                if col_leak in clinical_features:
                    clinical_features.pop(col_leak)
            clinical_summary = " ".join([f"{k}: {v}" for k, v in clinical_features.items() if pd.notna(v)])

            text_dict = {
                "Surgery and Radiology Report": radiology_report,
                "Clinical Summary": clinical_summary,
                "Surgery Description": self._read_text_file('surgery_descriptions_english', f'SurgeryDescriptionEnglish_{patient_id}.txt'),
                "Patient History": self._read_text_file('histories_english', f'SurgeryReport_History_{patient_id}.txt'),
                "OPS Codes": self._read_text_file('ops_codes', f'SurgeryReports_OPS_Codes_{patient_id}.txt'),
                "ICD Codes": self._read_text_file('icd_codes', f'SurgeryReport_ICD_Codes_{patient_id}.txt'),
                "Blood Work Summary": self._generate_blood_summary(patient_id)
            }

            # random shuffle the sections to introduce variability
            text_sents = [f"{key}: {value}" for key, value in text_dict.items() if value]
            random.shuffle(text_sents)

            full_weak_text = "\n".join(text_sents)
            output_dict["text-clinical"] = full_weak_text.strip()

        # --- Tabular Modalities ---
        if "tabular-clinical-52" in self.modalities:
            tabular_data = self._generate_clinical_tabular_data(patient_id)
            output_dict["tabular-clinical-52"] = tabular_data  # maybe None when no data
        
        if "tabular-pathology-17" in self.modalities:
            tabular_data = self._generate_pathology_tabular_data(patient_id)
            output_dict["tabular-pathology-17"] = tabular_data  # maybe None when no data

        # --- Survival Labels (Y and c) ---
        label_info = self.clinical_df_str.loc[patient_id]
        
        has_recurrence = label_info.get('recurrence') == 'yes'
        time_to_recurrence = float(label_info.get('days_to_recurrence'))
        time_to_last_info = float(label_info.get('days_to_last_information'))

        event_time = -1.0
        event_flag = 0   # event_flag = 0 means censored, event_flag = 1 means event occurred.

        if has_recurrence and pd.notna(time_to_recurrence):
            # The patient had a recurrence event.
            event_flag = 0
            event_time = time_to_recurrence
        elif pd.notna(time_to_last_info):
            # No recurrence event, use last follow-up time. Always censored.
            event_flag = 0
            event_time = time_to_last_info  # We observe them until their last follow-up or the end of the study, whichever comes first.
        
        output_dict['labels'] = {
            'label_time': event_time,
            'label_event': event_flag
        }

        # --- Data Integrity Check ---
        # If no requested modalities were found for this patient, get the next one.
        modalities_found = sum(1 for m in self.modalities if output_dict.get(m) is not None and (isinstance(output_dict.get(m), str) and output_dict.get(m).strip() != "" or not isinstance(output_dict.get(m), str)))
        if modalities_found == 0:
             return self.__getitem__((index + 1) % len(self))

        return output_dict



if __name__ == "__main__":
    print("HANCOCK Dataset")
    import os
    os.chdir("/home/Guanjq/NewWork/MedAlignFusion/Code")
    dataset = HANCOCKDataset(mode="train", modalities="all")
    print(len(dataset))
    print(dataset[0])