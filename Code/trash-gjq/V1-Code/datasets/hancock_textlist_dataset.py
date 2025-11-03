import os
import json
from typing import Dict, Any, List

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random



class HANCOCK_TextList_Dataset(Dataset):

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
        self.clinical_df = None
        self.pathological_df = None
        self.blood_data_map = {}
        self.blood_ref_df = None
        self.pat_to_wsi_embeddings = {}
        self.patient_ids = []
        self.target_patids = []

        # --- Survival Analysis Parameters ---
        self.five_years_in_days = 5 * 365.0
        self.num_time_bins = 10
        self.time_bins = np.linspace(0, self.five_years_in_days, self.num_time_bins + 1)

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

    def _get_labels(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        """
        labels_y = []
        for patient_id in self.patient_ids:
            label_info = self.clinical_df.loc[patient_id]
            
            has_recurrence = label_info.get('recurrence') == 'yes'
            time_to_recurrence = label_info.get('days_to_recurrence')
            time_to_last_info = label_info.get('days_to_last_information')

            event_time = -1.0
            
            if has_recurrence and pd.notna(time_to_recurrence):
                if time_to_recurrence < self.five_years_in_days:
                    event_time = time_to_recurrence
                else:
                    event_time = self.five_years_in_days
            elif pd.notna(time_to_last_info):
                event_time = min(time_to_last_info, self.five_years_in_days)
            
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
            return ["images", "strong_related_text", "weak_related_text"]

        valid_modalities = {"images", "strong_related_text", "weak_related_text"}
        modalities = [m.strip() for m in modalities_str.split(',')]

        for m in modalities:
            if m not in valid_modalities:
                raise ValueError(f"Unknown modality '{m}'. Valid options are {valid_modalities}")

        return modalities

    def _load_json_to_df(self, file_path: str, target_pids: List[str]) -> pd.DataFrame:
        """Helper to load a JSON file and filter it for target patient IDs."""
        try:
            index_col = "patient_id"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filtered_data = [item for item in data if item.get(index_col) in target_pids]
            if not filtered_data:
                return pd.DataFrame()

            df = pd.DataFrame(filtered_data)
            if index_col in df.columns:
                df.set_index(index_col, inplace=True)
            return df

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
        self.clinical_df = self._load_json_to_df(
            os.path.join(self.structured_data_dir, "clinical_data.json"), self.target_patids
        )

        # 2. Conditionally load data for text modalities
        if "strong_related_text" in self.modalities or "weak_related_text" in self.modalities:
            self.pathological_df = self._load_json_to_df(
                os.path.join(self.structured_data_dir, "pathological_data.json"), self.target_patids
            )
            self.blood_ref_df = pd.read_json(os.path.join(self.structured_data_dir, "blood_data_reference_ranges.json"))
            blood_df = self._load_json_to_df(
                 os.path.join(self.structured_data_dir, "blood_data.json"), self.target_patids
            )
            if not blood_df.empty:
                for patient_id, group in blood_df.reset_index().groupby('patient_id'):
                    self.blood_data_map[patient_id] = group.to_dict('records')

        # 3. Conditionally load WSI embeddings for the image modality
        if "images" in self.modalities:
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
        if patient_id not in self.blood_data_map or self.clinical_df is None or self.blood_ref_df is None:
            return "No blood data available."

        patient_sex = self.clinical_df.loc[patient_id].get('sex', 'unknown')
        patient_blood_records = self.blood_data_map[patient_id]

        summary_lines = []
        for record in patient_blood_records:
            analyte, value, unit = record.get('analyte_name'), record.get('value'), record.get('unit', '')
            if analyte is None or value is None: continue

            ref_range_row = self.blood_ref_df[self.blood_ref_df['analyte_name'] == analyte]
            line = f"- {analyte}: {value:.2f} {unit if unit else ''}"
            if not ref_range_row.empty:
                ref = ref_range_row.iloc[0]
                min_val, max_val = ref.get(f'normal_{patient_sex}_min'), ref.get(f'normal_{patient_sex}_max')
                if pd.notna(min_val) and pd.notna(max_val):
                    status = "normal"
                    if value < min_val: status = "low"
                    elif value > max_val: status = "high"
                    line += f" (Status: {status}, Reference: {min_val}-{max_val} {ref.get('unit', '')})"
            summary_lines.append(line)

        return "\n".join(summary_lines) if summary_lines else "No blood data available."

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single patient's data, including survival labels `label_Y` and `label_c`.
        """
        patient_id = self.patient_ids[index]
        output_dict = {"pid": patient_id}

        # --- Image Modality ---
        if "images" in self.modalities:
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

                output_dict["images"] = wsi_tensor
            else:
                output_dict["images"] = None

        # --- Text Modalities ---
        if "strong_related_text" in self.modalities or "weak_related_text" in self.modalities:
            pathological_features = self.pathological_df.loc[patient_id].to_dict() if self.pathological_df is not None and patient_id in self.pathological_df.index else {}
            pathology_summary = " ".join([f"{k}: {v}" for k, v in pathological_features.items() if pd.notna(v)])
            radiology_report = self._read_text_file('reports_english', f'SurgeryReport_{patient_id}.txt')

            if "strong_related_text" in self.modalities:
                strong_text = f"Pathology Summary: {pathology_summary}"
                output_dict["strong_related_text"] = strong_text.strip()

            if "weak_related_text" in self.modalities:
                clinical_features = self.clinical_df.loc[patient_id].drop(labels=self.label_leakage_columns, errors='ignore').to_dict()
                clinical_summary = " ".join([f"{k}: {v}" for k, v in clinical_features.items() if pd.notna(v)])
                weak_texts = {
                    "Surgery and Radiology Report": radiology_report,
                    "Clinical Summary": clinical_summary,
                    "Surgery Description": self._read_text_file('surgery_descriptions_english', f'SurgeryDescriptionEnglish_{patient_id}.txt'),
                    "Patient History": self._read_text_file('histories_english', f'SurgeryReport_History_{patient_id}.txt'),
                    "OPS Codes": self._read_text_file('ops_codes', f'SurgeryReports_OPS_Codes_{patient_id}.txt'),
                    "ICD Codes": self._read_text_file('icd_codes', f'SurgeryReport_ICD_Codes_{patient_id}.txt'),
                    "Blood Work Summary": self._generate_blood_summary(patient_id)
                }

                # random shuffle the sections to introduce variability
                text_sents = [f"{key}: {value}" for key, value in weak_texts.items() if value]
                random.shuffle(text_sents)

                # full_weak_text = "\n".join(text_sents)
                output_dict["weak_related_text"] = text_sents  # full_weak_text.strip()

        # --- Survival Labels (Y and c) ---
        label_info = self.clinical_df.loc[patient_id]
        
        has_recurrence = label_info.get('recurrence') == 'yes'
        time_to_recurrence = label_info.get('days_to_recurrence')
        time_to_last_info = label_info.get('days_to_last_information')

        event_time = -1.0
        # c = 1 means censored, c = 0 means event occurred.
        censorship = 1 

        if has_recurrence and pd.notna(time_to_recurrence):
            # The patient had a recurrence event.
            if time_to_recurrence < self.five_years_in_days:
                # Event occurred WITHIN the 5-year study period.
                censorship = 0
                event_time = time_to_recurrence
            else:
                # Event occurred AFTER 5 years. For a 5-year analysis, this is censored.
                censorship = 1
                event_time = self.five_years_in_days
        elif pd.notna(time_to_last_info):
            # No recurrence event, use last follow-up time. Always censored.
            censorship = 1
            # We observe them until their last follow-up or the end of the study, whichever comes first.
            event_time = min(time_to_last_info, self.five_years_in_days)
        
        # Discretize the event time into bins (0 to 9)
        # Y is the discrete time interval index
        time_bin = -1
        if event_time >= 0:
            # np.digitize finds which bin the value falls into.
            # We subtract 1 to get a 0-based index.
            time_bin = np.digitize(event_time, self.time_bins) - 1
            # Ensure the index is within the valid range [0, num_bins-1]
            time_bin = min(time_bin, self.num_time_bins - 1)

        # output_dict["label_Y"] = int(time_bin)
        # output_dict["label_c"] = censorship

        output_dict['labels'] = {
            'label_Y': int(time_bin),
            'label_c': censorship,
        }
        
        output_dict['original_labels'] = {
            'label_Y': event_time,
            'label_c': censorship
        }

        # --- Data Integrity Check ---
        # If no requested modalities were found for this patient, get the next one.
        modalities_found = sum(1 for m in self.modalities if output_dict.get(m) is not None and (isinstance(output_dict.get(m), str) and output_dict.get(m).strip() != "" or not isinstance(output_dict.get(m), str)))
        if modalities_found == 0:
             return self.__getitem__((index + 1) % len(self))

        return output_dict






# Example of how to use the dataset class
if __name__ == '__main__':
    # IMPORTANT: Adjust this path to the root of your project directory
    # The dataset expects to find the data at ../Data/HANCOCK relative to this path.
    os.chdir("/home/Guanjq/NewWork/MedAlignFusion/Code") 
    
    try:
        print("\nInitializing 'train' dataset with all modalities...")
        # Test with all modalities
        train_dataset = HANCOCKDataset(mode='train', modalities="all")
        print(f"Successfully loaded the train dataset with {len(train_dataset)} patients.")
        
        if len(train_dataset) > 0:
            # --- Demonstrate usage with DataLoader and custom collate function ---
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=hancock_custom_collate_fn
            )

            print("\n--- Testing DataLoader with custom collate function ---")
            first_batch = next(iter(train_loader))

            print(f"Batch keys: {first_batch.keys()}")
            print(f"Patient IDs in batch: {first_batch['patient_id']}")
            print(f"Label events tensor shape: {first_batch['label_Y'].shape}")
            print(f"Label times tensor: {first_batch['label_c']}")

            # --- Inspect each modality in the batch ---
            
            # 1. Image Modality
            if 'image' in first_batch:
                image_batch = first_batch['image']
                print("\n--- Image Modality ---")
                print(f"Image data type: {type(image_batch)}")
                if isinstance(image_batch, list):
                    print(f"Number of image tensors in batch: {len(image_batch)}")
                    for i, tensor in enumerate(image_batch):
                        if tensor is not None:
                            print(f"  - Image tensor {i} shape: {tensor.shape}")
                        else:
                            print(f"  - Image tensor {i}: None (patient might be missing this modality)")
            
            # 2. Strong Related Text Modality
            if 'strong_related_text' in first_batch:
                strong_text_batch = first_batch['strong_related_text']
                print("\n--- Strong Related Text Modality ---")
                print(f"Strong text data type: {type(strong_text_batch)}")
                if isinstance(strong_text_batch, list) and strong_text_batch:
                    print(f"Number of strong text items in batch: {len(strong_text_batch)}")
                    print(f"  - Example of first strong text (first 100 chars): '{strong_text_batch[0][:100]}...'")

            # 3. Weak Related Text Modality
            if 'weak_related_text' in first_batch:
                weak_text_batch = first_batch['weak_related_text']
                print("\n--- Weak Related Text Modality ---")
                print(f"Weak text data type: {type(weak_text_batch)}")
                if isinstance(weak_text_batch, list) and weak_text_batch:
                    print(f"Number of weak text items in batch: {len(weak_text_batch)}")
                    print(f"  - Example of first weak text (first 100 chars): '{weak_text_batch[0][:100]}...'")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

