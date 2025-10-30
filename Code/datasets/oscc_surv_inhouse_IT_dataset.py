from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np
import random
import os
import json
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomAutocontrast, \
    RandomAdjustSharpness, RandomResizedCrop, Normalize, ToTensor, Resize
import pandas as pd 
from typing import List, Dict, Any



# ==========================================================================================
# Custom Transforms and Dataset Class
# ==========================================================================================

MEAN=[175.14728804175988, 110.57123792228117, 176.73598615775617]
STD=[21.239463551725915, 39.15991384752335, 10.99100631656543]
MEAN = [m / 255.0 for m in MEAN]
STD = [s / 255.0 for s in STD]


def TrainTransforms():
    """Returns a composition of transforms for training data augmentation."""
    return Compose([
        RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
        RandomVerticalFlip(p=0.5), 
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=(-45, 45)),
        RandomAutocontrast(p=0.5), 
        RandomAdjustSharpness(sharpness_factor=3, p=0.5),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])

def InferTransforms():
    """Returns a composition of transforms for inference."""
    return Compose([
        Resize(size=(512, 512)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ])



class OSCCSurvInHouseITDataset(Dataset):
    """
    Updated Dataset class that dynamically loads data based on requested modalities.
    - It loads pre-processed images from .npy files.
    - It skips patients who are missing all of the requested modalities.
    """
    def __init__(self, mode="train", modalities="all"):
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"

        self.dataset_dir = os.path.join(os.getcwd(), "../Data/Multi-OSCCPI-Dataset")
        self.npy_dir = os.path.join(self.dataset_dir, "Multi-OSCCPI-Npy-512")

        # --- Member variables ---
        self.mode = mode
        self.items = []
        self.transforms = None
        self.clinical_df = None

        # --- MODIFIED: Parse and store the list of required modalities ---
        self.modalities = self._parse_modalities(modalities)
        print(f"Dataset will be initialized for modalities!!: {self.modalities}")

        # --- Initialization logic ---
        self._load_clinical_data()
        self._load_and_filter_items() # Changed from _load_items

        # --- Survival Bins ---
        self.five_years_in_days = 5 * 365.0
        self.num_time_bins = 10
        self.time_bins = np.linspace(0, self.five_years_in_days, self.num_time_bins + 1)

        if mode == "train":
            self.transforms = TrainTransforms()
        else:
            self.transforms = InferTransforms()

        if mode == "train":
            random.shuffle(self.items)

        print(f"Dataset loaded: mode='{self.mode}'. Final valid item count: {len(self.items)}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single patient's data, including survival labels `label_Y` and `label_c`.
        """
        item_info = self.items[index]
        pid_int = int(item_info['pid'])  # Clinical_DF.inedx 是 Int64, 因此转成int访问
        output_dict = {"pid": pid_int}

        # --- Survival Labels (Y and c) ---
        has_recurrence = item_info.get('recurrence') == 'yes'
        time_to_recurrence = item_info.get('days_to_recurrence')
        time_to_last_info = item_info.get('days_to_last_information')

        event_time = -1.0
        censorship = 1 # c = 1 means censored, c = 0 means event occurred.

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

        # --- Labels (always included) ---
        output_dict['labels'] = {
            'label_Y': int(time_bin),
            'label_c': censorship,
        }

        output_dict['original_labels'] = {
            'label_Y': event_time,
            'label_c': censorship
        }



        # -- Dynamically build the output dictionary ---
        # --- Image modality ---
        if "images" in self.modalities:
            npy_path = os.path.join(self.npy_dir, f"{pid_int}.npy")
            try:
                images_array = np.load(npy_path)
                assert images_array.shape[0] == 6, f"Expected 6 images, got {images_array.shape[0]} for PID {pid_int}."
                
                loaded_images = [Image.fromarray(images_array[i]) for i in range(images_array.shape[0])]
                
                if self.transforms:
                    transformed_images = [self.transforms(img) for img in loaded_images]
                else:
                    transformed_images = loaded_images
                
                output_dict["images"] = transformed_images

            except (FileNotFoundError, AssertionError) as e:
                # This case should be rare due to the pre-filtering in _load_and_filter_items
                print(f"Warning: NPY file missing or invalid for {pid_int} at getitem: {e}")
                pass # Skip adding the 'images' key

        # --- Text modalities ---
        if "text" in self.modalities:

            # assert pid_int in self.clinical_df.index, f"Clinical data missing for PID {pid_int}, Index = {self.clinical_df.index[:100]}"

            if self.clinical_df is not None and pid_int in self.clinical_df.index:
                patient_series = self.clinical_df.loc[pid_int]  # ALL Value in clinical df is STRING
                strong_text, weak_text = self._generate_clinical_text(patient_series)
                output_dict["text"] = strong_text + " " + weak_text
            else:
                output_dict["text"] = None


        # --- Data Integrity Check ---
        # If no requested modalities were found for this patient, get the next one.
        modalities_found = sum(1 for m in self.modalities if output_dict.get(m) is not None and (isinstance(output_dict.get(m), str) and output_dict.get(m).strip() != "" or not isinstance(output_dict.get(m), str)))
        if modalities_found == 0:
            return self.__getitem__((index + 1) % len(self))

        return output_dict

    def __len__(self):
        return len(self.items)

    def _parse_modalities(self, modalities_str: str) -> List[str]:
        """Parses the modalities string into a list of valid modality keys."""
        if modalities_str == "all":
            return ["images", "text"]
    
        valid_set = {"images", "text"}
        # Allow for both '-' and ',' as separators
        parsed = [m.strip() for m in modalities_str.replace('-', ',').split(',')]
        
        for m in parsed:
            if m not in valid_set:
                raise ValueError(f"Invalid modality '{m}' specified. Must be one of {valid_set}")
        return parsed

    def _load_clinical_data(self):
        """Loads the clinical data CSV into a pandas DataFrame."""
        clinical_data_path = os.path.join(self.dataset_dir, "clinical_data.csv")
        try:
            self.clinical_df = pd.read_csv(clinical_data_path)
            if 'PID' in self.clinical_df.columns:
                self.clinical_df.set_index('PID', inplace=True)
            print("Successfully loaded clinical data.")
        except FileNotFoundError:
            print(f"Warning: Clinical data file not found at {clinical_data_path}. Text modalities will be unavailable.")
            self.clinical_df = None # Ensure it's None if file not found

    def _get_labels(self):
        """
        Returns a list of all labels (time bins) in the dataset.
        This is used by the SurvivalBalancedBatchSampler.
        """
        labels_y = []
        for index in range(len(self.items)):
            label_info = self.items[index]
            
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
    

    def _load_and_filter_items(self):
        """Loads dataset items and filters them based on modality availability."""
        metadata_path = os.path.join(self.dataset_dir, "oscc_recurrence_survival_data.json")
        split_path = os.path.join(self.dataset_dir, "split_seed=2024.json")
        print("Load Split File:", split_path)

        with open(metadata_path, 'r') as f:
            all_patients_info = {str(item['pid']): item for item in json.load(f)}  # ALL PID use String type
        print(len(all_patients_info))

        with open(split_path, 'r') as f:
            split_data = json.load(f)

        target_pids = set([str(pid) for pid in split_data[self.mode]])
        # self.patient_ids = list(target_pids)
        
        initial_items = [
            all_patients_info[pid] for pid in target_pids
            if pid in all_patients_info
        ]

        # Make sure the patient_ids are same order with self.items
        self.items = initial_items #[item for pid in self.patient_ids for item in initial_items if int(item['pid']) == str(pid)]

        print("Load data successfully")
        if len(self.items) != len(target_pids):
            print(f"Warning: Some PIDs from split file not found in metadata. Found {len(self.items)}/{len(target_pids)}.")

    def _generate_clinical_text(self, patient_series):
        """Generates natural language descriptions from clinical data. (No changes needed here)"""
        # This function remains unchanged from your original code.
        strong_sentences, weak_sentences = [], []
        strong_cols = [
            "TumorT", "TumorN", "TumorM", "Pathology", "TumorDifferentiation(1high/2med/3low)",
            "CancerThrombus(0/1)", "SurroundingTissueInvasion(0/1)", "SurgicalMargin(0/1)", "LNM(0/1)", "Ki-67",
            "CK5_6(0/1)", "P63(0/1)", "P16(0/1)", "HPV(0/1)", "PD_L1", "IA(+)", "IB(+)", "IIA(+)", "IIB(+)", "III(+)",
            "AccessoryChain(+)", "VascularInvasion(+)", "PerineuralInvasion(+)"
        ]
        weak_cols = [
            "Gender(0male/1female)", "Age(Y)", "Weight(kg)", "Height(cm)", "AlcoholHistory(0no/1yes)",
            "SmokingHistory(0no/1yes)", "BetelNutHistory(0no/1yes)", "SurgicalMethod", "TumorLocation",
            "PreoperativeHistory(0no/1yes)", "Diabetes(0no/1yes)", "RespiratoryDisease(0no/1yes)",
            "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)", "NeckMass(+)",
            "Metastasis(0no/1yes)", "Radiotherapy(0no/1yes)", "Chemotherapy(0no/1yes)"
        ]

        def add_sentence(text_list, column_name, value):
            if pd.isna(value) or str(value).strip() in ['/', '']: return
            if isinstance(value, float) and value.is_integer(): value = int(value)
            sentence = ""
            if column_name == "TumorT": sentence = f"The primary tumor stage (T stage) is {value}."
            elif column_name == "TumorN": sentence = f"The regional lymph node stage (N stage) is {value}."
            elif column_name == "TumorM": sentence = f"The distant metastasis stage (M stage) is {value}."
            elif column_name == "TumorDifferentiation(1high/2med/3low)":
                diff_map = {1: "well-differentiated", 2: "moderately-differentiated", 3: "poorly-differentiated"}
                sentence = f"The tumor differentiation is {diff_map.get(value, 'not specified')}."
            elif "(0/1)" in column_name or "(+)" in column_name:
                status = "present" if value == 1 else "absent"
                feature_name = column_name.replace("(0/1)", "").replace("(+)", "").replace("_", " ")
                sentence = f"{feature_name} is {status}."
            elif "(0no/1yes)" in column_name:
                status = "yes" if value == 1 else "no"
                feature_name = column_name.replace("(0no/1yes)", "").replace("History", " history")
                sentence = f"The patient has a record of {feature_name}: {status}."
            elif column_name == "Age(Y)": sentence = f"The patient's age is {value} years."
            elif column_name == "Gender(0male/1female)": sentence = f"The patient is {'female' if value == 1 else 'male'}."
            elif column_name in ["Pathology", "SurgicalMethod", "TumorLocation", "Ki-67", "PD_L1"]:
                sentence = f"The {column_name.lower()} is recorded as: {value}."
            if sentence: text_list.append(sentence)

        for col in strong_cols:
            if col in patient_series: add_sentence(strong_sentences, col, patient_series[col])
        for col in weak_cols:
            if col in patient_series: add_sentence(weak_sentences, col, patient_series[col])

        strong_text = "Pathological findings include: " + " ".join(strong_sentences) if strong_sentences else "No detailed pathological information available."
        weak_text = "Clinical and demographic profile: " + " ".join(weak_sentences) if weak_sentences else "No detailed clinical information available."

        return strong_text, weak_text


