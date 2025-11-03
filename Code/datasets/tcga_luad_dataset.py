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

        # --- Survival Analysis Parameters ---
        self.five_years_in_days = 5 * 365.0
        self.num_time_bins = 10
        self.time_bins = np.linspace(0, self.five_years_in_days, self.num_time_bins + 1)


        # --- Load and preprocess all data sources ---
        self._load_data()
        print(f"Dataset for mode '{self.mode}' initialized. Found {len(self.patient_ids)} patients.")


    def _parse_modalities(self, modalities_str: str) -> List[str]:
        """Parses the modalities string into a list of valid modality keys."""

        if modalities_str == "all":
            return [
                "image-pathology", 
                "genomics-genomics",
                "tabular-pathology", 
                "tabular-clinical", 
                "tabular-genomics",
            ]
        

        valid_modalities = {
            "image-pathology", 
            "genomics-genomics",
            "tabular-pathology", 
            "tabular-clinical", 
            "tabular-genomics",
        }




