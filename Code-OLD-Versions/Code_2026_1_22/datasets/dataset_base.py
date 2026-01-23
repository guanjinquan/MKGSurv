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
import joblib # 导入 joblib
from sklearn.cluster import KMeans # <-- 新增导入



class MultiModalDataset(Dataset):
    def get_survival_bins(self):
        raise NotImplementedError

    def parse_modalities(self, modalities_str: str) -> List[str]:
        raise NotImplementedError

    def get_active_modalities(self):
        raise NotImplementedError

    def get_training_image_embeddings_prototypes(self, num_prototypes=64):
        # Support for Panther modal
        raise NotImplementedError
