from torch.utils.data import Dataset, DataLoader, sampler
import numpy as np
import random
import os
import json
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomAutocontrast, RandomAdjustSharpness, RandomResizedCrop
import pandas as pd # <-- NEW: Import pandas

# ==========================================================================================
# Sampler Classes (No Changes)
# ==========================================================================================

class BalancedBatchSampler(sampler.Sampler):
    """
    A sampler that performs balanced sampling to handle class imbalance.
    It ensures that each batch contains samples from different classes in a balanced way.
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        
        # Get all labels from the dataset
        labels = dataset._get_labels()
        self.dataset = dict()      # {label0: [indices of label0], label1: [...]}
        self.balanced_max = 0      # The size of the largest class
        
        # Group indices by class label
        for idx in range(len(dataset)):
            key = labels[idx]
            if key not in self.dataset:
                self.dataset[key] = list()
            self.dataset[key].append(idx)
            # Find the size of the largest class
            if len(self.dataset[key]) > self.balanced_max:
                self.balanced_max = len(self.dataset[key])

        # Oversample smaller classes to match the size of the largest class
        for key in self.dataset.keys():
            while len(self.dataset[key]) < self.balanced_max:
                self.dataset[key].append(random.choice(self.dataset[key]))
        
        self.keys = list(self.dataset.keys())
        self.currentkey_idx = 0
        self.indices = self._init_indices()

    def _init_indices(self):
        """Resets the indices for iteration."""
        indices = dict()
        for key in self.keys:
            indices[key] = -1
        return indices
    
    def __iter__(self):
        """Returns an iterator that yields sample indices in a balanced manner."""
        for key in self.keys:
            random.shuffle(self.dataset[key])
            
        while self.indices[self.keys[self.currentkey_idx]] < self.balanced_max - 1:
            self.indices[self.keys[self.currentkey_idx]] += 1
            yield self.dataset[self.keys[self.currentkey_idx]][self.indices[self.keys[self.currentkey_idx]]]
            self.currentkey_idx = (self.currentkey_idx + 1) % len(self.keys)
        self.indices = self._init_indices()

    def __len__(self):
        """Returns the total number of samples after balancing."""
        return self.balanced_max * len(self.keys)


class DistributedBalancedBatchSampler(sampler.Sampler):
    """
    A distributed version of the BalancedBatchSampler for use with PyTorch's DistributedDataParallel.
    """
    def __init__(self, dataset, seed=0):
        super().__init__(dataset)

        self.seed = seed
        self.labels = dataset._get_labels()
        self.length = len(dataset)
        self.class_nums = len(set(self.labels))
        assert self.class_nums > 1, "class_nums must be greater than 1"
        self.build_sampler(self.seed)

    def build_sampler(self, seed=0):
        """Builds the sampler for the current rank in the distributed setup."""
        self.dataset = dict()      # {label0: [indices of label0], label1: [...]}
        self.balanced_max = 0      # The size of the largest class on this rank
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Assign indices to the current rank
        np.random.seed(seed)
        for idx in np.random.permutation(self.length):
            if (idx - rank) % world_size != 0:
                continue
            label = self.labels[idx]
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            if len(self.dataset[label]) > self.balanced_max:
                self.balanced_max = len(self.dataset[label])

        # Oversample smaller classes
        for label in self.dataset.keys():
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        
        self.keys = list(self.dataset.keys())
        self.currentkey_idx = 0
        self.indices = self._init_indices()

    def _init_indices(self):
        """Resets the indices for iteration."""
        indices = dict()
        for key in self.keys:
            indices[key] = -1
        return indices

    def set_epoch(self, epoch):
        """Sets the seed for a new epoch to ensure different shuffling."""
        self.build_sampler(seed=epoch + self.seed)

    def __iter__(self):
        """Returns an iterator that yields sample indices in a balanced manner for the current rank."""
        for key in self.keys:
            random.shuffle(self.dataset[key])
            
        while self.indices[self.keys[self.currentkey_idx]] < self.balanced_max - 1:
            self.indices[self.keys[self.currentkey_idx]] += 1
            yield self.dataset[self.keys[self.currentkey_idx]][self.indices[self.keys[self.currentkey_idx]]]
            self.currentkey_idx = (self.currentkey_idx + 1) % len(self.keys)
        self.indices = self._init_indices()

    def __len__(self):
        """Returns the total number of samples for the current rank after balancing."""
        return self.balanced_max * len(self.keys)


# ==========================================================================================
# Custom Transforms and Dataset Class
# ==========================================================================================


def TrainTransforms():
    """Returns a composition of transforms for training data augmentation."""
    return Compose([
        RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        RandomVerticalFlip(p=0.5), 
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=(-45, 45)),
        RandomAutocontrast(p=0.5), 
        RandomAdjustSharpness(sharpness_factor=3, p=0.5),
    ])

class MultiOSCCDataset(Dataset):
    """
    Updated Dataset class to load pre-processed images from .npy files.
    Each .npy file corresponds to a patient and contains a stack of 6 images.
    """
    def __init__(self, mode="train"):
        super().__init__()
        assert mode in ["train", "valid", "test"], "mode must be one of 'train', 'valid', or 'test'"

        self.dataset_dir = os.path.join(os.getcwd(), "../Data/Multi-OSCCPI-Dataset")
        # --- MODIFIED: Point to the new directory with .npy files ---
        self.npy_dir = os.path.join(self.dataset_dir, "Multi-OSCCPI-Patches-1536-224")

        # --- Member variables ---
        self.mode = mode
        self.items = []
        self.transforms = None
        self.clinical_df = None  # To store clinical data
        self.num_classes = 2

        # --- Initialization logic ---
        self._load_items()

        if mode == "train":
            self.transforms = TrainTransforms()

        # Shuffle items if in training mode
        if mode == "train":
            random.shuffle(self.items)

        print(f"Dataset loaded: mode='{self.mode}'. Length: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _load_items(self):
        """Loads dataset items and metadata based on the specified mode."""
        metadata_path = os.path.join(self.dataset_dir, "all_metadata.json")
        split_path = os.path.join(self.dataset_dir, "split_seed=2024.json")
        clinical_data_path = os.path.join(self.dataset_dir, "clinical_data.csv")

        # Load clinical data using pandas
        try:
            self.clinical_df = pd.read_csv(clinical_data_path)
            if 'PID' in self.clinical_df.columns:
                self.clinical_df.set_index('PID', inplace=True)
            print("Successfully loaded clinical data.")
        except FileNotFoundError:
            print(f"Error: Clinical data file not found at {clinical_data_path}")
            raise

        with open(metadata_path, 'r') as f:
            all_patients_info = {item['pid']: item for item in json.load(f)['datainfo']}

        with open(split_path, 'r') as f:
            split_data = json.load(f)

        target_pids = set(split_data[self.mode])

        self.items = [
            all_patients_info[pid] for pid in target_pids
            if pid in all_patients_info
        ]

        if len(self.items) != len(target_pids):
            print(f"Warning: Some patient IDs were not found in metadata. Found {len(self.items)} out of {len(target_pids)}.")

        # --- MODIFIED: Check for .npy files instead of image directories ---
        assert self._check_npy_files(), "One or more .npy files are missing!"

    def _generate_clinical_text(self, patient_series):
        """Generates natural language descriptions from a patient's clinical data. (No changes needed here)"""
        strong_sentences = []
        weak_sentences = []

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
            # "Recurrence(0no/1yes)", 
            # # 不能输入标签
        ]

        def add_sentence(text_list, column_name, value):
            if pd.isna(value) or str(value).strip() in ['/', '']:
                return

            if isinstance(value, float) and value.is_integer():
                value = int(value)

            sentence = ""
            if column_name == "TumorT":
                sentence = f"The primary tumor stage (T stage) is {value}."
            elif column_name == "TumorN":
                sentence = f"The regional lymph node stage (N stage) is {value}."
            elif column_name == "TumorM":
                sentence = f"The distant metastasis stage (M stage) is {value}."
            elif column_name == "TumorDifferentiation(1high/2med/3low)":
                diff_map = {1: "well-differentiated (high grade)", 2: "moderately-differentiated (medium grade)", 3: "poorly-differentiated (low grade)"}
                sentence = f"The tumor differentiation is {diff_map.get(value, 'not specified')}."
            elif "(0/1)" in column_name or "(+)" in column_name:
                status = "present" if value == 1 else "absent"
                feature_name = column_name.replace("(0/1)", "").replace("(+)", "").replace("_", " ")
                sentence = f"{feature_name} is {status}."
            elif "(0no/1yes)" in column_name:
                status = "yes" if value == 1 else "no"
                feature_name = column_name.replace("(0no/1yes)", "").replace("History", " history")
                sentence = f"The patient has a record of {feature_name}: {status}."
            elif column_name == "Age(Y)":
                sentence = f"The patient's age is {value} years."
            elif column_name == "Gender(0male/1female)":
                gender = "female" if value == 1 else "male"
                sentence = f"The patient is {gender}."
            elif column_name in ["Pathology", "SurgicalMethod", "TumorLocation", "Ki-67", "PD_L1"]:
                sentence = f"The {column_name.lower()} is recorded as: {value}."

            if sentence:
                text_list.append(sentence)

        for col in strong_cols:
            if col in patient_series:
                add_sentence(strong_sentences, col, patient_series[col])

        for col in weak_cols:
            if col in patient_series:
                add_sentence(weak_sentences, col, patient_series[col])

        strong_text = "Pathological findings include: " + " ".join(strong_sentences) if strong_sentences else "No detailed pathological information available."
        weak_text = "Clinical and demographic profile: " + " ".join(weak_sentences) if weak_sentences else "No detailed clinical information available."

        return strong_text, weak_text

    def __getitem__(self, index):
        item_info = self.items[index]
        pid = item_info['pid']
        label = item_info['REC']

        # Clinical text generation remains the same
        if self.clinical_df is None or pid not in self.clinical_df.index:
            strong_text, weak_text = None, None
            print(f"Modal missing for PID {pid}: clinical data not found. strong/weak text set to None.")
        else:
            patient_series = self.clinical_df.loc[pid]
            strong_text, weak_text = self._generate_clinical_text(patient_series)

        # --- MODIFIED: Load images from a single .npy file ---
        # The new logic is much simpler as you've handled missing data.
        npy_path = os.path.join(self.npy_dir, f"{pid}.npy")
        
        try:
            # Load the numpy array which should contain 6 images
            images_array = np.load(npy_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"NPY file not found for PID {pid} at path: {npy_path}")

        # The loaded array is assumed to be in shape (24, H, W, C), e.g., (24, 224, 224, 3)
        assert images_array.shape[0] == 24, \
            f"Expected 24 images in the npy file, but got {images_array.shape[0]} for PID {pid}."

        loaded_images = []
        for i in range(images_array.shape[0]):
            # Convert each numpy array slice to a PIL Image for the transforms
            img_array = images_array[i]
            img = Image.fromarray(img_array)
            loaded_images.append(img)
        # --- End of modification ---
        
        # Apply transformations to each image
        if self.transforms is not None:
            transformed_images = []
            for img in loaded_images:
                transformed_images.append(self.transforms(img))
        else:
            # If no transforms, you might need to convert images to tensors manually
            # For simplicity, this example assumes transforms are always present for training
            transformed_images = loaded_images

        # one-hot label
        one_hot_label = torch.zeros(self.num_classes, dtype=torch.long)
        one_hot_label[label] = 1

        return {
            "images": transformed_images,
            "labels": one_hot_label,
            "strong_related_text": strong_text,
            "weak_related_text": weak_text,
        }

    def _get_labels(self):
        """Returns a list of all labels in the dataset."""
        return [item['REC'] for item in self.items]

    # --- MODIFIED: Renamed and updated to check for .npy files ---
    def _check_npy_files(self):
        """Verifies that .npy files exist for all patients in the dataset split."""
        missing_files = []
        for item in self.items:
            pid = item['pid']
            npy_path = os.path.join(self.npy_dir, f"{pid}.npy")
            if not os.path.exists(npy_path):
                missing_files.append(pid)
        
        if missing_files:
            print(f"Error: The following {len(missing_files)} patient .npy files are missing:")
            print(missing_files)
            return False
        
        print("All patient .npy files verified.")
        return True
    
    

# ==========================================================================================
# Custom Collate Function (MODIFIED)
# ==========================================================================================
def mutli_oscc_custom_collate_fn(batch):
    """
    Custom collate function to handle batches of dictionaries containing PIL images and text.
    """
    # batch is a list of dictionaries like:
    # [{'images': [img1, img2, ...], 'labels': tensor(0), 'strong_related_text': "...", 'weak_related_text': "..."}, ...]
    
    images = [item['images'] for item in batch]
    labels = [item['labels'] for item in batch]
    strong_texts = [item['strong_related_text'] for item in batch] # <-- NEW
    weak_texts = [item['weak_related_text'] for item in batch]     # <-- NEW
    
    return {
        'images': images,
        'labels': torch.stack(labels),
        'strong_related_text': strong_texts, # <-- NEW
        'weak_related_text': weak_texts      # <-- NEW
    }


# ==========================================================================================
# Example Usage (MODIFIED)
# ==========================================================================================

if __name__ == '__main__':
    # Adjust this path to the root of your project structure
    # This ensures the relative path to Data works correctly.
    os.chdir("/home/Guanjq/NewWork/MedAlignFusion/Code")
    
    # 1. Create the training dataset
    train_dataset = MultiOSCCDataset(mode="train")

    # 2. Get label distribution
    labels = train_dataset._get_labels()
    print(f"\nOriginal training set distribution: Class 0: {labels.count(0)}, Class 1: {labels.count(1)}")

    # 3. Create the BalancedBatchSampler
    train_sampler = BalancedBatchSampler(train_dataset)
    print(f"Total samples after balancing with sampler: {len(train_sampler)}")

    # 4. Create the DataLoader with the custom sampler and collate function
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=mutli_oscc_custom_collate_fn # Use the updated custom collate function
    )

    # 5. Iterate through the DataLoader to see the balanced batches with text
    print("\nIterating through a few batches from the DataLoader...")
    num_batches_to_show = 2
    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_show:
            break
        
        batch_labels = batch['labels'].tolist()
        
        print(f"--- Batch {i+1} ---")
        print(f"  Labels: {batch_labels}")
        print(f"  Distribution: Class 0: {batch_labels.count(0)}, Class 1: {batch_labels.count(1)}")
        print(f"  Number of samples in batch: {len(batch['images'])}")
        
        # <-- MODIFIED: Print the generated text for the first item in the batch -->
        print("\n  --- Text for first sample in batch ---")
        print(f"  Strongly-Related Text:\n  '{batch['strong_related_text'][0]}'")
        print(f"\n  Weakly-Related Text:\n  '{batch['weak_related_text'][0]}'\n")
