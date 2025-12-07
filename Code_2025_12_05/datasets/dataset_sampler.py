from torch.utils.data import sampler
import numpy as np
import random
import torch.distributed as dist




# ==========================================================================================
# Sampler Classes  
# ==========================================================================================
class MixUpBalancedBatchSampler(sampler.Sampler):
    """
    A sampler that performs balanced sampling to handle class imbalance.
    It ensures that each batch contains samples from different classes in a balanced way.
    """
    def __init__(self, do_mixup, dataset):
        super().__init__(dataset)
        
        # Get all labels from the dataset
        labels = dataset.get_survival_bins()
        self.dataset = dict()      # {label0: [indices of label0], label1: [...]}
        self.balanced_max = 0      # The size of the largest class
        self.do_mixup = do_mixup
        self.dataset_size = len(dataset)
        
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
            item_idx = self.dataset[self.keys[self.currentkey_idx]][self.indices[self.keys[self.currentkey_idx]]]
            yield item_idx                          # yield the idx without mixup
            if self.do_mixup:
                yield item_idx + self.dataset_size  # yield the idx with mixup
            self.currentkey_idx = (self.currentkey_idx + 1) % len(self.keys)
        self.indices = self._init_indices()

    def __len__(self):
        """Returns the total number of samples after balancing."""
        return self.balanced_max * len(self.keys)

