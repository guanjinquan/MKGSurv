from torch.utils.data import sampler
import numpy as np
import random
import torch.distributed as dist




# ==========================================================================================
# Sampler Classes  
# ==========================================================================================
class BalancedBatchSampler(sampler.Sampler):
    """
    A sampler that performs balanced sampling to handle class imbalance.
    It ensures that each batch contains samples from different classes in a balanced way.
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        
        # Get all labels from the dataset
        labels = dataset.get_survival_bins()
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
        self.labels = dataset.get_survival_bins()
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
