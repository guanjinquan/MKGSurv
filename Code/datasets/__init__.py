import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.multi_oscc_dataset import MultiOSCCDataset
from datasets.hancock_dataset import HANCOCKDataset
from datasets.multi_oscc_IT_dataset import MultiOSCCITDataset
from datasets.multi_oscc_split_text_dataset import MultiOSCCSplitDataset
from datasets.oscc_surv_inhouse_dataset import OSCCSurvInHouseDataset
from datasets.dataset_sampler import BalancedBatchSampler, DistributedBalancedBatchSampler
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List



def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batches of dictionaries with various data types.
    It stacks labels into tensors but keeps modalities with variable sizes (like WSI embeddings)
    or text data as lists.
    """
    if not batch:
        return {}
    
    # Filter out any None items that might have resulted from errors in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    keys = batch[0].keys()
    collated_batch = {}

    for key in keys:
        collated_batch[key] = [item[key] for item in batch]
            
    return collated_batch



def GetDataLoader(args):
    train_set = GetDataset("train", args)
    valid_set = GetDataset("valid", args)
    test_set = GetDataset("test", args)
    
    if args.use_ddp:
        num_gpus = torch.cuda.device_count()
        assert args.batch_size % num_gpus == 0, "Batch size should be divisible by number of GPUs"
        train_loader = DataLoader(train_set, batch_size=args.batch_size // num_gpus,
            sampler=DistributedBalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
        print("Using DDP with batch size: ", args.batch_size // num_gpus)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, 
            sampler=BalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
        print("Using batch size: ", args.batch_size)


    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader



def GetDataset(mode, args):
    dataset = args.dataset
    if dataset == "multi_oscc":
        return MultiOSCCDataset(mode=mode, modalities=args.modalities)
    elif dataset == "hancock":
        return HANCOCKDataset(mode=mode, modalities=args.modalities)
    elif dataset == "multi_oscc_it":
        return MultiOSCCITDataset(mode=mode, modalities=args.modalities)
    elif dataset == "multi_oscc_split":
        return MultiOSCCSplitDataset(mode=mode, modalities=args.modalities)
    elif dataset == "oscc_inhouse":
        return OSCCSurvInHouseDataset(mode=mode, modalities=args.modalities)
    else:
        raise ValueError(f"Dataset {dataset} not supported")