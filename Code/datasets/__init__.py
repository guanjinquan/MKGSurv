import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.multi_oscc_dataset import MultiOSCCDataset, mutli_oscc_custom_collate_fn
from datasets.hancock_dataset import HANCOCKDataset, hancock_custom_collate_fn
from datasets.dataset_sampler import BalancedBatchSampler, DistributedBalancedBatchSampler
import torch
from torch.utils.data import DataLoader



def GetDataLoader(args):
    dataset = args.dataset
    if dataset == "multi_oscc":
        return GetMultiOSCCDataLoader(args)
    elif dataset == "hancock":
        return GetHancockDataLoader(args)
    
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")



def GetMultiOSCCDataLoader(args):
    
    train_set = GetDataset("train", args)
    valid_set = GetDataset("valid", args)
    test_set = GetDataset("test", args)
    

    if args.use_ddp:
        num_gpus = torch.cuda.device_count()
        assert args.batch_size % num_gpus == 0, "Batch size should be divisible by number of GPUs"
        train_loader = DataLoader(train_set, batch_size=args.batch_size // num_gpus,
            sampler=DistributedBalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=mutli_oscc_custom_collate_fn)
        print("Using DDP with batch size: ", args.batch_size // num_gpus)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, 
            sampler=BalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=mutli_oscc_custom_collate_fn)
        print("Using batch size: ", args.batch_size)


    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=mutli_oscc_custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=mutli_oscc_custom_collate_fn)

    return train_loader, valid_loader, test_loader



def GetHancockDataLoader(args):
    train_set = GetDataset("train", args)
    valid_set = GetDataset("valid", args)
    test_set = GetDataset("test", args)
    

    if args.use_ddp:
        num_gpus = torch.cuda.device_count()
        assert args.batch_size % num_gpus == 0, "Batch size should be divisible by number of GPUs"
        train_loader = DataLoader(train_set, batch_size=args.batch_size // num_gpus,
            sampler=DistributedBalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=hancock_custom_collate_fn)
        print("Using DDP with batch size: ", args.batch_size // num_gpus)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, 
            sampler=BalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=hancock_custom_collate_fn)
        print("Using batch size: ", args.batch_size)


    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=hancock_custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=hancock_custom_collate_fn)

    return train_loader, valid_loader, test_loader



def GetDataset(mode, args):
    dataset = args.dataset
    if dataset == "multi_oscc":
        return MultiOSCCDataset(mode=mode, modalities=args.modalities)
    elif dataset == "hancock":
        return HANCOCKDataset(mode=mode, modalities=args.modalities)
    else:
        raise ValueError(f"Dataset {dataset} not supported")