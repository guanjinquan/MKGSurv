
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from modules.general_utils.config import parse_arguments
import numpy as np
import random
from modules.trainer import Trainer
from modules.tester import Tester

try:
    import swanlab
except ImportError:
    swanlab = None


if __name__ == '__main__':
    args = parse_arguments()
    os.chdir(os.path.dirname(__file__))

    import torch
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    # start work
    summary_of_folds_valid = {}
    summary_of_folds_test = {}
    run_path = [args.model_task, args.runs_id + "+" + args.fusion_type]

    if "medkgat_fusion" in args.fusion_type:
        args.points_save_path = os.path.join(args.log_path, "draw", "points_" + args.runs_id + "+" + args.fusion_type + ".jsonl")

    for fold in range(5):
        args.fold = fold

        # Testing
        tester = Tester(args=args)
        # valid_metrics = tester.valid()
        test_metrics = tester.test()

        # for key, value in valid_metrics.items():
        #     summary_of_folds_valid[key] = summary_of_folds_valid.get(key, []) + [value]
        for key, value in test_metrics.items():
            summary_of_folds_test[key] = summary_of_folds_test.get(key, []) + [value]
    
    # Print Summary and save in run_path/summary.txt
    os.makedirs(os.path.join(args.log_path, *run_path), exist_ok=True)
    with open(os.path.join(args.log_path, *run_path, "summary.txt"), "w") as f:
        f.write("Validation Summary:\n")
        for key, value in summary_of_folds_valid.items():
            f.write(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}\n")
            f.write(f" - List = {value}\n")
        f.write("\nTest Summary:\n")
        for key, value in summary_of_folds_test.items():
            f.write(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}\n")
            f.write(f" - List = {value}\n")

    print("Validation Summary:")
    for key, value in summary_of_folds_valid.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        print(f" - List = {value}")
    print("Test Summary:")
    for key, value in summary_of_folds_test.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        print(f" - List = {value}")
