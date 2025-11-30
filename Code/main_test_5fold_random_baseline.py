
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from modules.general_utils.config import parse_arguments
import numpy as np
import random
from modules.trainer import Trainer
from modules.tester import Tester
from modules.random_tester import RandomTester
import swanlab


if __name__ == '__main__':
    args = parse_arguments()
    os.chdir(os.path.dirname(__file__))

    import torch
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # start work
    summary_of_folds_test = {}

    for fold in range(5):
        args.fold = fold

        # Random Baseline
        random_tester = RandomTester(args=args)
        test_metrics = random_tester.test()

        for key, value in test_metrics.items():
            summary_of_folds_test[key] = summary_of_folds_test.get(key, []) + [value]
    
    # Print Summary
    print("Test Summary:")
    for key, value in summary_of_folds_test.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        print(f" - List = {value}")

