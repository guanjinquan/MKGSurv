
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from modules.general_utils.config import parse_arguments
import numpy as np
import random
from modules.trainer import Trainer
from modules.tester import Tester
import swanlab


if __name__ == '__main__':
    args = parse_arguments()
    os.chdir(os.path.dirname(__file__))

    import torch
    torch.multiprocessing.set_start_method('spawn')

    # start work
    summary_of_folds_valid = {}
    summary_of_folds_test = {}

    for fold in range(5):
        args.fold = fold
        
        # Training 
        try:
            trainer = Trainer(args=args)
            trainer.run()
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            
        
        # Testing
        tester = Tester(args=args)
        valid_metrics = tester.valid()
        test_metrics = tester.test()

        for key, value in valid_metrics.items():
            summary_of_folds_valid[key] = summary_of_folds_valid.get(key, []) + [value]
        for key, value in test_metrics.items():
            summary_of_folds_test[key] = summary_of_folds_test.get(key, []) + [value]
    
    # Print Summary
    print("Validation Summary:")
    for key, value in summary_of_folds_valid.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        print(f" - List = {value}")
    print("Test Summary:")
    for key, value in summary_of_folds_test.items():
        print(f"{key}: {np.mean(value):.4f} ± {np.std(value):.4f}")
        print(f" - List = {value}")

