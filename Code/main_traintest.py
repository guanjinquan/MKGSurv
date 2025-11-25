
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
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    # print("args.gpu_id = ", args.gpu_id)

    os.chdir(os.path.dirname(__file__))

    import torch
    torch.multiprocessing.set_start_method('spawn')
 
    # start work
    print(f"INFO: Current training fold is : {args.fold}")

    # Training 
    trainer = Trainer(args=args)
    trainer.run()

    # Testing
    tester = Tester(args=args)
    valid_metrics = tester.valid()
    test_metrics = tester.test()

    print("valid_metrics = ", valid_metrics)
    print("test_metrics = ", test_metrics)