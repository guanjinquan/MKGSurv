
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from modules.general_utils.config import parse_arguments
import numpy as np
import random
from modules.trainer import Trainer
import swanlab


if __name__ == '__main__':
    args = parse_arguments()
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    # print("args.gpu_id = ", args.gpu_id)

    os.chdir(os.path.dirname(__file__))

    import torch
    torch.multiprocessing.set_start_method('spawn')
 
    
    # start work
    trainer = Trainer(args=args)
        
    # if trainer.local_rank == 0:
    #     swanlab.init(
    #         project="NewMedAlignFusion",
    #         name=f"{args.model_task}-{args.fusion_type}-{args.runs_id}",
    #         config={
    #             'batch_size': args.batch_size * args.acc_step,
    #             'model_task': args.model_task, 
    #             'decode_task': args.decode_task, 
    #             'dataset': args.dataset, 
    #             'num_epochs': args.num_epochs,
    #             'learning_rate': args.learning_rate,
    #             'backbone_lr': args.backbone_lr,
    #             'weight_decay': args.weight_decay,
    #             'backbones': args.model_task,
    #             "fusion_type": args.fusion_type,
    #             'optimizer': args.optimizer,
    #             'scheduler': args.scheduler,
    #             'seed': args.seed,
    #             "modalities": args.modalities,
    #         },
    #         settings=swanlab.Settings(_service_wait=300)
    #     )
        
    trainer.run()
    
    # if trainer.local_rank == 0:
    #     swanlab.finish()
