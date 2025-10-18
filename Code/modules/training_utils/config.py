import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MLP Model')

    # path settings
    parser.add_argument('--ckpt_path', type=str, default='../Checkpoints/', help='the path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='../Results', help='the path to save log')
    parser.add_argument('--load_pth_path', type=str)
    
    # dataset settings 
    parser.add_argument('--dataset', type=str, default="multi_oscc")
    parser.add_argument('--modalities', type=str, default="all")
    parser.add_argument("--debug_mode", type=bool, default=False)
    
    # models settings 
    parser.add_argument('--model_task', type=str, default='multi_oscc', help="model_task: [multi_oscc]")
    parser.add_argument('--fusion_type', type=str, default='hier_align', help="fusion_block: [hier_align, concat, LMF, gated, msa, i2moe, healnet]")
    parser.add_argument('--with_multimodal_align', action='store_true')
    
    # trainer settings
    parser.add_argument("--runs_id", type=str)
    parser.add_argument("--acc_step", type=int)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=109, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--backbone_lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choose optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='choose scheduler')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--freezed_backbone', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    
    
    args = parser.parse_args()

    return args