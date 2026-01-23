import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MLP Model')

    # path settings
    parser.add_argument('--ckpt_path', type=str, default='../Checkpoints/', help='the path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='../Results', help='the path to save log')
    parser.add_argument('--load_pth_path', type=str)
    parser.add_argument('--points_save_path', type=str, default=None, help='the path to save points of medical knowledge guided fusion')
    parser.add_argument('--view_groups_attention_path', type=str, default=None, help='the path to save attention weights')


    # dataset settings 
    parser.add_argument('--dataset', type=str, default="multi_oscc")
    parser.add_argument('--modalities', type=str, default="all")
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--do_mixup", action='store_true')
    parser.add_argument("--knowledge_source", type=str, default="deepseek")
    parser.add_argument("--use_medical_knowledge", action='store_true')
    parser.add_argument("--knowledge_type", type=str, default="all", help="knowledge_type: [all, survival, relationship]")
    
    # models settings 
    parser.add_argument('--model_task', type=str, default='multi_oscc', help="model_task: [multi_oscc]")
    parser.add_argument('--decode_task', type=str, default='surv_pred', help="decode_task: only support [surv_pred, treatment_pred]")
    parser.add_argument('--image_aggregater', type=str, default='transmil', help="image_aggregater: only support [transmil, panther]")
    parser.add_argument('--fusion_type', type=str, default='hier_align', help="fusion_block: [hier_align, concat, LMF, gated, msa, i2moe, healnet]")
    parser.add_argument('--with_multimodal_align', action='store_true')
    parser.add_argument('--with_multimodal_vib', action='store_true')
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--kl_loss_weight', type=int, default=None)
    
    # trainer settings
    parser.add_argument("--runs_id", type=str)
    parser.add_argument("--acc_step", type=int)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=109, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
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
    
    
    # Tester settings
    parser.add_argument('--draw_kaplan_meier', action='store_true')
    
    
    args = parser.parse_args()

    return args