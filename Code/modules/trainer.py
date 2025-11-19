import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import GetDataLoader
from modules.model import GetModel
from modules.training_utils.optims import GetOptimizer, GetScheduler
from modules.training_utils import Logger, save_trainer, save_model, load_model, load_trainer
from typing import Dict, List, Any

try:
    import swanlab
except ImportError:
    print("swanlab not installed, logging to swanlab will be disabled.")
    swanlab = None


def get_score(metrics):
    if "Accuracy_valid" in metrics:
        return metrics.get(f"Accuracy_valid", 0)
    
    elif "C-Index_valid" in metrics:
        return metrics.get(f"C-Index_valid", 0)

    else:
        raise NotImplementedError(f"No target metric found in {metrics} !!!")


class Trainer:
    def __init__(self, args):
        assert args is not None, 'Please input args!!!'
        self.args = args

        # --- DDP ADDITION: 初始化分布式环境 ---
        self.local_rank = 0
        if self.args.use_ddp:
            # local_rank 通常由 torchrun 或 torch.distributed.launch 自动设置
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        
        self.device = torch.device("cuda", self.local_rank)
        print("device ", self.device)
        
        # --- DDP CHANGE: 数据加载器需要 DistributedSampler ---
        self.train_loader, self.val_loader, self.test_loader = GetDataLoader(self.args)

        # --- DDP CHANGE: 将模型移动到正确的设备 ---
        dataset = self.train_loader.dataset
        self.model = GetModel(self.args, dataset).to(self.device)
        
        # --- DDP ADDITION: 使用 DDP 包装模型 ---
        if self.args.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        self.optimizer = GetOptimizer(self.args, self.model)
        self.scheduler = GetScheduler(self.args, self.optimizer)
        
        self.epoch = 0
        self.save_epoch_limit = max(20, self.args.num_epochs // 5)
        self.iters = 0
        self.acc_step = self.args.acc_step
            
        if self.args.continue_training:
            if self.local_rank == 0:
                print(f"Continue training from {self.args.load_pth_path} !!!", flush=True)
            
            # 需要确保所有进程都加载了相同的状态
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            ckp_trainer = load_trainer(self.args.load_pth_path)
            self.epoch = ckp_trainer.epoch
            self.optimizer.load_state_dict(ckp_trainer.optimizer.state_dict())
            self.scheduler.load_state_dict(ckp_trainer.scheduler.state_dict())
            
            model_state_dict = ckp_trainer.model.state_dict()
            model_to_load = self.model.module if self.args.use_ddp else self.model
            
            cleaned_state_dict = model_state_dict
            model_to_load.load_state_dict(cleaned_state_dict)
            del ckp_trainer
            
        elif self.args.finetune:
            if self.local_rank == 0:
                print(f"Fine-tune from {self.args.load_pth_path}!!!", flush=True)
            cp = load_model(self.args.load_pth_path)
            pretrain = cp['model']
            
            model_to_load = self.model.module if self.args.use_ddp else self.model
            pretrain_filtered = {k: v for k, v in pretrain.items() if k in model_to_load.state_dict()}
            model_to_load.load_state_dict(pretrain_filtered, strict=False)
        
        # amp
        self.scaler = GradScaler() if self.args.use_amp else None

        self.loss_history = []
        self.patience = 100
        self.monitor_length = 10
        self.best_metrics = {}
        self.best_score = 0
        
        # --- 只有主进程 (rank 0) 才进行日志记录和文件保存 ---
        if self.local_rank == 0:
            run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type]
            self.log_path = os.path.join(self.args.log_path, *run_path)
            self.ckpt_path = os.path.join(self.args.ckpt_path, *run_path) 
            print("log_path : ", self.log_path, flush=True)
            print("ckpt_path : ", self.ckpt_path, flush=True)
            
            if os.path.exists(os.path.join(self.ckpt_path, 'Final_Trainer.pkl')):
                raise ValueError("Trainer already exists!!!")
            
            os.makedirs(self.log_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.log = Logger(os.path.join(self.log_path, 'log.txt'))
            self.log.write("settings : " + str(args))

    def early_stop(self):
        epoch_run = len(self.loss_history)
        early_stop_flag = torch.zeros(1).to(self.device)
        
        if self.local_rank == 0 and epoch_run > self.patience and epoch_run > self.monitor_length:
            if self.loss_history[-1] >= np.mean(self.loss_history[-self.monitor_length:]):
                early_stop_flag[0] = 1
        
        if self.args.use_ddp:
            dist.all_reduce(early_stop_flag, op=dist.ReduceOp.SUM)
        
        if early_stop_flag.item() == 1:
            if self.local_rank == 0:
                print("Early stopping!!! on epoch " + str(self.epoch), flush=True)
                self.log.write("Early stopping!!! on epoch " + str(self.epoch))
            return True
        return False

    def run(self):
        start_epoch = self.epoch
        for epoch_id in range(start_epoch, self.args.num_epochs + 1):
            if epoch_id > start_epoch:
                self.train_epoch(self.train_loader)
                self.scheduler.step()
                # if self.local_rank == 0 and swanlab:
                #     for idx, groups in enumerate(self.optimizer.param_groups):
                #         swanlab.log({f"lr_{['backbones', 'others'][idx]}": groups['lr']}, step=self.epoch)

            # # DDP CHANGE: 评估需要在所有进程上运行
            if self.val_loader is not None:
                self.eval_epoch(self.val_loader, 'valid')
            if self.test_loader is not None:
                self.eval_epoch(self.test_loader, 'test')
            
            # DDP ADDITION: 等待所有进程完成评估
            if self.args.use_ddp:
                dist.barrier()
            
            # DDP CHANGE: 只有主进程保存、记录和检查早停
            if self.local_rank == 0:
                self.on_epoch_end()
            
            if self.early_stop():
                break

    def train_epoch(self, train_loader):
        self.model.train()
        if self.args.use_ddp:
            train_loader.sampler.set_epoch(self.epoch)
        
        # --- 主要改动 1: 初始化列表，用于收集训练过程中的结果 ---
        local_outs, local_true, local_loss = [], [], {}

        assert len(train_loader) > 0, f"train_loader is empty!!!"
        print(f"Training set length is {len(train_loader)}")

        self.optimizer.zero_grad()
        for i, batch_data in enumerate(train_loader, 1):
            batch_size = len(batch_data['pid'])

            # --- 训练步骤 (与之前相同) ---
            if self.args.use_amp:
                with autocast():
                    out = self.model(batch_size, batch_data)
                    total_loss = out['losses']['total_loss']
                    total_loss = total_loss / self.acc_step
                self.scaler.scale(total_loss).backward()
            else:
                out = self.model(batch_size, batch_data)
                total_loss = out['losses']['total_loss']
                total_loss = total_loss / self.acc_step
                total_loss.backward()
            
            if i % self.acc_step == 0 or i == len(train_loader):
                if self.args.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # --- 主要改动 2: 收集当前批次的结果 ---
            # .detach() 用于切断反向传播的计算图
            # .cpu() 将数据移至CPU，避免占用过多显存
            local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
            local_true.extend(batch_data['labels'])
            for k, v_loss in out['losses'].items():
                local_loss.setdefault(k, []).append(v_loss.item())

        # --- 主要改动 3: 训练结束后，像验证集一样，收集并计算指标 ---
        # DDP 模式下，收集所有进程的结果到主进程
        if self.args.use_ddp:
            for k in local_loss:
                loss_tensor = torch.tensor(np.mean(local_loss[k]), device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    local_loss[k] = [loss_tensor.item()]

            gathered_outs = [None] * dist.get_world_size()
            gathered_true = [None] * dist.get_world_size()
            dist.gather_object(local_outs, gathered_outs if self.local_rank == 0 else None, dst=0)
            dist.gather_object(local_true, gathered_true if self.local_rank == 0 else None, dst=0)

        # 只有主进程计算和记录指标
        if self.local_rank == 0:
            if self.args.use_ddp:
                final_outs = [item for sublist in gathered_outs for item in sublist]
                final_true = [item for sublist in gathered_true for item in sublist]
                final_loss = local_loss
            else:
                final_outs = local_outs
                final_true = local_true
                final_loss = local_loss
            
            # 调用 on_loader_exit 来处理训练集的结果
            self.on_loader_exit('train', final_loss, final_outs, final_true)

    def eval_epoch(self, val_loader, mode='valid'):
        self.model.eval()
        local_outs, local_true, local_loss = [], [], {}
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
                local_true.extend(batch_data['labels'])
                for k, v in out['losses'].items():
                    # OPTIMIZATION: 使用 setdefault 和 append
                    if isinstance(v, torch.Tensor):
                        local_loss.setdefault(k, []).append(v.item())
                    elif isinstance(v, float):
                        local_loss.setdefault(k, []).append(v)

        # DDP ADDITION: 收集所有进程的评估结果到主进程
        if self.args.use_ddp:
            # 收集 loss，并计算所有卡的平均值
            for k in local_loss:
                loss_tensor = torch.tensor(np.mean(local_loss[k]), device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    local_loss[k] = [loss_tensor.item()]

            # 收集 outs 和 true
            gathered_outs = [None] * dist.get_world_size()
            gathered_true = [None] * dist.get_world_size()
            dist.gather_object(local_outs, gathered_outs if self.local_rank == 0 else None, dst=0)
            dist.gather_object(local_true, gathered_true if self.local_rank == 0 else None, dst=0)
        
        # 只有主进程计算和记录指标
        if self.local_rank == 0:
            if self.args.use_ddp:
                final_outs = [item for sublist in gathered_outs for item in sublist]
                final_true = [item for sublist in gathered_true for item in sublist]
                final_loss = local_loss
            else:
                final_outs = local_outs
                final_true = local_true
                final_loss = local_loss
            self.on_loader_exit(mode, final_loss, final_outs, final_true) 

    def on_epoch_end(self):
        # # 这个函数只在主进程被调用
        # if self.epoch >= self.save_epoch_limit:
        #     # save_trainer(self, os.path.join(self.ckpt_path, 'Final_Trainer.pkl'))
        #     save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'Final.pth'))
        
        # torch.cuda.empty_cache()
        self.log.write(f"Best Score : {self.best_score}")
        self.log.write(f"Best Metrics : {self.best_metrics}")
        self.epoch += 1

    def save_best_model(self, metrics_dict):
        score = get_score(metrics_dict)

        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics_dict
            save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'valid_Best.pth'))

    def on_loader_exit(self, mode, loss, outs, true):
        loss_dict, metrics_dict = {}, {}
        
        for k, v in loss.items():
            loss_dict[f"loss_{k}_{mode}"] = np.mean(v)

        metrics = self.get_metrics(outs, true)
        for m, a in metrics.items():
            metrics_dict[f"{m}_{mode}"] = a
        
        if mode == 'valid':
            key = f"loss_total_loss_{mode}"
            if key in loss_dict:
                self.loss_history.append(float(loss_dict[key]))
            else:
                # 如果模型输出的损失字典里没有 'total_loss'，打印一个警告
                print(f"Warning: 'total_loss' not found for mode '{mode}' for early stopping.")
        
        # 日志记录 (主进程执行)
        # if swanlab:
        #     swanlab.log(loss_dict, step=self.epoch)
        #     swanlab.log(metrics_dict, step=self.epoch)
        self.log.write(f"{mode} epoch_{self.epoch} : {loss_dict}")
        self.log.write(f'metrics : ' + str(metrics_dict))
        print(f"{mode} epoch_{self.epoch} : {loss_dict}", flush=True)
        print('metrics : ' + str(metrics_dict), flush=True)
            
        # 保存最佳模型仍然只基于验证集
        if mode in ['valid']:
            if self.epoch >= self.save_epoch_limit:
                self.save_best_model(metrics_dict)
            else:
                print(f"Current epoch is {self.epoch}, only save best when greater than {self.save_epoch_limit}")
    def get_metrics(self, logits: List[Any], labels: List[Any]):

        # print 5: examples
        # for 

        return self.model.task_head.METRICS_FN(logits, labels)  # 调用模型中的 METRICS_FN