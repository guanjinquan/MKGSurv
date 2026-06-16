import os
import sys
import time
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import GetDataLoader
from modules.model import GetModel
from modules.general_utils.optims import GetOptimizer, GetScheduler
from modules.general_utils import Logger, save_trainer, save_model, load_model, load_trainer
from typing import Dict, List, Any
import random
from collections import defaultdict
import json


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

        # 固定种子
        seed = int(args.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
    
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
        self.save_epoch_limit = int(self.args.num_epochs // 5)
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

        self.metric_history = []
        self.patience = int(self.args.num_epochs * 0.8)
        self.monitor_length = 10
        self.best_metrics = {}
        self.best_score = 0
        
        # --- 只有主进程 (rank 0) 才进行日志记录和文件保存 ---
        if self.local_rank == 0:
            if self.args.fold is not None:
                run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type, f"Fold{self.args.fold}"]
            else:
                run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type]
 
            self.log_path = os.path.join(self.args.log_path, *run_path)
            self.ckpt_path = os.path.join(self.args.ckpt_path, *run_path) 
            print("log_path : ", self.log_path, flush=True)
            print("ckpt_path : ", self.ckpt_path, flush=True)
            
            if os.path.exists(os.path.join(self.ckpt_path, 'valid_Best.pth')):
                raise FileExistsError("Trainer already exists!!!")
            
            os.makedirs(self.log_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.log = Logger(os.path.join(self.log_path, 'log.txt'))
            self.log.write("settings : " + str(args))

    def early_stop(self):
        epoch_run = len(self.metric_history)
        early_stop_flag = torch.zeros(1).to(self.device)
        
        if self.local_rank == 0 and epoch_run > self.patience and epoch_run > self.monitor_length:
            if self.metric_history[-1] <= np.mean(self.metric_history[-self.monitor_length:]):
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
        # 计时 (仅主进程需要)
        start_time = time.time() if self.local_rank == 0 else None

        self.model.train()
        if self.args.use_ddp:
            train_loader.sampler.set_epoch(self.epoch)
        
        # 收集训练过程中的结果
        local_outs, local_true, local_loss = [], [], {}

        assert len(train_loader) > 0, f"train_loader is empty!!!"
        print(f"Training set length is {len(train_loader)}")

        self.optimizer.zero_grad()
        for i, batch_data in enumerate(train_loader, 1):
            batch_size = len(batch_data['pid'])

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
            
            # 收集当前批次的结果
            local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
            local_true.extend(batch_data['labels'])
            for k, v in out['losses'].items():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                        local_loss.setdefault(k, []).append(v.item())
                    else:
                        local_loss.setdefault(k, []).append(v.cpu().numpy().tolist())
                elif isinstance(v, float):
                    local_loss.setdefault(k, []).append(v)

        # 收集 DDP 结果到主进程
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

        # 只有主进程计算指标并记录时间
        if self.local_rank == 0:
            if self.args.use_ddp:
                final_outs = [item for sublist in gathered_outs for item in sublist]
                final_true = [item for sublist in gathered_true for item in sublist]
                final_loss = local_loss
            else:
                final_outs = local_outs
                final_true = local_true
                final_loss = local_loss
            
            # 原有的指标记录
            self.on_loader_exit('train', final_loss, final_outs, final_true)

            # 计算并输出训练时间
            elapsed = time.time() - start_time
            num_samples = len(final_outs)          # 实际处理的样本总数
            avg_time_ms = (elapsed / num_samples) * 1000
            time_info = (f"Train time: {elapsed:.2f}s, samples: {num_samples}, "
                         f"avg: {avg_time_ms:.2f}ms/sample")
            self.log.write(time_info)
            print(time_info, flush=True)

    def eval_epoch(self, val_loader, mode='valid'):
        # 计时 (仅主进程需要)
        start_time = time.time() if self.local_rank == 0 else None

        self.model.eval()
        local_outs, local_true, local_loss = [], [], {}
        other_info_dict = defaultdict(list)
        
        with torch.no_grad():
            for batch_data in val_loader:
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
                local_true.extend(batch_data['labels'])
                for k, v in out['losses'].items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                            local_loss.setdefault(k, []).append(v.item())
                        else:
                            local_loss.setdefault(k, []).append(v.cpu().numpy().tolist())
                    elif isinstance(v, float):
                        local_loss.setdefault(k, []).append(v)

                if 'other_info' in out:
                    for k, v in out['other_info'].items():
                        if isinstance(v, torch.Tensor):
                            if v.dim() == 0 or (v.dim() == 1 and v.shape[0] == 1):
                                other_info_dict[k].append(v.item())
                            else:
                                other_info_dict[k].append(v.cpu().numpy().tolist())
                        else:
                            other_info_dict[k].append(v)

        # DDP 汇总
        if self.args.use_ddp:
            # 汇总 Loss
            for k in local_loss:
                loss_tensor = torch.tensor(np.mean(local_loss[k]), device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    local_loss[k] = [loss_tensor.item()]

            # 汇总 other_info_dict
            for k in list(other_info_dict.keys()): # Use list to avoid runtime error if we modify dict
                # Try to mean the local values first
                try:
                     local_mean = np.mean(other_info_dict[k])
                except Exception:
                     # If it can't be meaned (e.g. string), we don't reduce it
                     continue
                
                info_tensor = torch.tensor(local_mean, device=self.device, dtype=torch.float32)
                dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    other_info_dict[k] = [info_tensor.item()] # Wrap in list to keep structure consistent

            gathered_outs = [None] * dist.get_world_size()
            gathered_true = [None] * dist.get_world_size()
            dist.gather_object(local_outs, gathered_outs if self.local_rank == 0 else None, dst=0)
            dist.gather_object(local_true, gathered_true if self.local_rank == 0 else None, dst=0)
        
        # 主进程处理
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

            # 计算并输出评估时间
            elapsed = time.time() - start_time
            num_samples = len(final_outs)
            avg_time_ms = (elapsed / num_samples) * 1000
            time_info = (f"{mode.capitalize()} time: {elapsed:.2f}s, samples: {num_samples}, "
                         f"avg: {avg_time_ms:.2f}ms/sample")
            self.log.write(time_info)
            print(time_info, flush=True)

            # 处理并打印 other_info_dict
            if dict(other_info_dict):
                averaged_other_info = {}
                for k, v in other_info_dict.items():
                    try:
                        if isinstance(v[0], list):
                            avg_v = [np.mean([vv[pos] for vv in v]) for pos in range(len(v[0]))]
                            averaged_other_info[k] = avg_v
                        else:
                            averaged_other_info[k] = np.mean(v)
                    except Exception as e:
                         # Fallback if the data cannot be easily averaged
                         averaged_other_info[k] = "Could not average data."
                         print(f"Warning: Could not average other_info key '{k}': {e}")
                
                print(f"\nOther Info ({mode}):", flush=True)
                print(json.dumps(averaged_other_info, indent=4), flush=True)
                self.log.write(f"Other Info ({mode}): " + str(averaged_other_info))

    def on_epoch_end(self):
        # 这个函数只在主进程被调用
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
            if isinstance(v[0], list) and len(v[0]) > 1:
                v = [np.mean([vv[pos] for vv in v]) for pos in range(len(v[0]))]
                loss_dict[f"loss_{k}_{mode}"] = v
            else:   
                loss_dict[f"loss_{k}_{mode}"] = np.mean(v)

        metrics = self.get_metrics(outs, true)
        for m, a in metrics.items():
            metrics_dict[f"{m}_{mode}"] = a
        
        if mode == 'valid':
            key = f"C-Index_{mode}"
            self.metric_history.append(float(metrics_dict[key]))

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
            if self.epoch > self.save_epoch_limit:
                self.save_best_model(metrics_dict)
            else:
                print(f"Current epoch is {self.epoch}, only save best when greater than {self.save_epoch_limit}")

    def get_metrics(self, logits: List[Any], labels: List[Any]):
        return self.model.task_head.METRICS_FN(logits, labels)