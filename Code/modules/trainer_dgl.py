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
# --- DGL: 确保从正确的 model 文件
from modules.model_dgl import GetModel 
from modules.training_utils.optims import GetOptimizer, GetScheduler
from modules.training_utils import Logger, save_trainer, save_model, load_model, load_trainer
from typing import Dict, List, Any

try:
    import swanlab
except ImportError:
    print("swanlab not installed, logging to swanlab will be disabled.")
    swanlab = None


def get_score(metrics):
    if "AUC_valid" in metrics:
        return metrics.get(f"AUC_valid", 0)
    
    elif "C-Index_valid" in metrics:
        return metrics.get(f"C-Index_valid", 0)

    else:
        # --- DGL: 增加一个默认返回值，以防验证集没有目标指标 ---
        # 或者你可以选择一个你任务中总会存在的指标
        auc_keys = [k for k in metrics if 'AUC' in k and 'valid' in k]
        if auc_keys:
            return metrics.get(auc_keys[0], 0)
        
        c_index_keys = [k for k in metrics if 'C-Index' in k and 'valid' in k]
        if c_index_keys:
            return metrics.get(c_index_keys[0], 0)
            
        print(f"Warning: No target metric (AUC_valid or C-Index_valid) found in {metrics}. Defaulting to 0.")
        return 0


class Trainer:
    def __init__(self, args):
        assert args is not None, 'Please input args!!!'
        self.args = args
        
        # --- DGL: DGL 训练策略需要一个 alpha 值 ---
        assert hasattr(args, 'alpha'), "DGL Trainer 需要 args.alpha (用于 loss_unimodal)"
        self.alpha = args.alpha

        # --- DDP ADDITION: 初始化分布式环境 ---
        self.local_rank = 0
        if self.args.use_ddp:
            # local_rank 通常由 torchrun 或 torch.distributed.launch 自动设置
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        
        self.device = torch.device("cuda", self.local_rank)
        print("device ", self.device)
        
        # --- DDP CHANGE: 将模型移动到正确的设备 ---
        self.model = GetModel(self.args).to(self.device)
        
        # --- DDP ADDITION: 使用 DDP 包装模型 ---
        if self.args.use_ddp:
            # --- DGL: find_unused_parameters=True 很重要 ---
            # 因为 DGL 的反向传播是分两步的，有些参数可能在一步中未被使用
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        self.optimizer = GetOptimizer(self.args, self.model)
        self.scheduler = GetScheduler(self.args, self.optimizer)
        
        # --- DDP CHANGE: 数据加载器需要 DistributedSampler ---
        self.train_loader, self.val_loader, self.test_loader = GetDataLoader(self.args)
        
        self.epoch = 0
        self.iters = 0
        self.acc_step = self.args.acc_step
            
        if self.args.continue_training:
            # BUG FIX: 加载模型和训练器状态时，确保只在主进程打印信息
            if self.local_rank == 0:
                print(f"Continue training from {self.args.load_pth_path} !!!", flush=True)
            
            # 需要确保所有进程都加载了相同的状态
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            # --- DGL: BUG FIX, 确保 map_location 生效 ---
            ckp_trainer = load_trainer(self.args.load_pth_path, map_location=map_location)
            self.epoch = ckp_trainer.epoch
            self.optimizer.load_state_dict(ckp_trainer.optimizer.state_dict())
            self.scheduler.load_state_dict(ckp_trainer.scheduler.state_dict())
            
            # BUG FIX: 像 finetune 一样处理 'module.' 前缀，以兼容DDP和单卡模型
            model_state_dict = ckp_trainer.model.state_dict()
            model_to_load = self.model.module if self.args.use_ddp else self.model
            
            cleaned_state_dict = model_state_dict
            model_to_load.load_state_dict(cleaned_state_dict)
            del ckp_trainer
            
        elif self.args.finetune:
            if self.local_rank == 0:
                print(f"Fine-tune from {self.args.load_pth_path}!!!", flush=True)
            # --- DGL: BUG FIX, 确保 map_location 生效 ---
            map_location = {'cuda:0': f'cuda:{self.local_rank}'}
            cp = load_model(self.args.load_pth_path, map_location=map_location)
            pretrain = cp['model']
            
            model_to_load = self.model.module if self.args.use_ddp else self.model
            pretrain_filtered = {k: v for k, v in pretrain.items() if k in model_to_load.state_dict()}
            model_to_load.load_state_dict(pretrain_filtered, strict=False)
        
        # amp
        self.scaler = GradScaler() if self.args.use_amp else None

        self.loss_history = []
        self.patience = 150
        self.monitor_length = 20
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
                print(f"Warning: Trainer at {self.ckpt_path} already exists. May overwrite.", flush=True)
                # raise ValueError("Trainer already exists!!!") # 暂时注释掉，方便调试
            
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
                if self.local_rank == 0 and swanlab:
                    for idx, groups in enumerate(self.optimizer.param_groups):
                        swanlab.log({f"lr_{['backbones', 'others'][idx]}": groups['lr']}, step=self.epoch)

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
        
        # --- DGL: 初始化列表，用于收集训练过程中的结果 ---
        local_outs, local_true, local_loss = [], [], {}

        pbar = tqdm.tqdm(total=len(train_loader), disable=(self.local_rank != 0))
        
        assert len(train_loader) > 0, f"train_loader is empty!!!"
        if self.local_rank == 0:
            print(f"Training set length is {len(train_loader)}")

        # --- DGL: 梯度清零移到 accumulation step 内部 ---
        self.optimizer.zero_grad()
        
        for i, batch_data in enumerate(train_loader, 1):
            batch_size = len(batch_data['pid'])

            # --- DGL: 核心修改：实现解耦反向传播 ---
            
            use_amp = self.args.use_amp
            
            # 1. Forward Pass
            with autocast(enabled=use_amp):
                out = self.model(batch_size, batch_data)
                # 从模型获取分离的损失
                loss_a = out['losses']['loss_a']
                loss_v = out['losses']['loss_v']
                loss_f = out['losses']['loss_f']
                
                # 计算单模态总损失
                loss_unimodal = (loss_a + loss_v) * self.alpha
                
                # 为梯度累积进行缩放
                loss_unimodal_acc = loss_unimodal / self.acc_step
                loss_f_acc = loss_f / self.acc_step
            
            # 2. Backward for Unimodal Loss (更新 Encoders)
            if use_amp:
                self.scaler.scale(loss_unimodal_acc).backward(retain_graph=True)
            else:
                loss_unimodal_acc.backward(retain_graph=True)

            # 3. Zero Gradients for Fusion/Decoder parts
            # 关键步骤：阻止 unimodal_loss 的梯度流向 fusion 和 decoder
            model_to_access = self.model.module if self.args.use_ddp else self.model
            
            # 定义需要清空梯度的参数组
            param_groups_to_zero = [
                model_to_access.fusion_module.parameters(),
                model_to_access.task_head.decoder.parameters(), # 假设 decoder 在 task_head 中
                model_to_access.unimodal_classifier_a.parameters(),
                model_to_access.unimodal_classifier_v.parameters()
            ]
            
            for group in param_groups_to_zero:
                for p in group:
                    if p.grad is not None:
                        p.grad = None # 直接设置为 None

            # 4. Backward for Fusion Loss (更新 Encoders + Fusion + Decoder)
            if use_amp:
                self.scaler.scale(loss_f_acc).backward()
            else:
                loss_f_acc.backward()

            # --- DGL: 结束核心修改 ---

            # 5. Optimizer Step (处理梯度累积)
            if i % self.acc_step == 0 or i == len(train_loader):
                if use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 清空梯度，为下一次累积做准备
                self.optimizer.zero_grad()
            
            # --- DGL: 收集当前批次的结果 (原始损失，非 acc 缩放) ---
            local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
            local_true.extend(self.model.module.task_head.get_labels(batch_data)) # 确保获取正确标签
            
            local_loss.setdefault('loss_a', []).append(loss_a.item())
            local_loss.setdefault('loss_v', []).append(loss_v.item())
            local_loss.setdefault('loss_f', []).append(loss_f.item())
            # 记录用于 early stopping 和比较的 "total_loss"
            total_loss_item = loss_unimodal.item() + loss_f.item()
            local_loss.setdefault('total_loss', []).append(total_loss_item)

            if i % 100 == 0 and self.local_rank == 0:
                 print(f" ->>>> Train step-{i}: ", flush=True)
                 print(f" ->>>> loss_a: {loss_a.item():.4f}, loss_v: {loss_v.item():.4f}, loss_f: {loss_f.item():.4f}", flush=True)
                 print(f" ->>>> total_loss (unimodal*alpha + f): {total_loss_item:.4f}", flush=True)

            pbar.update(1)

        pbar.close()

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
        
        model_to_access = self.model.module if self.args.use_ddp else self.model
        
        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(val_loader), disable=(self.local_rank != 0))
            for batch_data in val_loader:
                batch_size = len(batch_data['pid'])
                
                # --- DGL: eval forward pass ---
                out = self.model(batch_size, batch_data)
                
                local_outs.extend(out['logits'].detach().cpu().numpy().tolist())
                local_true.extend(model_to_access.task_head.get_labels(batch_data)) # 确保获取正确标签
                
                # --- DGL: 收集所有损失 ---
                loss_a = out['losses']['loss_a'].item()
                loss_v = out['losses']['loss_v'].item()
                loss_f = out['losses']['loss_f'].item()
                
                local_loss.setdefault('loss_a', []).append(loss_a)
                local_loss.setdefault('loss_v', []).append(loss_v)
                local_loss.setdefault('loss_f', []).append(loss_f)
                
                # 计算总损失
                total_loss_item = (loss_a + loss_v) * self.alpha + loss_f
                local_loss.setdefault('total_loss', []).append(total_loss_item)
                
                pbar.update(1)
            pbar.close()

        if self.args.use_ddp:
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
        # 这个函数只在主进程被调用
        save_trainer(self, os.path.join(self.ckpt_path, 'Final_Trainer.pkl'))
        save_model(self.model, self.epoch, os.path.join(self.ckpt_path, f'Final.pth'))
        torch.cuda.empty_cache()
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
        
        # --- DGL: loss 字典现在包含 loss_a, loss_v, loss_f, total_loss ---
        for k, v in loss.items():
            # 键将是 loss_a_valid, loss_v_valid, loss_f_valid, total_loss_valid
            loss_dict[f"{k}_{mode}"] = np.mean(v)

        model_to_access = self.model.module if self.args.use_ddp else self.model
        metrics = self.get_metrics(outs, true, model_to_access.task_head)
        for m, a in metrics.items():
            metrics_dict[f"{m}_{mode}"] = a
        
        if mode == 'valid':
            # --- DGL: 确保使用 'total_loss_valid' 进行早停 ---
            key = f"total_loss_{mode}"
            if key in loss_dict:
                self.loss_history.append(float(loss_dict[key]))
            else:
                # 如果模型输出的损失字典里没有 'total_loss'，打印一个警告
                print(f"Warning: 'total_loss' not found for mode '{mode}' for early stopping.")
        
        # 日志记录 (主进程执行)
        if swanlab:
            swanlab.log(loss_dict, step=self.epoch)
            swanlab.log(metrics_dict, step=self.epoch)
        self.log.write(f"{mode} epoch_{self.epoch} : {loss_dict}")
        self.log.write(f'metrics : ' + str(metrics_dict))
        print(f"{mode} epoch_{self.epoch} : {loss_dict}", flush=True)
        print('metrics : ' + str(metrics_dict), flush=True)
            
        # 保存最佳模型仍然只基于验证集
        if mode in ['valid']:
            self.save_best_model(metrics_dict)

    def get_metrics(self, logits: List[Any], labels: List[Any], task_head_ref):
        """
        DGL: 修改为接受 task_head 引用，以便调用正确的 METRICS_FN
        """
        # print("Type of logits:", type(logits)) # 调试时用
        return task_head_ref.METRICS_FN(logits, labels)
