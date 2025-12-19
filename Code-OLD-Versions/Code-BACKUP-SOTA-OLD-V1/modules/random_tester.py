import os
import sys
import tqdm
import numpy as np
import torch
import json
import copy 
import random # 新增 import
from collections import defaultdict

# 假设这些模块路径都正确
from datasets import GetDataLoader
from modules.model import GetModel



class RandomTester:
    """
    Random Baseline Tester:
    Generates random logits (log_h) and random treatment recommendations 
    to establish a lower-bound performance baseline.
    """
    def __init__(self, args):
        assert args is not None, 'Please provide arguments for testing!'
        self.args = args

        # --- Device Configuration ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device, flush=True)
        
        # --- Data Loader ---
        _, valid_loader, test_loader = GetDataLoader(self.args)
        self.valid_loader = valid_loader
        self.test_loader = test_loader or valid_loader
        assert self.test_loader is not None, "Test loader could not be created."

        # --- Model Initialization (仅用于获取配置和 METRICS_FN) ---
        dataset = self.test_loader.dataset
        
        # 我们初始化模型结构是为了访问 task_head 中定义的 METRICS_FN
        # 但我们不会加载任何权重
        print("[INFO] Initializing model structure to access metric functions (Weights are NOT loaded)...")
        self.model = GetModel(self.args, dataset).to(self.device)
        self.model.eval()
        
    def test(self):
        metrics = self._test_(self.test_loader, title="Random Baseline Test Set")
        return metrics
    
    def _test_(self, dataloader, title, save_results=True):
        """
        Executes the random testing loop.
        """
        
        # Lists to store results from all batches
        all_pids, all_logits, all_labels, all_original_labels = [], [], [], []
        all_losses = {}
        all_predicted_treatments = []

        print(f"[INFO] Starting {title} with RANDOM generation...", flush=True)

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(dataloader), desc=title)
            for batch_data in dataloader:
                
                batch_size = len(batch_data['pid'])
                
                # --- 1. Random Logits (log_h) Generation ---
                # 假设 log_h 是标量 (batch_size, 1)，使用标准正态分布模拟
                # 如果你的任务是多分类或离散时间 (DeepHit)，可能需要调整为 (batch_size, num_bins)
                random_logits = torch.randn(batch_size, 1).to(self.device)
                
                # 模拟一个随机的 loss，防止 log_results 报错
                fake_loss = torch.tensor(random.random()).to(self.device)
                
                # Collect logits and labels
                all_pids.extend(batch_data['pid'])
                all_logits.extend(random_logits.detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels']) 
                
                if 'labels' in batch_data:
                    all_original_labels.extend(batch_data['labels'])
                
                # Collect fake loss
                all_losses.setdefault('random_loss', []).append(fake_loss.item())
                

                pbar.update(1)
            pbar.close()
        
        # KM Plot Logic (保持不变)
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            if not all_original_labels:
                self.args.draw_kaplan_meier = False

        # --- Metric Calculation ---
        # 这里会使用 self.model.task_head.METRICS_FN 来计算指标
        # 因为 logits 是随机的，结果应该接近 C-Index 0.5
        return self.log_results(
            all_pids, 
            all_losses, 
            all_logits, 
            all_labels,                  
            all_original_labels,        
            all_predicted_treatments,    
            title=title,
            save_results=save_results
        )

    def log_results(self, pids, losses, logits, labels, original_labels, predicted_treatments, title, save_results=False):
        """
        Calculates, logs, and prints the final evaluation metrics.
        Same as original Tester but tailored for Random results context.
        """
        loss_dict, metrics_dict = {}, {}
        ret_metrics_dict = {}
        
        # 1. Average Losses (Fake)
        for key, value_list in losses.items():
            loss_dict[f"{key}_{title}"] = np.mean(value_list)

        # 2. Survival Metrics (C-Index should be ~0.5)
        # 即使模型未训练，我们仍然使用模型类中定义的静态方法或工具函数来计算指标
        metrics = self.model.task_head.METRICS_FN(logits, labels)
        for metric_name, metric_value in metrics.items():
            metrics_dict[f"{metric_name}_{title}"] = metric_value

        ret_metrics_dict.update(metrics_dict)
            
        print(f"\n--- {title} Results (Survival) ---", flush=True)
        print("Losses (Random):", json.dumps(loss_dict, indent=4), flush=True)
        print("Metrics (Random):", json.dumps(metrics_dict, indent=4), flush=True)

        return ret_metrics_dict