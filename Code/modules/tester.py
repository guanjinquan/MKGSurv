import os
import sys
# 假设父目录已在 sys.path 中，或者根据您的项目结构调整
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tqdm
import numpy as np
import torch
import json

# 假设这些模块路径都正确
from datasets import GetDataLoader
from modules.model import GetModel
from modules.training_utils import Logger, load_model
from modules.training_utils.kaplan_meier_plotter import plot_risk_stratified_km, convert_logits_to_risk_scores

class Tester:
    """
    Handles the model testing process: loading a trained model, running inference on the test set,
    and calculating performance metrics.
    """
    def __init__(self, args):
        """
        Initializes the Tester object.

        Args:
            args (argparse.Namespace): Command-line arguments specifying the configuration.
        """
        assert args is not None, 'Please provide arguments for testing!'
        self.args = args

        # --- Device Configuration ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device, flush=True)
        
        # --- Data Loader ---
        _, _, self.test_loader = GetDataLoader(self.args)
        assert self.test_loader is not None, "Test loader could not be created."

        # --- Model Initialization and Loading ---
        modalities = self.test_loader.dataset.modalities
        self.model = GetModel(self.args, modalities_of_dataset=modalities).to(self.device)

        if self.args.load_pth_path is None:
            run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type]
            self.args.load_pth_path = os.path.join(self.args.ckpt_path, *run_path, "valid_Best.pth") 
        else:
            # Load the trained model checkpoint
            assert os.path.exists(self.args.load_pth_path), f"Checkpoint not found at {self.args.load_pth_path}"
        
        print(f"Loading model from {self.args.load_pth_path}...", flush=True)
        
        # Load state dict and handle 'module.' prefix if trained with DDP
        checkpoint = load_model(self.args.load_pth_path)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        
        print("Model loaded successfully.", flush=True)

        # --- Logging Setup ---
        # The log file will be saved in the same directory as the model checkpoint.
        self.log_path = os.path.dirname(self.args.load_pth_path)
        self.log = Logger(os.path.join(self.log_path, 'test_log.txt'))
        self.log.write("Test settings: " + str(args))

    def test(self):
        """
        Executes the main testing loop.
        """
        self.model.eval()
        
        # Lists to store results from all batches
        # (*** all_original_labels 是新增的 ***)
        all_pids, all_logits, all_labels, all_original_labels, all_losses = [], [], [], [], {}

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(self.test_loader), desc="Testing")
            for batch_data in self.test_loader:
                # Forward pass
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                # Collect logits and labels
                all_pids.extend(batch_data['pid'])
                # .tolist() 确保数据是可 JSON 序列化的
                all_logits.extend(out['logits'].detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels'])
                
                # (*** 新增 ***) 收集真实的、未离散化的标签用于KM图
                # 我们假设 dataloader 返回一个 'original_labels' 键
                if 'original_labels' in batch_data:
                    all_original_labels.extend(batch_data['original_labels'])
                
                # Collect loss values
                for key, value in out['losses'].items():
                    all_losses.setdefault(key, []).append(value.item())
                
                pbar.update(1)
            pbar.close()
        
        # (*** 新增 ***) 检查我们是否收集到了KM图所需的数据
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            if not all_original_labels:
                print("\n警告: 请求绘制KM图, 但 'original_labels' 未从 Dataloader 中收集到。")
                print("请确保 Dataloader 返回 'original_labels' 键。跳过绘图。")
                self.log.write("\n警告: 'draw_kaplan_meier=True' 但 'original_labels' 未找到。跳过绘图。")
                # 将其关闭，这样 log_results 就不会再次尝试
                self.args.draw_kaplan_meier = False

        # --- Metric Calculation and Logging ---
        # (*** 修改 ***) 传入 all_original_labels
        self.log_results(all_pids, all_losses, all_logits, all_labels, all_original_labels)

    def log_results(self, pids, losses, logits, labels, original_labels):
        """
        Calculates, logs, and prints the final evaluation metrics and losses.
        
        Args:
            pids (list): A list of patient/sample IDs.
            losses (dict): A dictionary of lists containing loss values for each batch.
            logits (list): A list of model outputs (logits).
            labels (list): A list of ground truth discrete labels (用于计算 C-Index 等).
            original_labels (list): A list of ground truth original continuous labels (用于绘制 KM 图).
        """
        loss_dict, metrics_dict = {}, {}
        
        # Calculate average loss for each loss component
        for key, value_list in losses.items():
            loss_dict[f"loss_{key}_test"] = np.mean(value_list)

        # Calculate performance metrics using the DISCRETE labels
        metrics = self.model.task_head.METRICS_FN(logits, labels)

        for metric_name, metric_value in metrics.items():
            metrics_dict[f"{metric_name}_test"] = metric_value
            
        # Print results to console
        print("\n--- Test Results ---", flush=True)
        print("Losses:", flush=True)
        print(json.dumps(loss_dict, indent=4), flush=True)
        print("\nMetrics:", flush=True)
        print(json.dumps(metrics_dict, indent=4), flush=True)
        print("--------------------", flush=True)

        # Log results to file
        self.log.write("\n--- Test Results ---")
        self.log.write("Losses: " + str(loss_dict))
        self.log.write("Metrics: " + str(metrics_dict))

        # Save metrics to a JSON file for easy access
        with open(os.path.join(self.log_path, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        # --- (*** 修改：将 logits 和 labels 合并到一个文件 ***) ---
        print("\nSaving PID-to-Data (logits+labels) mapping...", flush=True)
        if original_labels: # 确保我们收集到了
            try:
                pid_to_data = {}
                # 遍历所有 pids，创建一个新字典
                for i in range(len(pids)):
                    pid_to_data[pids[i]] = {
                        "logits": logits[i],       # logits 已经是 .tolist()
                        "label": original_labels[i] # original_labels 是 dict
                    }
                
                # 2. 定义保存路径
                data_save_path = os.path.join(self.log_path, 'test_pid_to_data.json')

                # 3. 将字典保存为 JSON 文件
                with open(data_save_path, 'w') as f:
                    json.dump(pid_to_data, f, indent=4)
                    
                self.log.write(f"\nPID-to-Data mapping saved to: {data_save_path}")
                print(f"PID-to-Data mapping saved to: {data_save_path}", flush=True)

            except Exception as e:
                print(f"\nError saving PID-to-Data mapping: {e}", flush=True)
                self.log.write(f"\nError saving PID-to-Data mapping: {e}")
        else:
            print("Warning: 'original_labels' is empty. Skipping saving PID-to-Data mapping.", flush=True)
            self.log.write("\nWarning: 'original_labels' is empty. Skipping saving PID-to-Data mapping.")
        # --- (*** 修改结束, 已替换单独的 pid_to_logits 和 pid_to_labels ***) ---
            
        # --- (*** 新增 ***) 绘制 Kaplan-Meier 曲线 ---
        # 检查 args 中是否有 draw_kaplan_meier 标志，并且它是否为 True
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            print("\nGenerating Kaplan-Meier plot...", flush=True)
            try:
                # 1. 准备数据
                # 将 logits (list of lists) 转换为 risk scores (list)
                risk_scores = convert_logits_to_risk_scores(logits)
                
                # 2. 从 all_original_labels (list of dicts) 中
                #    提取 'label_Y' (duration) 和 'label_c' (censorship)
                #    你的 __getitem__ 中: 'label_Y': event_time, 'label_c': censorship
                #    你的 'censorship' = 1 (删失), 0 (事件)
                #    lifelines 需要 'event_observed' = 0 (删失), 1 (事件)
                
                durations = [orig_label['label_Y'] for orig_label in original_labels]
                censorship = [orig_label['label_c'] for orig_label in original_labels]
                
                # 转换为 lifelines 格式 (1=事件, 0=删失)
                events_observed = [1 - c for c in censorship] 

                # 3. 定义保存路径
                save_path = os.path.join(self.log_path, 'Kaplan_Meier_Plot_Test_Set.png')

                # 4. 调用绘图函数
                plot_risk_stratified_km(
                    risk_scores,
                    durations,
                    events_observed,
                    save_path
                )
                self.log.write(f"\nKaplan-Meier plot saved to: {save_path}")
                print(f"Kaplan-Meier plot saved to: {save_path}", flush=True)

            except Exception as e:
                print(f"\nError generating Kaplan-Meier plot: {e}", flush=True)
                self.log.write(f"\nError generating Kaplan-Meier plot: {e}")


