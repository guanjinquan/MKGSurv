import os
import sys
import tqdm
import numpy as np
import torch
import json
import copy  
from collections import defaultdict


from datasets import GetDataLoader
from modules.model import GetModel
from modules.general_utils import Logger, load_model
from modules.general_utils.kaplan_meier_plotter import plot_risk_stratified_km, convert_logits_to_risk_scores


class Tester:
    """
    Handles the model testing process: loading a trained model, running inference on the test set,
    and calculating performance metrics.
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

        # --- Model Initialization ---
        dataset = self.test_loader.dataset
        self.modalities = dataset.get_active_modalities()
        self.model = GetModel(self.args, dataset).to(self.device)
        
        if self.args.load_pth_path is None:
            if self.args.fold is not None:
                run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type, f"Fold{self.args.fold}"]
            else:
                run_path = [self.args.model_task, self.args.runs_id + "+" + self.args.fusion_type]
 
            print("Run path:", run_path)
            load_pth_path = os.path.join(self.args.ckpt_path, *run_path, "valid_Best.pth") 
        else:
            assert os.path.exists(self.args.load_pth_path), f"Checkpoint not found at {self.args.load_pth_path}"
            load_pth_path = self.args.load_pth_path

        print(f"Loading model from {load_pth_path}...", flush=True)
        
        checkpoint = load_model(load_pth_path)
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        
        print("Model loaded successfully.", flush=True)

        self.log_path = os.path.dirname(load_pth_path)
        self.log = Logger(os.path.join(self.log_path, 'test_log.txt'))
        self.log.write("Test settings: " + str(args))

    def valid(self):
        if self.valid_loader:
            metrics = self._test_(self.valid_loader, title="Validation Set", save_results=False)
        return metrics

    def test(self):
        metrics = self._test_(self.test_loader, title="Test Set")
        return metrics
    
    def _test_(self, dataloader, title, save_results=True):
        self.model.eval()
        
        all_pids, all_logits, all_labels, all_labels = [], [], [], []
        all_losses = {}

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(dataloader), desc=title)
            for batch_data in dataloader:
                
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                all_pids.extend(batch_data['pid'])
                all_logits.extend(out['logits'].detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels']) 
                
                # for key, value in out['losses'].items():
                #     all_losses.setdefault(key, []).append(value.item())
                for k, v in out['losses'].items():
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 0:
                            all_losses.setdefault(k, []).append(v.item())
                        else:
                            all_losses.setdefault(k, []).append(v.cpu().numpy().tolist())
                    elif isinstance(v, float):
                        all_losses.setdefault(k, []).append(v)

                pbar.update(1)
            pbar.close()
        
        # Check for KM plot
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            if not all_labels:
                print("\nWarning: Requested KM plot, but 'labels' not collected.")
                self.args.draw_kaplan_meier = False

        # --- Metric Calculation ---
        return self.log_results(
            all_pids, 
            all_losses, 
            all_logits, 
            all_labels, 
            title=title,
            save_results=save_results
        )

    def log_results(self, 
        pids, 
        losses, 
        logits, 
        labels, 
        title, 
        save_results=False
    ):
        
        loss_dict, metrics_dict = {}, {}
        ret_metrics_dict = {}
        
        # --- 1. Average Losses ---
        for key, value_list in losses.items():
            if isinstance(value_list[0], list):
                v = [np.mean([vv[pos] for vv in value_list]) for pos in range(len(value_list[0]))]
                loss_dict[f"loss_{key}_{title}"] = v
            else:   
                loss_dict[f"loss_{key}_{title}"] = np.mean(value_list)

        # --- 2. Survival Metrics ---
        metrics = self.model.task_head.METRICS_FN(logits, labels)
        for metric_name, metric_value in metrics.items():
            metrics_dict[f"{metric_name}_{title}"] = metric_value

        ret_metrics_dict.update(metrics_dict)
            
        print(f"\n--- {title} Results (Survival) ---", flush=True)
        print("Losses:", flush=True)
        print(json.dumps(loss_dict, indent=4), flush=True)
        print("\nMetrics:", flush=True)
        print(json.dumps(metrics_dict, indent=4), flush=True)
        print("--------------------------------", flush=True)

        self.log.write("\n--- Test Results (Survival) ---")
        self.log.write("Losses: " + str(loss_dict))
        self.log.write("Metrics: " + str(metrics_dict))

        if save_results:
            with open(os.path.join(self.log_path, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_dict, f, indent=4)

        # --- 3. Save PID-to-Data ---
        if labels:
            try:
                pid_to_data = {}
                for i in range(len(pids)):
                    pid_to_data[pids[i]] = {
                        "logits": logits[i],
                        "label": labels[i]
                    }
                
                data_save_path = os.path.join(self.log_path, 'test_pid_to_data.json')
                if save_results:
                    with open(data_save_path, 'w') as f:
                        json.dump(pid_to_data, f, indent=4)
            except Exception as e:
                print(f"Error saving PID map: {e}")

        # --- 5. Kaplan-Meier Plot ---
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            print("\nGenerating Kaplan-Meier plot...", flush=True)
            try:
                risk_scores = convert_logits_to_risk_scores(logits)
                durations = [l['label_time'] for l in labels]
                events_observed = [l['label_event'] for l in labels] 
                save_path = os.path.join(self.log_path, 'Kaplan_Meier_Plot_Test_Set.png')

                plot_risk_stratified_km(risk_scores, durations, events_observed, save_path)
                print(f"Kaplan-Meier plot saved to: {save_path}", flush=True)
            except Exception as e:
                print(f"\nError generating KM plot: {e}", flush=True)

        print("\n--- Testing Complete ---", flush=True)
        return ret_metrics_dict