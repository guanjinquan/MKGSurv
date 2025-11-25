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
from modules.base_modules.treatment_pred_task_utils import get_treatment_risk, get_best_treamtent
from modules.general_utils.metrics import multiple_classification_metrics, recall_top_k



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
        
        self.test_treatment = False
        if "text-treatment" in self.modalities and len(self.modalities) > 2:
            self.test_treatment = True
            self.pre_op_modalities = self.test_loader.dataset.PRE_OP_MODALITIES
            self.post_op_modalities = self.test_loader.dataset.POST_OP_MODALITIES
            self.treatment_options = self.test_loader.dataset.TREATMENT_OPTIONS
            self.treatment_options_onehot = self.test_loader.dataset.TREATMENT_OPTIONS_ONEHOT
            self.treatment_options_embeds = self.test_loader.dataset.TREATMENT_OPTIONS_FEAT

            print("Treatment Length : ", len(self.treatment_options))
            print("[INFO] Start testing the treatment prediction accuracy.")
        else:
            print("[INFO] Don't evaluate the treatment prediction accuracy.")

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
        
        # Treatment Evaluation Lists
        all_predicted_treatments = []
        all_predicted_treatments_onehot = []
        all_treatment_risks = []      # Store raw risks for Recall@K
        all_treatment_gt_indices = [] # Store int ground truth for Recall@K

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(dataloader), desc=title)
            for batch_data in dataloader:
                
                # --- 1. Standard Survival Forward Pass ---
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                all_pids.extend(batch_data['pid'])
                all_logits.extend(out['logits'].detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels']) 
                
                for key, value in out['losses'].items():
                    all_losses.setdefault(key, []).append(value.item())
                
                # --- 2. Find Best Treatment ---
                if self.test_treatment:
                    batch_data_copy = copy.deepcopy(batch_data)
                    
                    # 1. 计算每个治疗方案的 Risk (Batch_size, Num_Treatments)
                    batch_risks = get_treatment_risk(
                        self.model,
                        batch_data_copy,
                        self.pre_op_modalities,
                        self.post_op_modalities,
                        self.treatment_options_embeds
                    )
                    
                    # 2. 根据 Risk 获取最佳治疗文本和 One-Hot
                    batch_best_treat, batch_best_treat_onehot = get_best_treamtent(
                        self.treatment_options,
                        self.treatment_options_onehot,
                        batch_risks
                    )
                    
                    # 3. 收集数据
                    all_predicted_treatments.extend(batch_best_treat)
                    all_predicted_treatments_onehot.extend(batch_best_treat_onehot)
                    all_treatment_risks.extend(batch_risks)
                    
                    # 4. 获取 Ground Truth Index (用于 Recall@K)
                    gt_indices = [self.treatment_options.index(label['treatment_type']) for label in batch_data['labels']]
                    all_treatment_gt_indices.extend(gt_indices)

                    for gt_id in gt_indices:
                        assert 0 <= gt_id < len(self.treatment_options), f"Invalid Ground Truth Index: {gt_id}"

                    # ---【DEBUG START】检查数据一致性 ---
                    for i, label in enumerate(batch_data['labels']):
                        # 1. 获取 Recall@1 认为的真值向量 (从你的 Options 列表里查)
                        gt_id = gt_indices[i]
                        vec_from_options = self.treatment_options_onehot[gt_id]
                        
                        # 2. 获取 Accuracy 认为的真值向量 (从 Label 字典里直接拿)
                        vec_from_label = label['treatment_type_onehot']
                        
                        # 确保转为 list 比较
                        if isinstance(vec_from_label, torch.Tensor):
                            vec_from_label = vec_from_label.cpu().numpy().tolist()
                        if isinstance(vec_from_options, np.ndarray):
                            vec_from_options = vec_from_options.tolist()
                            
                        # 3. 比较
                        if vec_from_options != vec_from_label:
                            print(f"\n[CRITICAL MISMATCH FOUND] PID: {batch_data['pid'][i]}")
                            print(f"Treatment Name: {label['treatment_type']}")
                            print(f"Option Index  : {gt_id}")
                            print(f"Vector in Options List: {vec_from_options}")
                            print(f"Vector in Label Dict  : {vec_from_label}")
                            
                            # 找出不一致的位
                            diff = [k for k in range(len(vec_from_options)) if vec_from_options[k] != vec_from_label[k]]
                            print(f"Mismatch at indices: {diff}")

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
            all_predicted_treatments, 
            all_predicted_treatments_onehot, 
            all_treatment_risks,      # New
            all_treatment_gt_indices, # New
            title=title,
            save_results=save_results
        )

    def log_results(self, 
        pids, 
        losses, 
        logits, 
        labels, 
        predicted_treatments, 
        predicted_treatments_onehot, 
        treatment_risks, 
        treatment_gt_indices,
        title, 
        save_results=False
    ):
        
        loss_dict, metrics_dict = {}, {}
        ret_metrics_dict = {}
        
        # --- 1. Average Losses ---
        for key, value_list in losses.items():
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
                    if isinstance(labels[i]['treatment_type_onehot'], torch.Tensor):
                        labels[i]['treatment_type_onehot'] = labels[i]['treatment_type_onehot'].cpu().numpy().tolist()
                    
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

        # --- 4. Treatment Metrics ---
        if self.test_treatment:
            print("\n--- Test Results (Treatment Recommendation) ---", flush=True)
            self.log.write("\n--- Test Results (Treatment Recommendation) ---")

            if not predicted_treatments or not labels:
                print("Error: Missing predictions or labels for treatment.", flush=True)
            else:
                try:
                    # A. Multi-label metrics (Precision, F1, etc.)
                    treatment_metrics = multiple_classification_metrics(
                        predicted_treatments_onehot, 
                        labels,
                        with_sigmoid=False
                    )
                    
                    # B. Recall@K Metrics (using Risks and Int Indices)
                    if treatment_risks and treatment_gt_indices:
                        print("Computing Recall@K...")

                        r1 = recall_top_k(treatment_risks, treatment_gt_indices, k=1)
                        r3 = recall_top_k(treatment_risks, treatment_gt_indices, k=3)
                        r5 = recall_top_k(treatment_risks, treatment_gt_indices, k=5)
                        
                        treatment_metrics["Recall@1"] = r1
                        treatment_metrics["Recall@3"] = r3
                        treatment_metrics["Recall@5"] = r5
                    
                    ret_metrics_dict.update(treatment_metrics)
                    
                    self.log.write("Metrics: " + str(treatment_metrics))
                    print("Metrics:", flush=True)
                    print(json.dumps(treatment_metrics, indent=4), flush=True)

                    # Top 5 Predicted Counts
                    treatment_counter = defaultdict(int)
                    for pred_str in predicted_treatments:
                        treatment_counter[pred_str] += 1
                    
                    top_5 = sorted(treatment_counter.items(), key=lambda item: item[1], reverse=True)[:5]
                    print("\nTop 5 Predicted Treatments (Counts):", flush=True)
                    print(json.dumps(dict(top_5), indent=4), flush=True)
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"\nError calculating treatment accuracy: {e}", flush=True)

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