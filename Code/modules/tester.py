import os
import sys
import tqdm
import numpy as np
import torch
import json
import copy  
from collections import defaultdict

# 假设这些模块路径都正确
from datasets import GetDataLoader
from modules.model import GetModel
from modules.training_utils import Logger, load_model
from modules.training_utils.kaplan_meier_plotter import plot_risk_stratified_km, convert_logits_to_risk_scores
from modules.common_modules.find_treatment_utils import find_best_treamtent


class Tester:
    """
    Handles the model testing process: loading a trained model, running inference on the test set,
    and calculating performance metrics.
    
    (NEW) Also handles finding the best treatment and calculating recommendation accuracy.
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
        _, valid_loader, test_loader = GetDataLoader(self.args)
        self.test_loader = test_loader or valid_loader
        assert self.test_loader is not None, "Test loader could not be created."

        # --- Model Initialization and Loading ---
        dataset = self.test_loader.dataset
        self.modalities = dataset.get_active_modalities()
        self.model = GetModel(self.args, dataset).to(self.device)
        
        self.test_treatment = False
        if "text-treatment" in self.modalities:
            self.test_treatment = True
            self.pre_op_modalities = self.test_loader.dataset.PRE_OP_MODALITIES
            self.post_op_modalities = self.test_loader.dataset.POST_OP_MODALITIES
            self.treatment_options = self.test_loader.dataset.TREATMENT_OPTIONS
            print("[INFO] Start testing the treatment prediction accuracy.")
        else:
            print("[INFO] Don't evaluate the treatment prediction accuracy.")

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
        all_pids, all_logits, all_labels, all_original_labels = [], [], [], []
        all_losses = {}
        
        # (NEW) List for treatment predictions
        all_predicted_treatments = []

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(self.test_loader), desc="Testing")
            for batch_data in self.test_loader:
                
                # --- 1. Standard Survival Forward Pass ---
                # This pass uses the *actual* post-op data
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                # Collect logits and labels for survival metrics
                all_pids.extend(batch_data['pid'])
                all_logits.extend(out['logits'].detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels']) # 'labels' from discretizer
                
                # Collect original, non-discretized labels for KM plot and treatment accuracy
                if 'labels' in batch_data: # This is the dict from __getitem__
                    all_original_labels.extend(batch_data['labels'])
                
                # Collect loss values
                for key, value in out['losses'].items():
                    all_losses.setdefault(key, []).append(value.item())
                
                # --- 2. (NEW) Find Best Treatment ---
                # This pass masks post-op data and iterates all treatments
                if self.test_treatment:
                    batch_data_copy = copy.deepcopy(batch_data)
                    
                    _, batch_best_treat = find_best_treamtent(
                        self.model,
                        batch_data_copy,
                        self.pre_op_modalities,
                        self.post_op_modalities,
                        self.treatment_options
                    )
                    all_predicted_treatments.extend(batch_best_treat)

                pbar.update(1)
            pbar.close()
        
        # Check if we collected data for KM plot
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            if not all_original_labels:
                print("\nWarning: Requested KM plot, but 'original_labels' (dict) not collected. Skipping plot.")
                self.log.write("\nWarning: 'draw_kaplan_meier=True' but 'original_labels' not found. Skipping plot.")
                self.args.draw_kaplan_meier = False

        # --- Metric Calculation and Logging ---
        self.log_results(
            all_pids, 
            all_losses, 
            all_logits, 
            all_labels,                  # Discrete labels for C-Index
            all_original_labels,         # Dict list for KM-plot and Treatment Acc
            all_predicted_treatments     # List of predicted treatment strings
        )

    def log_results(self, pids, losses, logits, labels, original_labels, predicted_treatments):
        """
        Calculates, logs, and prints the final evaluation metrics and losses.
        
        Args:
            pids (list): A list of patient/sample IDs.
            losses (dict): A dictionary of lists containing loss values for each batch.
            logits (list): A list of model outputs (logits).
            labels (list): A list of ground truth discrete labels (for C-Index).
            original_labels (list): A list of ground truth label dicts (for KM plot and Treatment Acc).
            predicted_treatments (list): A list of predicted best treatment strings.
        """
        loss_dict, metrics_dict = {}, {}
        
        # --- 1. Calculate Average Losses ---
        for key, value_list in losses.items():
            loss_dict[f"loss_{key}_test"] = np.mean(value_list)

        # --- 2. Calculate Survival Performance Metrics (using DISCRETE labels) ---
        metrics = self.model.task_head.METRICS_FN(logits, labels)
        for metric_name, metric_value in metrics.items():
            metrics_dict[f"{metric_name}_test"] = metric_value
            
        # Print Survival results to console
        print("\n--- Test Results (Survival) ---", flush=True)
        print("Losses:", flush=True)
        print(json.dumps(loss_dict, indent=4), flush=True)
        print("\nMetrics:", flush=True)
        print(json.dumps(metrics_dict, indent=4), flush=True)
        print("--------------------------------", flush=True)

        # Log Survival results to file
        self.log.write("\n--- Test Results (Survival) ---")
        self.log.write("Losses: " + str(loss_dict))
        self.log.write("Metrics: " + str(metrics_dict))

        # Save metrics to a JSON file
        with open(os.path.join(self.log_path, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        # --- 3. Save PID-to-Data (Logits + Original Labels) ---
        print("\nSaving PID-to-Data (logits+labels) mapping...", flush=True)
        if original_labels:
            try:
                pid_to_data = {}
                for i in range(len(pids)):
                    pid_to_data[pids[i]] = {
                        "logits": logits[i],
                        "label": original_labels[i] # This is the original dict
                    }
                
                data_save_path = os.path.join(self.log_path, 'test_pid_to_data.json')
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
            
        # --- 4. (NEW) Calculate Treatment Recommendation Accuracy ---
        if self.test_treatment:
            print("\n--- Test Results (Treatment Recommendation) ---", flush=True)
            self.log.write("\n--- Test Results (Treatment Recommendation) ---")

            if not predicted_treatments or not original_labels:
                print("Error: Cannot calculate treatment accuracy. Missing predictions or labels.", flush=True)
                self.log.write("Error: Cannot calculate treatment accuracy. Missing predictions or labels.")
            else:
                try:
                    # 1. Extract actual treatments
                    actual_treatments = [label_dict['treatment_type'] for label_dict in original_labels]

                    if len(actual_treatments) != len(predicted_treatments):
                        print(f"Error: Mismatch in lengths. Actual: {len(actual_treatments)}, Predicted: {len(predicted_treatments)}", flush=True)
                        self.log.write(f"Error: Mismatch in lengths. Actual: {len(actual_treatments)}, Predicted: {len(predicted_treatments)}")
                    else:
                        correct_predictions = 0
                        total_predictions = len(actual_treatments)

                        treatment_counter = defaultdict(int)

                        # 2. Compare actual and predicted
                        for i in range(total_predictions):
                            actual = actual_treatments[i]
                            predicted = predicted_treatments[i]
                            is_correct = (actual == predicted)
                            treatment_counter[predicted] += 1
                            
                            if is_correct:
                                correct_predictions += 1
                        
                        # 3. Calculate accuracy
                        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                        treatment_metrics = {'treatment_accuracy': f"{accuracy:.2f}%"}
                        
                        # 4. Log and print
                        self.log.write("Metrics: " + str(treatment_metrics))
                        print("Metrics:", flush=True)
                        print(json.dumps(treatment_metrics, indent=4), flush=True)

                        # Print top 5 treatment
                        sorted_by_count = sorted(treatment_counter.items(), key=lambda item: item[1], reverse=True)
                        top_5 = sorted_by_count[:5]
                        top_5_dict = dict(top_5)
                        print(json.dumps(top_5_dict, indent=4), flush=True)
                        
                except Exception as e:
                    print(f"\nError calculating treatment accuracy: {e}", flush=True)
                    self.log.write(f"\nError calculating treatment accuracy: {e}")

        # --- 5. (FIXED) Draw Kaplan-Meier Plot ---
        if hasattr(self.args, 'draw_kaplan_meier') and self.args.draw_kaplan_meier:
            print("\nGenerating Kaplan-Meier plot...", flush=True)
            try:
                # 1. Convert logits to risk scores
                risk_scores = convert_logits_to_risk_scores(logits)
                
                # 2. Extract durations and events from original_labels (list of dicts)
                # (FIXED) Using 'label_time' and 'label_event' from your dataset
                durations = [orig_label['label_time'] for orig_label in original_labels]
                
                # Your 'label_event' is 1 for event, 0 for censored.
                # lifelines 'events_observed' is 1 for event, 0 for censored.
                # They are already in the correct format.
                events_observed = [orig_label['label_event'] for orig_label in original_labels] 

                # 3. Define save path
                save_path = os.path.join(self.log_path, 'Kaplan_Meier_Plot_Test_Set.png')

                # 4. Call plotting function
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

        print("\n--- Testing Complete ---", flush=True)