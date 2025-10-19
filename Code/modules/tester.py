import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tqdm
import numpy as np
import torch
import json

from datasets import GetDataLoader
from modules.model import GetModel
from modules.training_utils import Logger, load_model

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
        
        # --- Model Initialization and Loading ---
        self.model = GetModel(self.args).to(self.device)

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

        # --- Data Loader ---
        _, _, self.test_loader = GetDataLoader(self.args)
        assert self.test_loader is not None, "Test loader could not be created."
        
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
        all_logits, all_labels, all_losses = [], [], {}

        with torch.no_grad():
            pbar = tqdm.tqdm(total=len(self.test_loader), desc="Testing")
            for batch_data in self.test_loader:
                # Forward pass
                batch_size = len(batch_data['pid'])
                out = self.model(batch_size, batch_data)
                
                # Collect logits and labels
                all_logits.extend(out['logits'].detach().cpu().numpy().tolist())
                all_labels.extend(batch_data['labels'])
                
                # Collect loss values
                for key, value in out['losses'].items():
                    all_losses.setdefault(key, []).append(value.item())
                
                pbar.update(1)
            pbar.close()

        # --- Metric Calculation and Logging ---
        self.log_results(all_losses, all_logits, all_labels)

    def log_results(self, losses, logits, labels):
        """
        Calculates, logs, and prints the final evaluation metrics and losses.
        
        Args:
            losses (dict): A dictionary of lists containing loss values for each batch.
            logits (list): A list of model outputs (logits).
            labels (list): A list of ground truth labels.
        """
        loss_dict, metrics_dict = {}, {}
        
        # Calculate average loss for each loss component
        for key, value_list in losses.items():
            loss_dict[f"loss_{key}_test"] = np.mean(value_list)

        # Calculate performance metrics
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
