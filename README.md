# Medical Knowledge-Guided Fusion of Holistic Hospital Data for Tumor Survival Analysis

This repository provides the implementation of MKGSurv for multimodal tumor survival analysis. The code is organized for TCGA cohorts and currently supports `tcga_luad`, `tcga_lusc`, `tcga_brca`, and `tcga_kirc`. The model combines clinical, genomic, pathology image, pathology text, treatment-related, and medical-knowledge features, depending on the selected run script.

## Environment

Use the prepared Conda environment before running training or evaluation:

```bash
conda activate mkgsurv
```

The expected environment includes PyTorch, scikit-survival, lifelines, and the other dependencies required by the training and testing scripts.

```bash
bash TrainScripts/tcga_lusc/run001_mkgsurv_kimi_pg.sh
bash TrainScripts/tcga_lusc/run002_mkgsurv_kimi_pgct.sh
```

## Data Organization

All data should be placed under `Data/`. Each cohort uses one folder:

```text
Data/
  TCGA-LUAD/
  TCGA-LUSC/
  TCGA-BRCA/
  TCGA-KIRC/
```

Each cohort folder should contain three main subdirectories:

```text
source/             raw clinical, genomic, report, and biospecimen files
processed/          model-ready features, labels, and 5-fold splits
processed_scripts/  scripts used to generate processed files
```

The training code reads from `processed/`. Typical required files include the patient split file, patient labels, RNA features, pathology image features for each fold, pathology and treatment text features, and medical knowledge features. The exact file names follow the existing dataset loaders in `Code/datasets/`.

The split file defines train, validation, and test patients for each fold. Image features are fold-specific; RNA, text, label, and knowledge files are shared across folds. Keep patient IDs consistent across modalities.

## Training and Evaluation

Training scripts are provided in `TrainScripts/<dataset>/`. For example, to run TCGA-LUSC with pathology image and genomics features:

```bash
bash TrainScripts/tcga_lusc/run001_mkgsurv_kimi_pg.sh
```

To run TCGA-LUSC with all available modalities:

```bash
bash TrainScripts/tcga_lusc/run002_mkgsurv_kimi_pgct.sh
```

Use the corresponding folder for other cohorts:

```bash
bash TrainScripts/tcga_luad/run001_mkgsurv_kimi_pg.sh
bash TrainScripts/tcga_brca/run002_mkgsurv_kimi_pgct.sh
bash TrainScripts/tcga_kirc/run002_mkgsurv_kimi_pgct.sh
```

Each script runs 5-fold training and testing through `Code/main_traintest_5fold.py`. The main arguments are already set inside the shell scripts, including dataset name, run ID, modalities, fusion type, knowledge source, learning rate, batch size, number of epochs, and MKGSurv-specific hyperparameters.

The run types differ mainly in active modalities. `run001` uses pathology image and genomics features. `run002` uses all configured modalities. To customize an experiment, copy a shell script and edit `RUN_ID`, `--modalities`, `--learning_rate`, `--num_epochs`, `--num_layers`, or `--kl_loss_weight`.

If a checkpoint already exists at:

```text
Checkpoints/<dataset>/<run_id>+mkgsurv_fusion/Fold*/valid_Best.pth
```

the trainer skips that fold and directly loads the saved checkpoint for evaluation. This behavior is useful for reproducing published results without retraining.

Checkpoint files are saved as PyTorch `state_dict` objects. The current fusion module name is `mkgsurv_fusion`, and released checkpoints should keep this directory naming.

## Outputs

Each fold stores results under its checkpoint directory. The key files are:

```text
valid_Best.pth        best validation checkpoint
test_metrics.json     survival metrics such as C-index and IPCW C-index
test_pid_to_data.json per-patient logits and labels
test_log.txt          testing configuration and logs
```

For reproducibility, keep the `Checkpoints/` structure unchanged when sharing pretrained weights.
