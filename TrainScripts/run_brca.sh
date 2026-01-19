source activate surv_pred
export CUDA_VISIBLE_DEVICES=2
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run001_medkgat_qwen.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run002_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run003_medkgat_deepseek.sh