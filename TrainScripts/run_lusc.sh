source activate surv_pred
export CUDA_VISIBLE_DEVICES=1
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run001_medkgat_qwen.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run002_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run003_medkgat_deepseek.sh