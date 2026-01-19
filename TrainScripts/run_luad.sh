source activate surv_pred
export CUDA_VISIBLE_DEVICES=0
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run001_medkgat_qwen.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run002_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run003_medkgat_deepseek.sh