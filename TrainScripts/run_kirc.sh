source activate surv_pred
export CUDA_VISIBLE_DEVICES=1
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run001_medkgat_qwen.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run002_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run003_medkgat_deepseek.sh