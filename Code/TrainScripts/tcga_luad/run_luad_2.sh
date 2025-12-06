set -e
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run011_inter_intra_random.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run010_inter_intra_deepseek.sh