set -e
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run009_no_group_medkgat_zero.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run008_group_medkgat_zero.sh