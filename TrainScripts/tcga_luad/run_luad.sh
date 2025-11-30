set -e
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run001_group_medkgat_deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run002_group_medkgat_random.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run003_no_group_medkgat_deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run004_no_group_medkgat_random.sh