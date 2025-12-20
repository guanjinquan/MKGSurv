export CUDA_VISIBLE_DEVICES=0
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run005_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run005_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run005_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run005_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run006_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run006_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run006_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run006_medkgat.sh