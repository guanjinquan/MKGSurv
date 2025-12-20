export CUDA_VISIBLE_DEVICES=1
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run010_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run010_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run010_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run010_medkgat.sh