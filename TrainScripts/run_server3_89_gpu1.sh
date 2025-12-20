export CUDA_VISIBLE_DEVICES=1
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run008_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run008_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run008_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run008_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run009_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run009_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run009_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run009_medkgat.sh