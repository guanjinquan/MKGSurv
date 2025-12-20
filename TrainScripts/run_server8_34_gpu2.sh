export CUDA_VISIBLE_DEVICES=2
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run003_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run003_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run003_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run003_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run004_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run004_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run004_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run004_medkgat.sh