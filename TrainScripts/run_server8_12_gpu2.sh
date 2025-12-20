export CUDA_VISIBLE_DEVICES=2
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run001_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run001_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run001_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run001_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run002_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run002_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run002_medkgat.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run002_medkgat.sh