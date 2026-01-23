source activate surv_pred
export CUDA_VISIBLE_DEVICES=0
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run041_medkgat_survival.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run041_medkgat_survival.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run041_medkgat_survival.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run041_medkgat_survival.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run041_medkgat_survival.sh