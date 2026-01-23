source activate surv_pred
export CUDA_VISIBLE_DEVICES=1
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run042_medkgat_relationship.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run042_medkgat_relationship.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run042_medkgat_relationship.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_lusc/run042_medkgat_relationship.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_brca/run042_medkgat_relationship.sh