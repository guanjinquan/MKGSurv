set -e
source activate surv_pred
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run001_image_pathology.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run002_rna.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run003_tabular_clinical.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run004_pre_op.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run005_treatment.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run006_post_op_pathology.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run007_tabular_post_op_pathology.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run008_all.sh
# bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/run009_text_pathology.sh