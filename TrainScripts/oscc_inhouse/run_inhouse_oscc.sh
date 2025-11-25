set -e
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run009_text_pathology.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run007_tabular_post_pathology.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run006_post_op_pathology.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run005_treatment.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run003_tabular_clinical.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run002_text_clinical.sh