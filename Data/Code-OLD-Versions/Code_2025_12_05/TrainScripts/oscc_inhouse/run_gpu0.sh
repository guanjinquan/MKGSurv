set -e
source activate surv_pred
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run004_all_medkgat_fusion_deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/oscc_inhouse/run001_image_pathology.sh