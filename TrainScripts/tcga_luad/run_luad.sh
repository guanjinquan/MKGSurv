set -e
source activate surv_pred

bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/kl_gated/run051_kl_gated_f0.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/kl_gated/run052_kl_gated_f1.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/kl_gated/run053_kl_gated_f2.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/kl_gated/run054_kl_gated_f3.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/kl_gated/run055_kl_gated_f4.sh

bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/healnet/run031_f0.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/healnet/run032_f1.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/healnet/run033_f2.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/healnet/run034_f3.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/healnet/run035_f4.sh

bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/msa_panther/run011_f0.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/msa_panther/run012_f1.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/msa_panther/run013_f2.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/msa_panther/run014_f3.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_luad/msa_panther/run015_f4.sh

