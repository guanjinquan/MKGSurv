source activate surv_pred
export CUDA_VISIBLE_DEVICES=2
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_brca/run033_healnet.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_kirc/run033_healnet.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_luad/run033_healnet.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_lusc/run033_healnet.sh