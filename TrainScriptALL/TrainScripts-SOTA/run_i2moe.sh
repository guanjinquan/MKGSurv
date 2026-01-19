source activate surv_pred
export CUDA_VISIBLE_DEVICES=2
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_lusc/run011_i2moe.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_luad/run011_i2moe.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_kirc/run011_i2moe.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-SOTA/tcga_brca/run011_i2moe.sh