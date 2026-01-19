source activate surv_pred
export CUDA_VISIBLE_DEVICES=1
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_lusc/run021_medkgat_no_edge.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_lusc/run022_medkgat_no_loss.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_lusc/run023_medkgat_no_intra.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_lusc/run024_medkgat_no_inter.sh