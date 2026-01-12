source activate surv_pred
export CUDA_VISIBLE_DEVICES=0
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_luad/run021_medkgat_no_edge.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_luad/run022_medkgat_no_loss.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_luad/run023_medkgat_no_intra.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_luad/run024_medkgat_no_inter.sh