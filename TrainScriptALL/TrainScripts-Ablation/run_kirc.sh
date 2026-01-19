source activate surv_pred
export CUDA_VISIBLE_DEVICES=0
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_kirc/run021_medkgat_no_edge.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_kirc/run022_medkgat_no_loss.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_kirc/run023_medkgat_no_intra.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_kirc/run024_medkgat_no_inter.sh