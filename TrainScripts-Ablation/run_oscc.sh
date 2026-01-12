source activate surv_pred
export CUDA_VISIBLE_DEVICES=2
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/oscc_inhouse/run021_medkgat_no_edge.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/oscc_inhouse/run022_medkgat_no_loss.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/oscc_inhouse/run023_medkgat_no_intra.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/oscc_inhouse/run024_medkgat_no_inter.sh