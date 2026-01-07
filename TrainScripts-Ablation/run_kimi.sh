source activate surv_pred
export CUDA_VISIBLE_DEVICES=1
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/oscc_inhouse/run050_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_brca/run050_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_kirc/run050_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_luad/run050_medkgat_kimi.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts-Ablation/tcga_lusc/run050_medkgat_kimi.sh