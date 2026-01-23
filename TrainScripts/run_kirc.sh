source activate surv_pred
export CUDA_VISIBLE_DEVICES=2
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run051_medkgat_kimi_nl=3_lw=2-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run050_medkgat_kimi_nl=3_lw=1-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run052_medkgat_kimi_nl=3_lw=3-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run053_medkgat_kimi_nl=3_lw=4-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run054_medkgat_kimi_nl=3_lw=5-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run055_medkgat_kimi_nl=3_lw=6-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run056_medkgat_kimi_nl=3_lw=7-deepseek.sh
bash /home/Guanjq/NewWork/MedAlignFusion/TrainScripts/tcga_kirc/run057_medkgat_kimi_nl=3_lw=8-deepseek.sh