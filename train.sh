# !/bin/bash
export OMP_NUM_THREADS=4

# 单卡
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config_path './conf/pretrain.yaml' \
    --out_dir './exp/usam_alltask_1114_testrun' \
    --data_dir './data/all_data' 
    # --resume_checkpoint './exp/randominit_s2tt_v3_1023/checkpoint-17989'

# 多卡
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun \
#     --nnode 1 --nproc_per_node 8 --master_port=41595 train.py \
#     --config_path './conf/pretrain.yaml' \
#     --out_dir './exp/pretrain_aac_mc_v4upm_1105' \
#     --data_dir './data/aac_mc' 