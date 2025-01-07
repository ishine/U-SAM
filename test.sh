# !/bin/bash


python3 test.py \
    --config_path './conf/test.yaml' \
    --checkpoint './exp/pretrain_alldata_newprompt_1016/checkpoint-6744/pytorch_model.bin' \
    --test_data './data/test_audiocaps.scp' \
    --result_dir './decode/pretrain_alldata_newprompt_1016/checkpoint-6744/test_audiocaps'