# python model_merger.py merge \
#     --backend fsdp \
#     --local_dir /fs-computility/prime/shared/zqy_models/checkpoints/TTRL-verl/NM_20K-Qwen2.5-Math-1.5B/0621/EMPO-Len@3k-bz-8-thres-1.0-grpo-085020/global_step_1800/actor \
#     --target_dir /fs-computility/prime/shared/zqy_models/checkpoints/hf_models/NM_20K-Qwen2.5-Math-1.5B/0621/EMPO-Len\@3k-bz-8-thres-1.0-grpo-091333/global_step_1800/
/mnt/shared-storage-user/yuzhiyin/conda_envs/verl_env/bin/python model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/checkpoints/TTRL-verl/NM_20K-Qwen2.5-Math-1.5B-Instruct/0219/gt-Len@3k-bz-8-thres-1.0-Qwen2.5-1.5b-grpo-134938-learninglike-10%-1/global_step_1874/actor \
    --target_dir /mnt/shared-storage-user/yuzhiyin/model