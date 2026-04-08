#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1


export WANDB_API_KEY=""

export WANDB_MODE=offline
export WANDB_DIR=/mnt/shared-storage-user/yuzhiyin/wandb_cache_qwen1.5b
export WANDB_CACHE_DIR=/mnt/shared-storage-user/yuzhiyin/wandb_cache_qwen1.5b

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="DM_20K"
BACKBONE="Qwen2.5-Math-1.5B"
ADVANTAGE="grpo"

K=3
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16
fi

EPISODE=1
DATA_TRAIN_BATCH_SIZE=8 # Rollout Prompt Num
# N_VOTES_PER_PROMPT=32
# N_SAMPLES_PER_PROMPT=32
N_VOTES_PER_PROMPT=8
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=2 # Actual mini batch size is MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT
MICRO_BATCH_SIZE=1

DATA_LOCAL_DIR="/mnt/shared-storage-user/yuzhiyin/EasyRL/verl/data"
BACKBONE_PATH="/mnt/shared-storage-user/ma4tool-shared/hug_ckpts/Qwen2.5-Math-1.5B"


ENTROPY_THRES=1.0

TARGET='gt'  # "gt" corresponds to GRPO, "semantic_entropy" to EMPO, and "voting" to TTRL


MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="${TARGET}-Len@${K}k-bz-${DATA_TRAIN_BATCH_SIZE}-thres-${ENTROPY_THRES}-Qwen2.5-1.5b"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
# OUTPUT_DIR="/fs-computility/prime/shared/zqy_models/checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"
OUTPUT_DIR="/mnt/shared-storage-user/ma4tool-shared/all_users_shared/yuzhiyin/checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}-round3-repeat1"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  reward_model.reward_kwargs.reward_type=$TARGET \
  reward_model.reward_kwargs.entropy_thres=$ENTROPY_THRES \
  data.train_files=["$DATA_LOCAL_DIR/deepmath_label.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/amc.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=500 \
  trainer.test_freq=100 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@" 
  # trainer.resume_mode="resume_path" \
  # trainer.resume_from_path="/fs-computility/MA4Tool/yuzhiyin/checkpoints/TTRL-verl/NM_20K-Qwen2.5-Math-7B/0714/semantic_entropy-Len@3k-bz-8-thres-1.0-grpo-143527/global_step_900"
  
echo "Output directory: $OUTPUT_DIR"