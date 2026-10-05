#!/usr/bin/env bash
# RELACE launch settings follow Appendix C of the accompanying manuscript.
set -euo pipefail

environment=${1:?Usage: run.sh alfworld|webshop [vllm] [Hydra overrides...]}
shift
ENGINE=vllm
if [[ ${1:-} == vllm ]]; then shift; fi
case "$environment" in
  alfworld) env_name=alfworld/AlfredTWEnv; max_steps=50; gpu_memory=0.3 ;;
  webshop) env_name=webshop; max_steps=15; gpu_memory=0.6 ;;
  *) echo "Unsupported environment: $environment" >&2; exit 2 ;;
esac

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
SEED=${SEED:-0}
NUM_GPUS=${NUM_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/relace}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-relace_${environment}_${MODEL_PATH##*/}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-console}
mini_batch=64
if [[ "$environment" == webshop && "$MODEL_PATH" == *7B* ]]; then mini_batch=128; fi

# Preserve externally supplied credentials; never put them in a command or log.
case "$LOGGER" in
  console) loggers='[console]' ;;
  wandb) : "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment for W&B logging}"; loggers='[console,wandb]' ;;
  *) echo 'LOGGER must be console or wandb' >&2; exit 2 ;;
esac
cd "$PROJECT_DIR"
cmd=("$PYTHON_BIN" "${PROJECT_DIR}/examples/relace_trainer/run_main_ppo_patched.py"
  algorithm.adv_estimator=relace
  "data.train_files=${DATA_DIR}/text/train.parquet"
  "data.val_files=${DATA_DIR}/text/test.parquet"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" "data.val_batch_size=${VAL_BATCH_SIZE}"
  data.max_prompt_length=8192 data.max_response_length=512
  data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$mini_batch}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE:-4}"
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 "actor_rollout_ref.rollout.name=${ENGINE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-$gpu_memory}"
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=False
  actor_rollout_ref.actor.use_invalid_action_penalty=False
  algorithm.use_kl_in_reward=False "algorithm.gamma=${RELACE_GAMMA:-0.6}"
  algorithm.gigpo.mode=mean_std_norm algorithm.gigpo.enable_similarity=True algorithm.gigpo.similarity_thresh=0.95
  "algorithm.relace.temp=${RELACE_TEMP:-1.0}" "algorithm.relace.clip_min=${RELACE_CLIP_MIN:-0.8}"
  "algorithm.relace.clip_max=${RELACE_CLIP_MAX:-1.2}" "algorithm.relace.omega=${RELACE_OMEGA:-1.0}"
  "algorithm.relace.gamma=${RELACE_GAMMA:-0.6}" "algorithm.relace.smooth=${RELACE_SMOOTH:-True}"
  "algorithm.relace.alpha=${RELACE_ALPHA:-0.5}" "algorithm.relace.no_harm=${RELACE_NO_HARM:-True}"
  "algorithm.relace.hindsight_template=${RELACE_HINDSIGHT_TEMPLATE:-$environment}"
  "env.env_name=${env_name}" "env.seed=${SEED}" "env.max_steps=${max_steps}"
  "env.rollout.n=${GROUP_SIZE}" env.history_length=2
  "env.resources_per_worker.num_cpus=${CPUS_PER_ENV:-0.1}"
  trainer.critic_warmup=0 "trainer.logger=${loggers}" trainer.project_name=relace
  "trainer.experiment_name=${EXPERIMENT_NAME}" "trainer.n_gpus_per_node=${NUM_GPUS}" trainer.nnodes=1
  trainer.save_freq=50 trainer.test_freq=5 "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.total_epochs=${TOTAL_EPOCHS:-250}" trainer.val_before_train=True
  "hydra.run.dir=${OUTPUT_DIR}/hydra" "hydra.sweep.dir=${OUTPUT_DIR}/hydra" "$@")

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "${PROJECT_DIR}/examples/data_preprocess/prepare.py" --mode text \
  --local_dir "$DATA_DIR" --train_data_size "$TRAIN_BATCH_SIZE" --val_data_size "$VAL_BATCH_SIZE"
"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
