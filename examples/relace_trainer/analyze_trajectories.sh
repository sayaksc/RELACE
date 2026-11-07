#!/usr/bin/env bash
# RELACE trajectory-analysis diagnostic: no training, one command.
#
#   CHECKPOINT=outputs/relace_alfworld_seed0/global_step_150 \
#     bash examples/relace_trainer/analyze_trajectories.sh alfworld
#
# Samples rollout groups from a fixed checkpoint until one has a success, a failure and a
# matched state shared across trajectories, then writes every RELACE intermediate for that
# group (plus a rho=1 recomputation) under $OUTPUT_DIR/trajectory_analysis/<timestamp>/.
#
#   CHECKPOINT    FSDP global_step_<N> directory saved by training; load it with the same
#                 NUM_GPUS and MODEL_PATH used for training. Unset = MODEL_PATH weights.
#                 (For a merged Hugging Face export, set MODEL_PATH to it instead.)
#   GROUP_SIZE    trajectories per group (default 8)
#   MAX_ATTEMPTS  groups to sample before giving up (default 20)
# All other run.sh variables (MODEL_PATH, NUM_GPUS, SEED, RELACE_*) and Hydra overrides apply.
set -euo pipefail

environment=${1:?Usage: analyze_trajectories.sh alfworld|webshop [Hydra overrides...]}
shift
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CHECKPOINT=${CHECKPOINT:-}
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT=$(cd "$CHECKPOINT" && pwd)
else
  CHECKPOINT=null
fi

export PROJECT_DIR
export TRAIN_BATCH_SIZE=1 VAL_BATCH_SIZE=1
export GROUP_SIZE=${GROUP_SIZE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-relace_trajectory_analysis_${environment}}
export OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}
export LOG_FILE=${LOG_FILE:-analysis.log}

exec bash "$SCRIPT_DIR/run.sh" "$environment" \
  trajectory_analysis.enable=True \
  "trajectory_analysis.checkpoint=${CHECKPOINT}" \
  "trajectory_analysis.group_size=${GROUP_SIZE}" \
  "trajectory_analysis.max_attempts=${MAX_ATTEMPTS:-20}" \
  trainer.val_before_train=False trainer.resume_mode=disable \
  "$@"
