#!/bin/bash

# Multi-policy OPD with an SGLang-backend teacher.
#
#   student          : trainable, paired Megatron + SGLang (m✓ s✓)
#   teacher_sglang   : frozen, standalone SGLang engine (m✗ s✓ trainable=false)
#                      serves per-token logprobs at rollout time.
#
# Cluster: 3 GPUs total — 1 student megatron + 1 student sglang + 1 teacher sglang.
#
# FRAMEWORK DEPENDENCIES (this script will fail today; see README and config.yaml):
#   - F1/F2/F3 from plan_colocate.md must land before the teacher_sglang
#     policy is correctly sized and spawned.
#   - A custom rollout function that queries teacher_sglang's engine and
#     stamps Sample.teacher_log_probs (see rollout_with_teacher_sglang.py
#     spec in this directory; not yet implemented).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Student and teacher share Megatron architecture (both Qwen3-0.6B); the
# teacher is a different fine-tune of the same base, served by SGLang.
source "${SCRIPT_DIR}/../../scripts/models/qwen3-0.6B.sh"

ROLLOUT_ARGS=(
   # Custom rollout: generates the student's response, then queries the
   # teacher_sglang engine for per-token logprobs of the response,
   # stamping Sample.teacher_log_probs. Spec at rollout_with_teacher_sglang.py.
   --custom-generate-function-path examples.multi_policy_opd_sglang.rollout_with_teacher_sglang.generate_with_teacher_sglang
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size 16
   --rollout-max-context-len 32768
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --balance-data
)

# Cluster sizing — derived from config.yaml (NO colocate):
#   actor_gpus   = sum(megatron_num_nodes * num_gpus_per_node) = 1 × 1 = 1
#                  (only student has a Megatron actor; teacher has megatron_num_nodes=0)
#   rollout_gpus = sum(sglang_num_nodes   * num_gpus_per_node) = 2 × 1 × 1 = 2
#                  (student sglang + teacher sglang)
#   total        = actor_gpus + rollout_gpus                   = 3
# Layout: GPU 0 = student megatron, GPU 1 = student sglang, GPU 2 = teacher sglang.
NUM_GPUS=3

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config.yaml"
   --save-interval 20
   --dump-details /tmp/multi_policy_opd_sglang/dump_details
)

EVAL_ARGS=(
   # --eval-interval 20
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-0.6B-opd-sglang
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}


####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
